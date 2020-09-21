import copy
import os
import math
import numpy as np

import Box2D
from Box2D.b2 import (fixtureDef, polygonShape, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import pyglet
from pyglet import gl

from szeth.envs.car_racing.car_dynamics_revised import Car
from szeth.utils.mprim import read_primitives


STATE_W = 96
STATE_H = 96

# VIEWER PARAMETERS
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000  # 1000
WINDOW_H = 1000  # 800
ZOOM = 2.7
ZOOM_FOLLOW = True

# TRACK PARAMETERS
# SCALE = 300.0
# SCALE = 300.0
SCALE = 450.0
TRACK_RAD = 900 / SCALE
# PLAYFIELD = 2000 / SCALE
PLAYFIELD = 2.0 * TRACK_RAD
BOUNDARY_LEFT = -PLAYFIELD + 0.7 * TRACK_RAD
BOUNDARY_RIGHT = PLAYFIELD - 0.7 * TRACK_RAD
BOUNDARY_DOWN = -PLAYFIELD + 0.7 * TRACK_RAD
BOUNDARY_UP = PLAYFIELD - 0.2 * TRACK_RAD

TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
# TRACK_WIDTH = 40/SCALE
TRACK_WIDTH = 150 / SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
ICE_COLOR = [165/255, 242/255, 243/255]

# SIMULATION PARAMETERS
FPS = 50.0

# DISCRETIZATION PARAMETERS
X_DISCRETIZATION = 100
Y_DISCRETIZATION = 100
THETA_DISCRETIZATION = 16
# X_BOUNDS = [-PLAYFIELD, PLAYFIELD]
# Y_BOUNDS = [-PLAYFIELD, PLAYFIELD]
X_BOUNDS = [-2.46, 3.46]
Y_BOUNDS = [-2.46, 2.46]
X_CELL_SIZE = (X_BOUNDS[1] - X_BOUNDS[0]) / X_DISCRETIZATION
Y_CELL_SIZE = (Y_BOUNDS[1] - Y_BOUNDS[0]) / Y_DISCRETIZATION
THETA_CELL_SIZE = (2 * math.pi) / THETA_DISCRETIZATION

NUM_CHECKPOINTS = 2

NUM_FULLY_ICY_TILES = 2
ICY_TILES_MAX_LENGTH = 10
ICY_TILES_MIN_LENGTH = 5

NUM_PARTIALLY_ICY_TILES = 3
PARTIAL_ICY_TILES_MIN_LENGTH = 3
PARTIAL_ICY_TILES_MAX_LENGTH = 6


def construct_params():
    params = {
        'PLAYFIELD': PLAYFIELD,
        'X_BOUNDS': X_BOUNDS,
        'Y_BOUNDS': Y_BOUNDS,
        'X_DISCRETIZATION': X_DISCRETIZATION,
        'Y_DISCRETIZATION': Y_DISCRETIZATION,
        'THETA_DISCRETIZATION': THETA_DISCRETIZATION
    }
    return params


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0/len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, seed=0,
                 verbose=1,
                 friction_params=None,
                 read_mprim=True,
                 no_boundary=False,
                 variable_speed=False):
        EzPickle.__init__(self)
        self.set_seed(seed)
        if friction_params is None:
            self.ice_friction = 1.0
            self.ice_color = ROAD_COLOR
        else:
            if 'ice' in friction_params:
                self.ice_friction = friction_params['ice']
                self.ice_color = ICE_COLOR
            else:
                self.ice_friction = 1.0
                self.ice_color = ROAD_COLOR
        self.grass_friction = 1.0
        self.no_boundary = no_boundary
        self.variable_speed = variable_speed
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World(
            (0, 0), contactListener=self.contactListener_keepref)
        # self.create_boundary()
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.discrete_road_poly = set()
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))
        self.boundary_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))
        self.boundary_tile.density = 1000
        self.boundary_tiles = []

        self.action_space = spaces.MultiDiscrete([2, 3])
        self.observation_space = spaces.Dict(
            {'observation': spaces.MultiDiscrete([
                X_DISCRETIZATION,
                Y_DISCRETIZATION,
                THETA_DISCRETIZATION])})

        if read_mprim:
            mprim_path = os.path.join(os.environ['HOME'],
                                      'workspaces/szeth_ws/src/szeth/save/car.mprim')
            params = construct_params()
            self.motion_primitives, self.params = read_primitives(
                mprim_path, params)

    def create_boundary(self):
        self.boundary = self.world.CreateStaticBody(
            position=(0, 0),
            angle=(0),
            fixtures=[
                fixtureDef(shape=polygonShape(box=((BOUNDARY_UP -
                                                    BOUNDARY_DOWN)//2,
                                                   0.1,
                                                   (0.3 * TRACK_RAD,
                                                    BOUNDARY_LEFT),
                                                   0))),
                fixtureDef(shape=polygonShape(box=(0.1,
                                                   (BOUNDARY_RIGHT -
                                                    BOUNDARY_LEFT)//2 +
                                                   0.3 * TRACK_RAD,
                                                   (BOUNDARY_UP, 0),
                                                   0))),
                fixtureDef(shape=polygonShape(box=((BOUNDARY_UP -
                                                    BOUNDARY_DOWN)//2,
                                                   0.1,
                                                   (0.3 * TRACK_RAD,
                                                    BOUNDARY_RIGHT),
                                                   0))),
                fixtureDef(shape=polygonShape(box=(0.1,
                                                   (BOUNDARY_RIGHT -
                                                    BOUNDARY_LEFT)//2 +
                                                   0.3 * TRACK_RAD,
                                                   (BOUNDARY_DOWN +
                                                    0.05 * TRACK_RAD, 0),
                                                   0)))
            ]
        )
        self.boundary.color = (1, 1, 1)

    def draw_boundary(self, viewer):
        # for f in self.boundary.fixtures:
        #     trans = f.body.transform
        #     path = [trans * v for v in f.shape.vertices]
        #     viewer.draw_polygon(path, color=self.boundary.color)

        for t in self.boundary_tiles:
            for f in t.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=t.color)

    def draw_checkpoints(self, viewer):
        for checkpoint in self.checkpoint_bodies:
            for f in checkpoint.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=checkpoint.color)

    def convert_xytheta_to_discrete(self, x, y, theta):
        # Clip continuous values so that they are in the grid
        x_clipped = min(max(x, X_BOUNDS[0]), X_BOUNDS[1])
        y_clipped = min(max(y, Y_BOUNDS[0]), Y_BOUNDS[1])

        x_shifted = x_clipped - X_BOUNDS[0]
        y_shifted = y_clipped - Y_BOUNDS[0]

        x_discrete = (
            x_shifted * X_DISCRETIZATION) // (X_BOUNDS[1] - X_BOUNDS[0])
        y_discrete = (
            y_shifted * Y_DISCRETIZATION) // (Y_BOUNDS[1] - Y_BOUNDS[0])

        while theta < 0:
            theta = theta + 2 * math.pi
        while theta > 2 * math.pi:
            theta = theta - 2 * math.pi

        theta_discrete = (theta * THETA_DISCRETIZATION) // (2 * math.pi)

        # Clip discrete values
        x_discrete = min(max(x_discrete, 0), X_DISCRETIZATION-1)
        y_discrete = min(max(y_discrete, 0), Y_DISCRETIZATION-1)
        theta_discrete = min(max(theta_discrete, 0), THETA_DISCRETIZATION-1)

        # compute discretization error
        (x_recomputed,
         y_recomputed,
         theta_recomputed) = self.convert_discrete_to_xytheta(x_discrete,
                                                              y_discrete,
                                                              theta_discrete)
        # discretization_error = np.linalg.norm(
        #     np.array([x, y, theta]) - np.array([x_recomputed,
        #                                         y_recomputed,
        #                                         theta_recomputed])
        # )
        discretization_error = {'x': abs(x - x_recomputed),
                                'y': abs(y - y_recomputed),
                                'theta': abs(theta - theta_recomputed)}

        return x_discrete, y_discrete, theta_discrete, discretization_error

    def convert_discrete_to_xytheta(self, xd, yd, thetad):
        x_cell_size = (X_BOUNDS[1] - X_BOUNDS[0]) / X_DISCRETIZATION
        y_cell_size = (Y_BOUNDS[1] - Y_BOUNDS[0]) / Y_DISCRETIZATION
        theta_cell_size = (2 * math.pi) / THETA_DISCRETIZATION

        x = xd * x_cell_size + x_cell_size / 2 + X_BOUNDS[0]
        y = yd * y_cell_size + y_cell_size / 2 + Y_BOUNDS[0]
        theta = thetad * theta_cell_size + theta_cell_size / 2

        return x, y, theta

    def set_seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            # alpha = 2*math.pi*c/CHECKPOINTS + \
            #     self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            # rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            alpha = 2*math.pi*c / CHECKPOINTS
            rad = TRACK_RAD
            if c == 0:
                alpha = 0
                rad = 1.5*TRACK_RAD
            if c == CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.5*TRACK_RAD
            checkpoints.append(
                (alpha, rad*math.cos(alpha), rad*math.sin(alpha)))

        # print "\n".join(str(h) for h in checkpoints)
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(
                        checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5*math.pi:
                beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi:
                beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append((alpha, prev_beta*0.5 + beta*0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i -
                                                                          1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" %
                  (i1, i2, i2-i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x*(track[0][2] - track[-1][2])) +
            np.square(first_perp_y*(track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        length_of_track = len(track)
        # CHECKPOINTS
        num_checkpoints = NUM_CHECKPOINTS
        # old version
        # tiles_per_checkpoint = length_of_track // num_checkpoints
        # idx_checkpoints = [idx for idx in range(
        #     len(track)) if idx % tiles_per_checkpoint == 5]
        # remaining_idx = [idx for idx in range(
        #     len(track)) if idx not in idx_checkpoints]
        # new version
        quarter_length = length_of_track // 4
        half_length = length_of_track // 2
        assert num_checkpoints < 4
        idx_checkpoints = [
            10 + i * half_length for i in range(num_checkpoints)]
        remaining_idx = [idx for idx in range(
            len(track)) if idx not in idx_checkpoints]
        self.checkpoint_bodies = []
        self.checkpoints = []
        # FULLY ICY TILES
        num_fully_icy_tiles = NUM_FULLY_ICY_TILES
        # old version
        idx_fully_icy_tiles = []
        for t in range(num_fully_icy_tiles):
            idx = self.np_random.choice(remaining_idx)
            length = self.np_random.randint(ICY_TILES_MIN_LENGTH,
                                            ICY_TILES_MAX_LENGTH+1)
            for i in range(idx, idx + length):
                if i in remaining_idx:
                    idx_fully_icy_tiles.append(i)
            remaining_idx = [
                j for j in remaining_idx if j not in idx_fully_icy_tiles]
        # new version
        # start_idx = self.np_random.randint(20)
        # idx_fully_icy_tiles = [
        #     start_idx + quarter_length + i for i in range(icy_tiles_length)]
        # remaining_idx = [
        #     idx for idx in remaining_idx if idx not in idx_fully_icy_tiles]
        # PARTIALLY ICY TILES
        num_partially_icy_tiles = NUM_PARTIALLY_ICY_TILES
        idx_partially_icy_tiles = []
        eta1s = {}
        eta2s = {}
        for t in range(num_partially_icy_tiles):
            idx = self.np_random.choice(remaining_idx)
            eta1 = self.np_random.uniform(0, 1)
            eta2 = self.np_random.uniform(eta1, 1)
            length = self.np_random.randint(PARTIAL_ICY_TILES_MIN_LENGTH,
                                            PARTIAL_ICY_TILES_MAX_LENGTH+1)
            for i in range(idx, idx + length):
                if i in remaining_idx:
                    idx_partially_icy_tiles.append(i)
                    eta1s[i] = eta1
                    eta2s[i] = eta2
                remaining_idx = [
                    j for j in remaining_idx if j not in idx_partially_icy_tiles]
        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]

            if i in idx_checkpoints:
                # Add checkpoint
                checkpoint = self.world.CreateStaticBody(
                    fixtures=[
                        fixtureDef(shape=polygonShape(box=(0.05, 0.05,
                                                           (x1,
                                                            y1),
                                                           0)))
                    ]
                )
                checkpoint.fixtures[0].sensor = True
                checkpoint.color = (0, 0, 1)
                self.checkpoint_bodies.append(checkpoint)
                betaavg = (beta1 + beta2) / 2
                self.checkpoints.append((x1, y1, betaavg))

            road1_l = np.array((x1 - TRACK_WIDTH*math.cos(beta1),
                                y1 - TRACK_WIDTH*math.sin(beta1)))
            road1_r = np.array((x1 + TRACK_WIDTH*math.cos(beta1),
                                y1 + TRACK_WIDTH*math.sin(beta1)))
            road2_l = np.array((x2 - TRACK_WIDTH*math.cos(beta2),
                                y2 - TRACK_WIDTH*math.sin(beta2)))
            road2_r = np.array((x2 + TRACK_WIDTH*math.cos(beta2),
                                y2 + TRACK_WIDTH*math.sin(beta2)))
            vertices = [tuple(road1_l),
                        tuple(road1_r),
                        tuple(road2_r),
                        tuple(road2_l)]
            # Create left boundary tile
            gamma1 = road1_r - road1_l
            gamma2 = road2_r - road2_l
            vertices_left_boundary = [tuple(road1_l - 0.4 * gamma1),
                                      tuple(road1_l - 0.3 * gamma1),
                                      tuple(road2_l - 0.3 * gamma2),
                                      tuple(road2_l - 0.4 * gamma2)]
            self.boundary_tile.shape.vertices = vertices_left_boundary
            if not self.no_boundary:
                if i not in np.arange(284, 299) and i not in np.arange(13, 26):
                    t = self.world.CreateStaticBody(
                        fixtures=self.boundary_tile)
                    t.color = [1, 1, 1]
                    self.boundary_tiles.append(t)
            # Create right boundary tile
            vertices_right_boundary = [tuple(road1_r + 0.3 * gamma1),
                                       tuple(road1_r + 0.4 * gamma1),
                                       tuple(road2_r + 0.4 * gamma2),
                                       tuple(road2_r + 0.3 * gamma2)]
            self.boundary_tile.shape.vertices = vertices_right_boundary
            if not self.no_boundary:
                t = self.world.CreateStaticBody(fixtures=self.boundary_tile)
                t.color = [1, 1, 1]
                self.boundary_tiles.append(t)
            # Create the road tile
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01*(i % 3)
            if i not in idx_fully_icy_tiles:
                # Road tile
                t.color = [ROAD_COLOR[0] + c,
                           ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
                t.road_friction = 1.0
                if i in idx_partially_icy_tiles:
                    eta1, eta2 = eta1s[i], eta2s[i]
                    gamma1 = road1_r - road1_l
                    gamma2 = road2_r - road2_l
                    ice_vertices = [tuple(road1_l + eta1 * gamma1),
                                    tuple(road1_l + eta2 * gamma1),
                                    tuple(road2_l + eta2 * gamma2),
                                    tuple(road2_l + eta1 * gamma2)]
                    self.fd_tile.shape.vertices = ice_vertices
                    ice_t = self.world.CreateStaticBody(fixtures=self.fd_tile)
                    ice_t.userData = ice_t
                    ice_t.color = self.ice_color
                    ice_t.road_friction = self.ice_friction
                    ice_t.road_visited = False
                    ice_t.fixtures[0].sensor = True
            else:
                # Fully Icy tile
                t.color = self.ice_color
                t.road_friction = self.ice_friction
            t.road_visited = False
            t.fixtures[0].sensor = True

            # Add discrete poly
            self.add_to_discrete_poly([road1_l, road1_r, road2_r, road2_l])

            self.road_poly.append(
                ([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if i in idx_partially_icy_tiles:
                self.road_poly.append((ice_vertices,
                                       ice_t.color))
                self.road.append(ice_t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1),
                        y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side*(TRACK_WIDTH+BORDER)*math.cos(beta1),
                        y1 + side*(TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2),
                        y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side*(TRACK_WIDTH+BORDER)*math.cos(beta2),
                        y2 + side*(TRACK_WIDTH+BORDER)*math.sin(beta2))
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1,
                                                                             0,
                                                                             0
                                                                             ))
                )
        self.track = track
        return True

    def reset(self, car_position_discrete=None):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        self.set_seed(self.seed)
        # Generate track
        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many of this messages)")

        # Generate car
        if car_position_discrete is None:
            theta, x, y = self.track[0][1:4]
            xd, yd, thetad, _ = self.convert_xytheta_to_discrete(x, y, theta)
            x, y, theta = self.convert_discrete_to_xytheta(xd, yd, thetad)
            self.car = Car(
                self.world, theta, x, y, variable_speed=self.variable_speed)
        else:
            x, y, theta = self.convert_discrete_to_xytheta(
                *car_position_discrete)
            self.car = Car(self.world, theta, x, y,
                           variable_speed=self.variable_speed)

        # Generate discrete cost map
        self._generate_cost_map()
        return self.step(None)[0]

    def reset_car_position(self, car_position_discrete):
        assert self.road is not None, "Reset must be called first"
        x, y, theta = self.convert_discrete_to_xytheta(*car_position_discrete)
        self.car.destroy()
        self.car = Car(self.world, theta, x, y,
                       variable_speed=self.variable_speed)
        # self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        # err = max(abs(self.car.hull.position[0] - x),
        #           abs(self.car.hull.position[1] - y),
        #           abs(self.car.hull.angle - theta))
        # while err > 0.01:
        #     self.car.hull.position = (x, y)
        #     self.car.hull.angle = theta
        #     self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        #     err = max(abs(self.car.hull.position[0] - x),
        #               abs(self.car.hull.position[1] - y),
        #               abs(self.car.hull.angle - theta))

        return self.step(None)[0]

    def add_to_discrete_poly(self, vertices):
        road1_l, road1_r, road2_r, road2_l = vertices
        current_point = road1_l.copy()
        eta_current = road1_r - road1_l
        eta_disp = 0
        xi_current = road2_l - road1_l
        phi_current = road2_r - road1_r
        xi_disp = 0
        while True:
            xd, yd, _, _ = self.convert_xytheta_to_discrete(
                current_point[0], current_point[1], 0)
            if (xd, yd) not in self.discrete_road_poly:
                self.discrete_road_poly.add((xd, yd))
            # Move in x
            current_point = current_point + 0.05 * eta_current
            eta_disp += 0.05
            # Check bounds
            if eta_disp > 1:
                # move to next
                xi_disp += 0.05
                current_point = road1_l + xi_disp * xi_current
                eta_current = (road1_r + xi_disp * phi_current) - current_point
                eta_disp = 0

            if xi_disp > 1:
                # Finished
                break

        return

    def _generate_cost_map(self):
        # Go through all the discrete cells in the discrete graph
        # Convert to the continuous space to get position
        # of the cell center
        # Figure out if the center of the cell is on road or grass
        # If on grass, have high cost
        # If on road, have low cost
        self.cost_map = np.ones((X_DISCRETIZATION,
                                 Y_DISCRETIZATION)) * 100
        for xd in range(X_DISCRETIZATION):
            for yd in range(Y_DISCRETIZATION):
                if (xd, yd) in self.discrete_road_poly:
                    # ROAD
                    self.cost_map[xd, yd] = 1

        return

    def get_actions(self):
        actions = []
        for speed in range(2):
            for steering_angle in range(3):
                actions.append((speed, steering_angle))

        return np.array(actions)

    def get_motion_primitives(self):
        return self.motion_primitives

    def get_observation(self):
        x, y = self.car.hull.position
        theta = self.car.hull.angle
        while theta < 0:
            theta += 2 * math.pi
        while theta > 2 * math.pi:
            theta -= 2 * math.pi
        xd, yd, thetad, d_error = self.convert_xytheta_to_discrete(x, y, theta)
        state = np.array([xd, yd, thetad], dtype=np.int32)
        obs = self.observation_from_state(
            state, np.array([x, y, theta]), d_error)
        return obs

    def step_mprim_execution(self, mprim, render=False):
        raise Exception('Is this even used?')
        # Get current heading
        obs = copy.deepcopy(self.get_observation())
        assert mprim.initial_heading == obs['observation'][2], "Incorrect mprim used"
        # Get true states
        true_states = mprim.true_states
        reward = 0
        idx = 1
        substeps = 5
        for state in true_states[1:]:
            self.t += self.car.move_to_state(
                obs['true_state'] + state, dt=1.0/(10 * FPS))
            print('--------------')
            curr_obs = self.get_observation()
            reward = reward - self.cost_map[curr_obs['observation'][0],
                                            curr_obs['observation'][1]]
            if render:
                self.render()
            obs = copy.deepcopy(curr_obs)
            idx += 1

        done = False
        obs = self.get_observation()
        return obs, reward, done, {}

    def step_mprim(self, mprim, render=False):
        # Get current heading
        obs = copy.deepcopy(self.get_observation())
        assert mprim.initial_heading == obs['observation'][2], "Incorrect mprim used"
        # Get mprim controls
        seq = mprim.seq
        reward = -self.get_cost(obs, mprim)
        for control in seq:
            self.car.step(control)
            self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
            self.t += 1.0 / FPS
            curr_obs = self.get_observation()
            # reward = reward - self.cost_map[curr_obs['observation'][0],
            #                                 curr_obs['observation'][1]]
            if render:
                self.render()
            obs = copy.deepcopy(curr_obs)

        done = False
        obs = self.get_observation()
        # Check if discretization error is high, in which case run a local controller
        # to snap to the cell center
        # TODO: For now, a bizarre hack to simply snap it to the cell center
        self.reset_car_position(obs['observation'])
        obs = self.get_observation()
        return obs, reward, done, {}

    def get_cost(self, obs, mprim):
        cost = 0
        for discrete_state in mprim.discrete_states:
            xd, yd, thetad = discrete_state
            current_observation = np.array(
                [max(min(obs['observation'][0] + xd, X_DISCRETIZATION-1), 0),
                 max(min(obs['observation'][1] + yd, Y_DISCRETIZATION-1), 0),
                 thetad], dtype=int)
            cost_step = self.cost_map[current_observation[0],
                                      current_observation[1]]
            cost += cost_step

        return cost

    def check_goal(self, obs, checkpoint):
        # Convert checkpoint to discrete state
        xd, yd, _, _ = self.convert_xytheta_to_discrete(
            checkpoint[0], checkpoint[1], math.pi - checkpoint[2])
        # Within a L1 ball of the checkpoint
        manhattan_dist = abs(xd - obs[0]) + abs(yd - obs[1])
        # euclidean_dist = np.sqrt((xd - obs[0])**2 + (yd - obs[1])**2)
        if manhattan_dist <= 2:
            # if euclidean_dist <= 1:
            return True
        return False

    def step(self, control):
        obs = self.get_observation()
        if control is not None:
            self.car.step(control)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        curr_obs = self.get_observation()
        reward = 0
        if control is not None:
            reward = -self.cost_map[curr_obs['observation'][0],
                                    curr_obs['observation'][1]]
        done = False

        obs = copy.deepcopy(curr_obs)

        return obs, reward, done, {}

    # def compute_reward(self, prev_obs, curr_obs):
    #     # Cost is distance travelled in XY space
    #     prev_x, prev_y, _ = prev_obs['true_state']
    #     x, y, _ = curr_obs['true_state']
    #     reward = - \
    #         np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
    #     # reward = -1
    #     # if car is completely on grass, add a small cost
    #     if self.grass_penalty:
    #         road = False
    #         num_wheels_on_road = 0
    #         for w in self.car.wheels:
    #             if len(w.tiles) > 0:
    #                 num_wheels_on_road += 1
    #         if num_wheels_on_road < 4:
    #             reward = 20 * reward
    #     return reward

    def observation_from_state(self, state, true_state, discretization_error):
        observation = {}
        observation['observation'] = state.copy()
        observation['true_state'] = true_state.copy()
        observation['discretization_error'] = discretization_error
        return observation

    def set_sim_state(self, true_state):
        x, y, theta = true_state
        self.car.hull.position = Box2D.b2Vec2(x, y)
        self.car.hull.angle = theta
        return True

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20, y=WINDOW_H*2.5/40.00,
                                                 anchor_x='left',
                                                 anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE * \
        #     min(self.t, 1)   # Animate zoom first second
        # zoom = 0.7 * SCALE
        zoom = 0.3 * SCALE
        # zoom_state = ZOOM*SCALE*STATE_W/WINDOW_W
        # zoom_video = ZOOM*SCALE*VIDEO_W/WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        # if np.linalg.norm(vel) > 0.5:
        #     angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        # self.transform.set_translation(
        #     WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) -
        #                   scroll_y*zoom*math.sin(angle)),
        #     WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) +
        #                   scroll_y*zoom*math.cos(angle)))
        # self.transform.set_rotation(angle)
        self.transform.set_translation(WINDOW_W/2, WINDOW_H/2)

        self.car.draw(self.viewer, mode != "state_pixels")
        self.draw_boundary(self.viewer)
        self.draw_checkpoints(self.viewer)

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view(
                ).backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k*x + k, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + k, 0)
                gl.glVertex3f(k*x + k, k*y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h, 0)
            gl.glVertex3f((place+0)*s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, 4*h, 0)
            gl.glVertex3f((place+val)*s, 4*h, 0)
            gl.glVertex3f((place+val)*s, 2*h, 0)
            gl.glVertex3f((place+0)*s, 2*h, 0)
        true_speed = np.sqrt(np.square(
            self.car.hull.linearVelocity[0]) + np.square(
                self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02*true_speed, (1, 1, 1))
        # ABS sensors
        vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0, 0, 1))
        vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01*self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0*self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8*self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


def make_car_racing_env(seed=0, friction_params=None,
                        variable_speed=False):
    return CarRacing(seed=seed,
                     friction_params=friction_params,
                     read_mprim=True,
                     variable_speed=variable_speed)


if __name__ == '__main__':
    from pyglet.window import key
    a = np.array([1, 1], dtype=np.int32)

    def key_press(k, mode):
        global a
        if k == key.LEFT:
            a[1] = 0
        if k == key.RIGHT:
            a[1] = 2
        if k == key.UP:
            a[0] = 1
        if k == key.DOWN:
            a[0] = 0

    def key_release(k, mode):
        global a
        if k == key.LEFT:
            a[1] = 1
        if k == key.RIGHT:
            a[1] = 1

    env = CarRacing(seed=0, read_mprim=False)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    isopen = True
    while isopen:
        obs = env.reset()
        origin = obs['observation']
        origin[2] = 2
        while True:
            s, r, done, info = env.step(a)
            isopen = env.render()
            if isopen is False:
                break

    env.close()
