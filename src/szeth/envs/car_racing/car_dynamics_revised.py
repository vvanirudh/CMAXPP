'''
Revised car dynamics modeled as a Reed-Shepps car
'''
import numpy as np
import math
from Box2D.b2 import (fixtureDef, polygonShape, revoluteJointDef)

# SIZE = 0.0008
SIZE = 0.0005

# INERTIAL PARAMETERS
ENGINE_POWER = 1e8 * SIZE**2
WHEEL_MOMENT_OF_INERTIA = 4e3 * SIZE**2
FRICTION_LIMIT = 1e6 * SIZE**2

# WHEEL DIMENSIONS
WHEEL_R = 27
WHEEL_W = 14
WHEEL_POLY = [
    (-WHEEL_W, +WHEEL_R), (+WHEEL_W, +WHEEL_R),
    (+WHEEL_W, -WHEEL_R), (-WHEEL_W, -WHEEL_R)
]

# WHEEL POSITIONS
WHEELPOS = [
    (-55, +80), (+55, +80),
    (-55, -82), (+55, -82)
]

# CAR POLY
HULL_POLY1 = [
    (-60, +130), (+60, +130),
    (+60, +110), (-60, +110)
]
HULL_POLY2 = [
    (-15, +120), (+15, +120),
    (+20, +20), (-20,  20)
]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20)
]
HULL_POLY4 = [
    (-50, -120), (+50, -120),
    (+50, -90),  (-50, -90)
]

# COLORS
WHEEL_COLOR = (0.0, 0.0, 0.0)
WHEEL_WHITE = (0.3, 0.3, 0.3)
MUD_COLOR = (0.4, 0.4, 0.0)

# DISCRETE CONTROL
# STEERING_ANGLES = (+0.4, 0, -0.4)
STEERING_ANGLES = (+0.6, 0, -0.6)
# SPEEDS = (-20, +20)
# SPEEDS = (-5, +5)
SPEEDS = (-2, +2)
TURNING_SPEEDS = (-1.5, +1.5)


class Car:
    def __init__(self, world, init_angle, init_x, init_y,
                 variable_speed=False):
        self.world = world
        self.grass_friction = 1.0
        self.variable_speed = variable_speed

        # Create car hull
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=(init_angle),
            fixtures=[
                fixtureDef(shape=polygonShape(
                    vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY1]),
                    density=1.0),
                fixtureDef(shape=polygonShape(
                    vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY2]),
                    density=1.0),
                fixtureDef(shape=polygonShape(
                    vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY3]),
                    density=1.0),
                fixtureDef(shape=polygonShape(
                    vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY4]),
                    density=1.0)
            ]
        )
        self.hull.color = (0.8, 0, 0)

        # Create wheels
        self.wheels = []
        for wx, wy in WHEELPOS:
            # First two wheels are front wheels
            # Next two are rear wheels
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * SIZE, init_y + wy * SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in WHEEL_POLY]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0
                )
            )
            w.front = True if wy > 0 else False
            w.wheel_rad = WHEEL_R * SIZE
            w.color = WHEEL_COLOR
            w.speed = 0.0  # linear speed
            w.steering_angle = 0.0  # steering angle
            w.omega = 0.0  # angular speed
            # Skid
            w.skid_particle = None
            w.skid_start = None
            # Revolute joint to model steering
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx * SIZE, wy * SIZE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * SIZE**2,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4)
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)

        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def move_to_state(self, goal_state, dt):
        t = 0
        K_rho = 1
        K_alpha = 0.1
        K_beta = 0.1
        current_state_x, current_state_y = self.hull.position
        current_state_theta = self.hull.angle
        while current_state_theta > np.pi:
            current_state_theta -= 2 * np.pi
        while current_state_theta < -np.pi:
            current_state_theta += 2 * np.pi
        goal_state_x, goal_state_y, goal_state_theta = goal_state
        while goal_state_theta > np.pi:
            goal_state_theta -= 2 * np.pi
        while goal_state_theta < -np.pi:
            goal_state_theta += 2 * np.pi

        xdiff = goal_state_x - current_state_x
        ydiff = goal_state_y - current_state_y
        rho = np.hypot(xdiff, ydiff)

        while rho > 1e-2:

            alpha = (np.arctan2(ydiff, xdiff) -
                     current_state_theta + np.pi) % (2 * np.pi) - np.pi
            beta = (goal_state_theta - current_state_theta -
                    alpha + np.pi) % (2 * np.pi) - np.pi

            speed = K_rho * rho
            angular_speed = K_alpha * alpha + K_beta * beta

            # theta_new = current_state_theta + angular_speed * dt
            # x_new = current_state_x + speed * np.cos(theta_new) * dt
            # y_new = current_state_y + speed * np.sin(theta_new) * dt

            steering_angle = (np.arctan2(angular_speed, speed) +
                              np.pi) % (2 * np.pi) - np.pi
            steering_angle = max(min(steering_angle, 0.6), -0.6)

            print(speed, steering_angle)
            # print(rho, alpha, beta)

            if alpha > np.pi/2 or alpha < -np.pi/2:
                speed = -speed

            # Apply force to car
            for w in self.wheels:
                # Rotate wheels
                if w.front:
                    w.joint.motorSpeed = np.sign(
                        steering_angle - w.joint.angle) * min(
                            100.0, 50.0 * abs(steering_angle - w.joint.angle))
                else:
                    w.joint.motorSpeed = np.sign(
                        0 - w.joint.angle) * min(100.0, 50.0 * abs(0 - w.joint.angle))

                # TODO: No friction for now
                forw = w.GetWorldVector((0, 1))
                side = w.GetWorldVector((1, 0))
                v = w.linearVelocity
                vf = np.dot(forw, v)  # forward speed
                vs = np.dot(side, v)  # side speed

                f_force = speed - vf
                p_force = -vs

                f_force *= 205e3 * SIZE**2
                p_force *= 205e3 * SIZE**2
                force = np.hypot(f_force, p_force)
                friction_limit = FRICTION_LIMIT

                # Force more than max friction
                if abs(force) > friction_limit:
                    f_force /= force
                    p_force /= force
                    force = friction_limit
                    f_force *= force
                    p_force *= force

                w.ApplyForceToCenter((
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1]),
                    True)

            self.world.Step(dt, 6 * 30, 2 * 30)
            t += dt
            current_state_x, current_state_y = self.hull.position
            current_state_theta = self.hull.angle
            while current_state_theta > np.pi:
                current_state_theta -= 2 * np.pi
            while current_state_theta < -np.pi:
                current_state_theta += 2 * np.pi
            goal_state_x, goal_state_y, goal_state_theta = goal_state
            while goal_state_theta > np.pi:
                goal_state_theta -= 2 * np.pi
            while goal_state_theta < -np.pi:
                goal_state_theta += 2 * np.pi

            xdiff = goal_state_x - current_state_x
            ydiff = goal_state_y - current_state_y
            rho = np.hypot(xdiff, ydiff)
        return t

    def step(self, control):
        '''
        control is a 2D vector consisting of [speed, steering_angle]
        '''
        speed, steering_angle = control
        assert speed < len(SPEEDS) and speed >= 0
        assert steering_angle < len(
            STEERING_ANGLES) and steering_angle >= 0
        for w in self.wheels:
            # If front wheel, steer the wheel
            if w.front:
                # w.angle = STEERING_ANGLES[steering_angle]
                steer = STEERING_ANGLES[steering_angle]
            else:
                steer = 0.0
            dir = np.sign(steer - w.joint.angle)
            val = abs(steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val,
                                           100.0)

            # Position => friction limit
            if len(w.tiles) == 0:
                # Wheel is on grass
                grass = True
                friction_limit = self.grass_friction * FRICTION_LIMIT
            else:
                # Wheel is not on grass
                grass = False
                friction_limit = np.inf
                for tile in w.tiles:
                    friction_limit = min(
                        friction_limit, FRICTION_LIMIT * tile.road_friction)

            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = np.dot(forw, v)  # forward speed
            vs = np.dot(side, v)  # side speed

            if self.variable_speed:
                # In case of variable speed,
                # we cannot turn at a high speed
                car_speed = SPEEDS[speed]
                if steer != 0:
                    # Car is turning
                    car_speed = TURNING_SPEEDS[speed]
            else:
                car_speed = SPEEDS[speed]

            f_force = car_speed - vf
            p_force = -vs

            f_force *= 205e3 * SIZE**2
            p_force *= 205e3 * SIZE**2
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0 * friction_limit:
                if w.skid_particle and w.skid_particle.grass == grass and len(
                        w.skid_particle.poly) < 30:
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass)
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            # Force more than max friction
            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit
                f_force *= force
                p_force *= force

            w.ApplyForceToCenter((
                p_force * side[0] + f_force * forw[0],
                p_force * side[1] + f_force * forw[1]),
                True)

    def draw(self, viewer, draw_particles=True):
        if draw_particles:
            for p in self.particles:
                viewer.draw_polyline(p.poly, color=p.color, linewidth=5)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)
                if "phase" not in obj.__dict__:
                    continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W*SIZE, +WHEEL_R*c1 *
                     SIZE), (+WHEEL_W*SIZE, +WHEEL_R*c1*SIZE),
                    (+WHEEL_W*SIZE, +WHEEL_R*c2 *
                     SIZE), (-WHEEL_W*SIZE, +WHEEL_R*c2*SIZE)
                ]
                viewer.draw_polygon(
                    [trans*v for v in white_poly], color=WHEEL_WHITE)

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass
        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
