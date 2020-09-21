import numpy as np
from pr2_sim_pybullet.sim import PR2SimBullet
import warnings


class pr2_env:
    def __init__(self, args, mass, use_gui, no_dynamics=False, no_kinematics=False):
        self.args = args
        self.mass = mass
        self.use_gui = use_gui
        self.no_dynamics = no_dynamics
        self.no_kinematics = no_kinematics
        self.seed = self.args.seed

        # Initialize simulation
        self.sim = PR2SimBullet(
            use_gui=use_gui, scenario=self.args.scenario, seed=self.args.seed)
        self.arm_joints = self.sim.get_arm_joints()
        self.gripper_joints = self.sim.get_gripper_joints()
        self.table = self.sim.table_id
        if self.args.scenario in [2, 3, 4]:
            self.table_other = self.sim.table_id_other

        self._add_objects_and_obstacles()

        # Add goal
        self._add_goal()

        # Pick the object
        self._pick_object()

    def _add_objects_and_obstacles(self):
        self._add_object()
        self._add_obstacle()

    def _add_object(self):
        self.object_dimensions = np.array((0.05, 0.05, 0.15))
        if self.args.scenario == 1:
            self.object_position = np.array(
                (-0.1, -0.4, self.sim.table_max_z+0.15/2))
        elif self.args.scenario == 2:
            self.object_position = np.array(
                (-0.2, -0.4, self.sim.table_max_z + 0.15/2))
        elif self.args.scenario == 3:
            self.object_position = np.array(
                (0.35, -0.4, self.sim.table_max_z + 0.15/2))
        elif self.args.scenario == 4:
            min_object_position_x = 0.35
            max_object_position_x = 0.45

            if self.seed is not None:
                rng = np.random.RandomState(self.seed + 10)
                object_position_x = rng.uniform(min_object_position_x,
                                                max_object_position_x)
            else:
                object_position_x = 0.35
            self.object_position = (object_position_x,
                                    -0.4,
                                    self.sim.table_max_z + 0.15/2 - 0.05/2)
        self.object_orientation = np.array((0, 0, 0))
        self.object = self.sim.add_object(shape='box',
                                          dimensions=self.object_dimensions,
                                          position=self.object_position,
                                          orientation=self.object_orientation,
                                          mass=self.mass,
                                          color=self.sim.object_color)

    def _add_obstacle(self):
        if self.args.scenario == 1:
            self.obstacle_dimensions = np.array((0.15, 0.15, 0.25))
            self.obstacle_orientation = np.array((0, 0, 0))
            self.obstacle_position = np.array(
                (0.1, -0.4, self.sim.table_max_z+0.25/2))
            self.obstacle = self.sim.add_obstacle(shape='box',
                                                  dimensions=self.obstacle_dimensions,
                                                  position=self.obstacle_position,
                                                  orientation=self.obstacle_orientation,
                                                  color=self.sim.obstacle_color)
        elif self.args.scenario == 2:
            max_obstacle_height = 0.3
            min_obstacle_height = 0.1

            max_obstacle_x = 0.1
            min_obstacle_x = -0.05

            if self.seed is not None:
                rng = np.random.RandomState(self.seed + 3)
                obstacle_height = rng.rand() * (max_obstacle_height - min_obstacle_height) + \
                    min_obstacle_height
                obstacle_x = rng.rand() * (max_obstacle_x - min_obstacle_x) + min_obstacle_x
            else:
                obstacle_height = 0.25
                obstacle_x = 0

            self.obstacle_dimensions = np.array((0.05, 0.15, obstacle_height))
            self.obstacle_orientation = np.array((0, 0, 0))
            self.obstacle_position = np.array(
                (obstacle_x,
                 -0.4,
                 self.sim.table_max_z + obstacle_height/2.0 - 0.03))
            self.obstacle = self.sim.add_obstacle(shape='box',
                                                  dimensions=self.obstacle_dimensions,
                                                  position=self.obstacle_position,
                                                  orientation=self.obstacle_orientation,
                                                  color=self.sim.obstacle_color)
        elif self.args.scenario == 3:
            max_obstacle_height = 0.3
            min_obstacle_height = 0.1

            min_obstacle_x = -0.05
            max_obstacle_x = 0.15
            if self.seed is not None:
                rng = np.random.RandomState(self.seed + 3)
                obstacle_height = rng.rand() * (max_obstacle_height - min_obstacle_height) + \
                    min_obstacle_height
                obstacle_x = rng.rand() * (max_obstacle_x - min_obstacle_x) + min_obstacle_x
            else:
                obstacle_height = 0.25
                obstacle_x = 0

            self.obstacle_dimensions = np.array(
                (0.1, 0.15, obstacle_height))
            self.obstacle_orientation = np.array((0, 0, 0))
            self.obstacle_position = np.array(
                (obstacle_x,
                 -0.4,
                 self.sim.table_max_z + obstacle_height/2.0 - 0.03))
            self.obstacle = self.sim.add_obstacle(shape='box',
                                                  dimensions=self.obstacle_dimensions,
                                                  position=self.obstacle_position,
                                                  orientation=self.obstacle_orientation,
                                                  color=self.sim.obstacle_color)

        elif self.args.scenario == 4:
            min_obstacle_position_x = 0.05
            max_obstacle_position_x = 0.15

            if self.seed is not None:
                rng = np.random.RandomState(self.seed + 5)
                obstacle_position_x = rng.uniform(min_obstacle_position_x,
                                                  max_obstacle_position_x)
            else:
                obstacle_position_x = 0.1
            obstacle_height = 0.2
            self.obstacle_position = np.array(
                (obstacle_position_x,
                 -0.4,
                 self.sim.table_max_z + obstacle_height/2 - 0.05/2))
            self.obstacle_dimensions = np.array((0.1, 0.15, obstacle_height))
            self.obstacle = self.sim.add_obstacle(shape='box',
                                                  dimensions=self.obstacle_dimensions,
                                                  position=self.obstacle_position,
                                                  orientation=(0, 0, 0),
                                                  color=self.sim.obstacle_color)

            # # add another obstacle
            # obstacle_other_position_x = 0.5
            # self.obstacle_other_position = np.array(
            #     (obstacle_other_position_x,
            #      -0.4,
            #      self.sim.table_max_z + obstacle_height/2 - 0.05/2))
            # self.obstacle_other_dimensions = np.array(
            #     (0.1, 0.15, obstacle_height))
            # self.obstacle_other = self.sim.add_obstacle(
            #     shape='box',
            #     dimensions=self.obstacle_other_dimensions,
            #     position=self.obstacle_other_position,
            #     orientation=(0, 0, 0),
            #     color=self.sim.obstacle_color)

    def _add_goal(self):
        pre_place_position_offset = 0.05
        if self.args.scenario == 1:
            self.goal_position = np.array(
                (0.4, -0.4, self.sim.table_max_z + 0.15/2 + pre_place_position_offset))
            self.place_position = np.array(
                (0.4, -0.4, self.sim.table_max_z + 0.15/2))
        elif self.args.scenario == 2:
            # the below position is reachable for mass 4 with goal threshold 1
            # self.goal_position = np.array(
            #     (0.35, -0.45, self.sim.table_max_z_other + 0.15/2 + pre_place_position_offset))
            # self.place_position = np.array(
            #     (0.35, -0.45, self.sim.table_max_z_other + 0.15/2))
            # The below position is reachable for mass 3 (but not mass 4) with goal threshold 0
            self.goal_position = np.array(
                (0.35, -0.4, self.sim.table_max_z_other + 0.15/2 + pre_place_position_offset))
            self.place_position = np.array(
                (0.35, -0.4, self.sim.table_max_z_other + 0.15/2))
        elif self.args.scenario == 3:
            self.goal_position = np.array(
                (-0.15, -0.4, self.sim.table_max_z_other + 0.15/2 + pre_place_position_offset))
            self.place_position = np.array(
                (-0.15, -0.4, self.sim.table_max_z_other + 0.15/2))
        elif self.args.scenario == 4:
            self.place_position = np.array((-0.05, -0.4,
                                            self.sim.table_max_z_other + 0.15/2 - 0.05/2))
            self.goal_position = self.place_position.copy()
            self.goal_position[2] += pre_place_position_offset/2
        self.goal_orientation = self.sim.ee_orientation
        self.goal_orientation_quat = self.sim.ee_orientation_quat
        return

    def _pick_object(self):
        all_paths = self.sim.compute_grasp_object_path(
            self.object, table_id=self.table, obstacles=self.get_relevant_obstacles(), ik_hack=True)
        if all_paths is None:
            raise Exception('Could not compute path to grasp object!')
        path, gripper_close_path, post_grasp_path = all_paths
        if not self.no_dynamics:
            # If simulating dynamics, then actually execute the grasp (including post grasp)
            self.sim.execute_grasp(self.object, self.arm_joints, self.gripper_joints,
                                   path, gripper_close_path, post_grasp_path)
            # Need to check if the grasp is actually successful, sometimes it fails
            gripper_position, _ = self.sim.get_gripper_pose()
            object_position = self.sim.get_position(self.object)
            distance = np.linalg.norm(
                np.array(gripper_position) - np.array(object_position))
            if distance > 0.05:
                print('GRASPING FAILED. RESETTING!')
                self.reset()
        else:
            # Simply set the arm to the last conf in post_grasp_path
            self.sim.set_arm_conf(post_grasp_path[-1])
            # Set gripper conf to the last conf in gripper_close_path
            self.sim.set_gripper_conf(gripper_close_path[-1])
            # Teleport the object as well
            gripper_pose = self.sim.get_gripper_pose()
            self.sim.set_pose(self.object, gripper_pose)
            # Manually create an attachment
            self.sim.add_attachment(self.object)
            self.sim.add_constraint(self.object)

    def get_relevant_obstacles(self):
        if self.args.scenario == 1:
            return [self.obstacle, self.table]
        elif self.args.scenario == 2:
            return [self.table_other, self.obstacle]
        elif self.args.scenario == 3:
            return [self.table_other, self.obstacle]
        elif self.args.scenario == 4:
            return [self.obstacle]

    def set_simulation_state(self, sim_state, goal=False):
        self.sim.set_simulation_state(sim_state)
        return self.get_current_observation(goal)

    def get_simulation_state(self):
        return self.sim.get_simulation_state()

    def successor(self, cell, acs):
        next_cell = cell.copy()
        if not isinstance(acs, tuple):
            # Single action
            acs = [acs]
        for ac in acs:
            next_cell = self.sub_successor(next_cell, ac)
        return next_cell

    def get_current_observation(self, goal=False):
        cell = self.get_current_cell()
        if self.no_kinematics:
            sim_state = None
        else:
            sim_state = self.get_simulation_state()
        if not goal:
            return {'observation': cell, 'sim_state': sim_state}
        else:
            return {'observation': cell, 'sim_state': sim_state,
                    'desired_goal': self.goal_cell}

    def set_observation(self, observation, goal=False):
        if self.no_kinematics:
            self.current_cell = observation['observation'].copy()
            if goal:
                self.goal_cell = observation['desired_goal'].copy()
            return self.get_current_observation(goal)
        if self.no_dynamics:
            # Not simulating dynamics
            if goal:
                self.goal_cell = observation['desired_goal'].copy()
            return self.set_cell_no_dynamics(observation['observation'])
        else:
            raise Exception('We should not be using this. not tested well')
            # Restore simulation state
            if goal:
                self.goal_cell = observation['desired_goal'].copy()
            return self.set_simulation_state(observation['sim_state'], goal)

    def execute_goal_completion(self):
        # Called when the robot has reached the goal
        # Re-orient the gripper so that it is upright
        gripper_position, gripper_orientation = self.sim.get_gripper_pose()
        path = self.sim.compute_move_end_effector_path(gripper_position,
                                                       self.goal_orientation_quat,
                                                       obstacles=[],
                                                       ik_hack=True)
        if path is not None:
            self.sim.execute_path(self.arm_joints, path)
            # Place the object
            if self.args.scenario in [2, 3, 4]:
                (path, gripper_open_path, post_place_path) = self.sim.compute_place_object_path(
                    self.object, self.place_position, table_id=self.table_other)
            elif self.args.scenario == 1:
                (path, gripper_open_path, post_place_path) = self.sim.compute_place_object_path(
                    self.object, self.place_position, table_id=self.table)
            # Execute the path
            if path is not None and gripper_open_path is not None and post_place_path is not None:
                self.sim.execute_place(self.object, self.arm_joints,
                                       self.gripper_joints, path,
                                       gripper_open_path, post_place_path)
            else:
                self.remove_object_constraints()
        else:
            self.remove_object_constraints()
        return

    def remove_object_constraints(self):
        self.sim.assign()
        self.sim.remove_attachment(self.object)
        self.sim.remove_constraint(self.object)
        return

    def reset(self, goal=True):
        self.sim.reset()
        self._add_objects_and_obstacles()
        self._pick_object()

        return

    def wait_for_user(self):
        self.sim.wait_for_user()
        return

    def _teleport_object(self):
        self.sim.set_pose(self.object, (self.object_position, None))
        return

    def recreate_object(self, place_failed=False):
        # HACK: We destroy the object placed, and recreate it at its start position,
        # so, the robot has to go back to pick it up
        if place_failed:
            self.remove_object_constraints()
        self._teleport_object()
        # if self.args.scenario in [2]:
        #     path = self.sim.compute_move_arm_path(
        #         self.sim.carry_conf, obstacles=self.get_relevant_obstacles() + [self.table])
        #     self.sim.execute_path(self.arm_joints, path)
        # else:
        # HACK: Just teleport the arm to carry conf
        self.sim.set_arm_conf(self.sim.carry_conf)
        self._pick_object()
        return

    def close(self):
        self.sim.close()
        return

    def sub_successor(self, cell, ac):
        raise NotImplementedError

    def _grid_to_continuous(self, cell):
        raise NotImplementedError

    def _continuous_to_grid(self, conf):
        raise NotImplementedError

    def move_to_cell(self, cell):
        raise NotImplementedError

    def step(self, ac):
        raise NotImplementedError

    def get_current_cell(self):
        raise NotImplementedError

    def set_cell_no_dynamics(self):
        raise NotImplementedError

    def get_actions(self):
        raise NotImplementedError

    def get_cost(self, cell, acs, next_cell, goal_cell=None):
        raise NotImplementedError

    def check_goal(self, cell, goal_cell=None):
        raise NotImplementedError

    def out_of_bounds(self, cell):
        raise NotImplementedError
