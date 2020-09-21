import numpy as np
from szeth.envs.pr2.pr2_env import pr2_env

ACTIONS = [i for i in range(14)]


class pr2_7d_jspace_env(pr2_env):
    def __init__(self, args, mass, use_gui, no_dynamics=False):
        pr2_env.__init__(self, args, mass, use_gui, no_dynamics)

        # Setup joint limits
        self.joint_limits = self.sim.get_arm_joint_limits()

        # Define grid limits
        self.grid_continuous_limits = self.joint_limits[:,
                                                        1] - self.joint_limits[:, 0]

        # Compute grid cell sizes
        self.grid_cell_sizes = self.grid_continuous_limits / self.args.grid_size

        # Get start cell
        self.start_cell = self._continuous_to_grid(self.sim.get_arm_conf())

        # # Do IK to figure out goal cell
        # self.goal_conf = np.array(self.sim.arm_ik_2(
        #     (self.goal_position, self.goal_orientation), ik_hack=True),
        #     dtype=np.float32)
        # self.goal_cell = self._continuous_to_grid(self.goal_conf)
        # print(self.goal_cell)
        # FIX: The goal is defined in 3D space

    def set_cell_no_dynamics(self, cell):
        # Get corresponding conf
        conf = self._grid_to_continuous(cell)
        if not self.sim.check_arm_collision(conf, self.get_relevant_obstacles()):
            # Directly set the arm conf
            self.sim.set_arm_conf(conf)
        return self.get_current_observation()

    def step(self, acs):
        if not isinstance(acs, tuple):
            acs = [acs]

        current_observation = self.get_current_observation()
        current_cell = current_observation['observation'].copy()
        cost = 0
        for ac in acs:
            next_cell = self.sub_successor(current_cell, ac)
            cost += self.get_cost(current_cell, ac, next_cell)
            current_cell = next_cell.copy()

        if self.no_dynamics:
            # Teleport to commanded cell
            no_dynamics_observation = self.set_cell_no_dynamics(next_cell)
            return no_dynamics_observation, cost
        # Move to commanded cell
        self.move_to_cell(next_cell)
        # Read in the current observation
        new_observation = self.get_current_observation()
        return new_observation, cost

    def sub_successor(self, cell, ac):
        joint_index = int(ac / 2)
        displacement = ac % 2

        # Displace the correct joint by the right displacement
        next_cell = cell.copy()
        next_cell[joint_index] = min(
            max(next_cell[joint_index] + (2 * displacement - 1), 0), self.args.grid_size - 1)

        return next_cell

    def _continuous_to_grid(self, conf):
        conf = np.array(conf, dtype=np.float32)
        wrapped_conf = conf.copy()
        alpha_4 = np.round(wrapped_conf[4] / (2 * np.pi))
        alpha_6 = np.round(wrapped_conf[6] / (2 * np.pi))
        wrapped_conf[4] = wrapped_conf[4] - alpha_4 * 2 * np.pi
        wrapped_conf[6] = wrapped_conf[6] - alpha_6 * 2 * np.pi

        zero_adjusted_grid = wrapped_conf - self.joint_limits[:, 0]
        grid_cell = np.array(zero_adjusted_grid //
                             self.grid_cell_sizes, dtype=np.int32)
        grid_cell = np.maximum(0, np.minimum(grid_cell, self.args.grid_size-1))
        return grid_cell

    def _grid_to_continuous(self, grid_cell):
        grid_cell = np.array(grid_cell, dtype=np.int32)
        joint_conf = grid_cell * self.grid_cell_sizes + \
            self.grid_cell_sizes / 2.0 + self.joint_limits[:, 0]
        return joint_conf

    def get_pose(self, cell):
        joint_conf = self._grid_to_continuous(cell)
        pose = self.sim.arm_fk(joint_conf)
        return pose

    def move_to_cell(self, cell):
        joint_conf = self._grid_to_continuous(cell)
        # path = self.sim.compute_move_arm_path(
        #     joint_conf, obstacles=self.get_relevant_obstacles())
        # if path is not None:
        #     self.sim.execute_path(self.arm_joints, path)
        self.sim.execute_path(self.arm_joints, [joint_conf])

        # read the current grid cell
        return self.get_current_cell()

    def get_current_cell(self):
        current_conf = self.sim.get_arm_conf()
        current_cell = self._continuous_to_grid(current_conf)
        return current_cell

    def get_actions(self):
        return ACTIONS

    def get_cost(self, cell, ac, next_cell):
        # Check if the next_conf is in collision
        next_conf = self._grid_to_continuous(next_cell)
        if self.sim.check_arm_collision(next_conf, self.get_relevant_obstacles()):
            return 1e6
        return 1

    def check_goal(self, cell):
        # if goal_position is None:
        #     goal_position = self.goal_position

        # Do FK on current cell
        # conf = self._grid_to_continuous(cell)
        # gripper_pose = self.sim.arm_fk(conf)
        # if np.all(np.isclose(gripper_pose[0], goal_position, atol=1e-3)) and np.all(np.isclose(gripper_pose[1], self.goal_orientation, atol=1e-2)):
        #     return True
        if np.array_equal(cell, self.goal_cell):
            return True
        return False
