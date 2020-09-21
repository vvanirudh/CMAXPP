import time
import numpy as np
from itertools import combinations_with_replacement, product
from szeth.envs.pr2.pr2_env import pr2_env
import pickle
import os
import os.path as osp

ACTIONS = [i for i in range(14)]
MPRIM_LENGTH = 2


class pr2_7d_xyzrpy_env(pr2_env):

    # static variable
    cell_to_joint_conf = {}
    cost_cache = {}

    def __init__(self, args, mass, use_gui, no_dynamics=False, no_kinematics=False):
        pr2_env.__init__(self, args, mass, use_gui, no_dynamics, no_kinematics)

        # Define workspace limits
        # self.workspace_x_limits = [-0.5, 0.5]
        self.workspace_x_limits = [-0.3, 0.5]
        # self.workspace_y_limits = [-0.6, -0.2]
        self.workspace_y_limits = [-0.5, -0.3]
        if self.args.scenario == 1:
            self.workspace_z_limits = [
                self.sim.table_max_z, self.sim.table_max_z + 0.5]
        else:
            self.workspace_z_limits = [
                self.sim.table_max_z, self.sim.table_max_z_other + 0.2]
        self.workspace_roll_limits = [-np.pi, np.pi]
        self.workspace_pitch_limits = [-np.pi, np.pi]
        self.workspace_yaw_limits = [-np.pi, np.pi]

        # Define redundant joint limits
        self.redundant_joint_limits = self.sim.redundant_joint_limits

        # Define grid limits
        self.grid_limits = np.zeros((7, 2))
        self.grid_limits[0] = self.workspace_x_limits
        self.grid_limits[1] = self.workspace_y_limits
        self.grid_limits[2] = self.workspace_z_limits
        self.grid_limits[3] = self.workspace_roll_limits
        self.grid_limits[4] = self.workspace_pitch_limits
        self.grid_limits[5] = self.workspace_yaw_limits
        self.grid_limits[6] = self.redundant_joint_limits

        # Compute grid cell sizes
        self.grid_cell_sizes = (
            self.grid_limits[:, 1] - self.grid_limits[:, 0]) / self.args.grid_size

        # Get start cell
        self.start_cell = self._continuous_to_grid(self.sim.get_arm_conf())
        self.current_cell = self.start_cell.copy()

        # Get goal cells
        self.goal_cells = self._6d_to_grid(
            self.goal_position, self.goal_orientation)
        self.goal_cell = self.goal_cells[0].copy()

        # IK cache
        # self.cell_to_joint_conf = {}

        # Get carry conf
        self.carry_conf = self.sim.carry_conf
        self.carry_cell = self._continuous_to_grid(self.carry_conf)

        # Get obstacle cell aa and bb
        self.obstacle_position_aa = self.obstacle_position - self.obstacle_dimensions/2
        self.obstacle_position_bb = self.obstacle_position + self.obstacle_dimensions/2
        self.obstacle_cell_aa = self._continuous_to_grid(
            self.obstacle_position_aa)
        self.obstacle_cell_bb = self._continuous_to_grid(
            self.obstacle_position_bb)

    def wrap_angle(self, theta, lower=-np.pi):
        # [-np.pi, np.pi)
        return (theta - lower) % (2 * np.pi) + lower

    def wrap_grid(self, cell):
        return cell % (self.args.grid_size)

    def is_circular(self, ind):
        if ind in [3, 4, 5]:
            return True
        return False

    def _continuous_to_grid(self, conf):
        conf = np.array(conf, dtype=np.float32)
        if conf.shape[0] == 7:
            # 7D Arm conf
            wrapped_conf = conf.copy()
            wrapped_conf[3] = self.wrap_angle(wrapped_conf[3])
            wrapped_conf[4] = self.wrap_angle(wrapped_conf[4])
            wrapped_conf[5] = self.wrap_angle(wrapped_conf[5])

            zero_adjusted_grid = wrapped_conf - self.grid_limits[:, 0]
            grid_cell = np.array(zero_adjusted_grid //
                                 self.grid_cell_sizes, dtype=np.int32)

            grid_cell = np.maximum(0, np.minimum(
                grid_cell, self.args.grid_size - 1))
        elif conf.shape[0] == 3:
            # 3D gripper pos
            zero_adjusted_grid = conf - self.grid_limits[:3, 0]
            grid_cell = np.array(zero_adjusted_grid //
                                 self.grid_cell_sizes[:3], dtype=np.int32)
            grid_cell = np.maximum(0, np.minimum(
                grid_cell, self.args.grid_size - 1))
        return grid_cell

    def _grid_to_continuous(self, grid_cell):
        grid_cell = np.array(grid_cell, dtype=np.int32)
        conf = grid_cell * self.grid_cell_sizes + \
            self.grid_cell_sizes / 2.0 + self.grid_limits[:, 0]
        return conf

    def _6d_to_grid(self, position, orientation):
        conf = np.zeros(6)
        conf[:3] = position
        conf[3] = self.wrap_angle(orientation[0])
        conf[4] = self.wrap_angle(orientation[1])
        conf[5] = self.wrap_angle(orientation[2])

        zero_adjusted_grid = conf - self.grid_limits[:6, 0]
        sub_grid_cell = np.array(zero_adjusted_grid //
                                 self.grid_cell_sizes[:6], dtype=np.int32)

        grid_cells = [np.append(sub_grid_cell, j)
                      for j in range(self.args.grid_size)]
        return grid_cells

    # def set_cell_no_dynamics(self, cell):
    #     assert self.no_dynamics, "Should only be called for no_dynamics"
    #     if tuple(cell) in pr2_7d_xyzrpy_env.cell_to_joint_conf:
    #         joint_conf = pr2_7d_xyzrpy_env.cell_to_joint_conf[tuple(cell)]
    #     else:
    #         # Get corresponding conf
    #         conf = self._grid_to_continuous(cell)
    #         # Get corresponding joint conf
    #         joint_conf = self.sim.get_joint_conf_from_xyz_rpy(
    #             conf, obstacles=self.get_relevant_obstacles())
    #         # cache
    #         pr2_7d_xyzrpy_env.cell_to_joint_conf[tuple(cell)] = joint_conf

    #     if joint_conf is not None:
    #         # Directly set the arm conf
    #         self.sim.set_arm_conf(joint_conf)
    #         # Set the object pose as well
    #         # TODO: The next two lines are actually needed; removing them
    #         # for now to get some gains
    #         # gripper_pose = self.sim.get_gripper_pose()
    #         # self.sim.set_pose(self.object, gripper_pose)
    #     return self.get_current_observation()

    def set_cell_no_dynamics(self, cell):
        assert self.no_dynamics
        self.current_cell = cell.copy()
        return self.get_current_observation(goal=True)

    def sub_successor(self, cell, ac):
        index = int(ac / 2)
        displacement = ac % 2

        # Displace the correct dimension by the correct displacement
        # Need to check circular dimensions
        next_cell = cell.copy()
        if self.is_circular(index):
            next_cell[index] += 2 * displacement - 1
            next_cell[index] = self.wrap_grid(next_cell[index])
        else:
            next_cell[index] = min(
                max(next_cell[index] + 2 * displacement - 1, 0), self.args.grid_size-1)

        return next_cell

    def step(self, acs):
        if not isinstance(acs, tuple):
            acs = [acs]

        current_observation = self.get_current_observation(goal=True)
        cell = current_observation['observation'].copy()
        goal_cell = current_observation['desired_goal'].copy()
        cost = 0
        substep = 0
        for ac in acs:
            obs_substep = {'observation': cell, 'desired_goal': goal_cell}
            if self.check_goal(obs_substep):
                # Reached goal
                break
            next_sub_cell = self.sub_successor(cell, ac)
            cost_substep = self.get_cost(
                cell, ac, next_sub_cell, goal_cell)
            if cost_substep > 1:
                # Collision
                cost = len(acs)
                break
            cost += cost_substep
            cell = next_sub_cell.copy()
            substep += 1

        if self.no_dynamics:
            # Teleport to commanded cell
            no_dynamics_observation = self.set_cell_no_dynamics(cell)
            return no_dynamics_observation, cost
        # Move to commanded cell
        self.move_to_cell(cell)
        # Read in the current observation
        new_observation = self.get_current_observation(goal=True)
        return new_observation, cost

    def move_to_cell(self, cell):
        if tuple(cell) in pr2_7d_xyzrpy_env.cell_to_joint_conf:
            joint_conf = pr2_7d_xyzrpy_env.cell_to_joint_conf[tuple(cell)]
        else:
            conf = self._grid_to_continuous(cell)
            # Get corresponding joint conf
            joint_conf = self.sim.get_joint_conf_from_xyz_rpy(
                conf)
            # cache
            pr2_7d_xyzrpy_env.cell_to_joint_conf[tuple(cell)] = joint_conf
        # Check arm collision
        if joint_conf is not None:
            if self.sim.check_arm_collision(joint_conf,
                                            obstacles=self.get_relevant_obstacles(),
                                            attachment=self.object):
                # In collision
                joint_conf = None
        if joint_conf is not None:
            # Compute path to the joint conf
            # TODO: How do you account for attachment collision and still not deal
            # with start conf being in collision (and thus end conf, since it is computed)
            # relative to start conf
            # path = self.sim.compute_move_arm_path(
            #     joint_conf, obstacles=self.get_relevant_obstacles(),
            #     check_attachments=False)
            path = self.sim.compute_move_arm_path(joint_conf, obstacles=[])
            if path is not None:
                # Execute path
                self.sim.execute_path(self.arm_joints, path)

        current_cell = self.get_current_cell()
        # HACK: Simulation can be stochastic due to discretization
        # Snap onto the center of the grid manually
        joint_conf = self.get_joint_conf(current_cell)
        if joint_conf is not None:
            self.sim.set_arm_conf(joint_conf)
        return current_cell

    # def get_current_cell(self):
    #     conf = self.sim.get_xyz_rpy()
    #     current_cell = self._continuous_to_grid(conf)
    #     return current_cell

    def get_joint_conf(self, cell):
        if tuple(cell) in pr2_7d_xyzrpy_env.cell_to_joint_conf:
            return pr2_7d_xyzrpy_env.cell_to_joint_conf[tuple(cell)]

        conf = self._grid_to_continuous(cell)
        joint_conf = self.sim.get_joint_conf_from_xyz_rpy(conf)
        # Cache it
        pr2_7d_xyzrpy_env.cell_to_joint_conf[tuple(cell)] = joint_conf
        return joint_conf

    def get_current_cell(self):
        if self.no_dynamics:
            return self.current_cell
        conf = self.sim.get_xyz_rpy()
        current_cell = self._continuous_to_grid(conf)
        return current_cell

    def get_actions(self):
        return ACTIONS

    def get_extended_actions(self):
        return list(combinations_with_replacement(ACTIONS, MPRIM_LENGTH))

    def get_repeated_actions(self):
        actions = []
        actions += ACTIONS[:5]  # 5 actions
        # for ac in ACTIONS[:4]:  # 8 actions
        #     actions.append((ac, 5))
        #     actions.append((5, ac))
        actions.append((5, 5))  # 1 action
        actions += ACTIONS[6:]  # 8 actions
        return actions

    def get_cost(self, cell, ac, next_cell, goal_cell=None):
        obs = {'observation': cell, 'desired_goal': goal_cell}
        if self.check_goal(obs):
            # Already at goal
            return 0
        # if goal_cell is not None:
        #     currently_at_goal = np.array_equal(cell[:6], goal_cell[:6])
        #     next_at_goal = np.array_equal(next_cell[:6], goal_cell[:6])
        #     if currently_at_goal and next_at_goal:
        #         # Already at goal and staying at goal
        #         return 0
        if tuple(next_cell) in pr2_7d_xyzrpy_env.cost_cache:
            return pr2_7d_xyzrpy_env.cost_cache[tuple(next_cell)]
        # Check if next conf is in collision
        if tuple(next_cell) in pr2_7d_xyzrpy_env.cell_to_joint_conf:
            next_joint_conf = pr2_7d_xyzrpy_env.cell_to_joint_conf[tuple(
                next_cell)]
        else:
            next_conf = self._grid_to_continuous(next_cell)
            next_joint_conf = self.sim.get_joint_conf_from_xyz_rpy(
                next_conf)
            # cache
            pr2_7d_xyzrpy_env.cell_to_joint_conf[tuple(
                next_cell)] = next_joint_conf
        if next_joint_conf is None:
            # cache
            pr2_7d_xyzrpy_env.cost_cache[tuple(next_cell)] = 1000
            return 1000
        if self.sim.check_arm_collision(next_joint_conf,
                                        obstacles=self.get_relevant_obstacles(),
                                        attachment=self.object):
            # cache
            pr2_7d_xyzrpy_env.cost_cache[tuple(next_cell)] = 1000
            return 1000
        # cache
        pr2_7d_xyzrpy_env.cost_cache[tuple(next_cell)] = 1
        return 1

    # def get_cost_simplified(self, cell, ac, next_cell, goal_cell=None):
    #     if np.array_equal(cell[:6], goal_cell[:6]):
    #         return 0

    #     if tuple(next_cell) in pr2_7d_xyzrpy_env.cost_cache:
    #         return pr2_7d_xyzrpy_env.cost_cache[tuple(next_cell)]

    #     next_conf = self._grid_to_continuous(next_cell)
    #     collision = self.sim.check_xyzrpy_collision(
    #         next_conf, self.get_relevant_obstacles(), self.object)

    #     cost = 1
    #     if collision:
    #         cost = 100

    #     return cost

    def check_goal(self, obs):
        cell = obs['observation']
        goal_cell = obs['desired_goal']
        cell_6d = cell[:6]
        goal_cell_6d = goal_cell[:6]
        if np.sum(np.abs(cell_6d - goal_cell_6d)) <= self.args.goal_threshold:
            return True
        return False

    def out_of_bounds(self, cell):
        np_cell = np.array(cell)
        if np.any(np_cell < 0) or np.any(np_cell >= self.args.grid_size):
            return True
        return False
