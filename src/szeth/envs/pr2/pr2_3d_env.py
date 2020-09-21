import numpy as np
from itertools import combinations_with_replacement
from szeth.envs.pr2.pr2_env import pr2_env

ACTIONS = [0, 1, 2, 3, 4, 5]
MPRIM_LENGTH = 2


class pr2_3d_env(pr2_env):
    def __init__(self, args, mass, use_gui, no_dynamics=False, no_kinematics=False):
        pr2_env.__init__(self, args, mass, use_gui, no_dynamics, no_kinematics)

        # Define grid limits
        self.grid_x_continuous_limits = [-0.5, 0.5]
        self.grid_y_continuous_limits = [-0.6, -0.2]
        self.grid_z_continuous_limits = [
            self.sim.table_max_z, self.sim.table_max_z + 0.5]

        # Define number of grid cells
        (self.grid_x_num_cells, self.grid_y_num_cells,
         self.grid_z_num_cells) = (args.grid_size, args.grid_size, args.grid_size)

        # Compute grid cell size
        self.grid_x_cell_size = (
            (self.grid_x_continuous_limits[1] - self.grid_x_continuous_limits[0]) / self.grid_x_num_cells)
        self.grid_y_cell_size = (
            (self.grid_y_continuous_limits[1] - self.grid_y_continuous_limits[0]) / self.grid_y_num_cells)
        self.grid_z_cell_size = (
            (self.grid_z_continuous_limits[1] - self.grid_z_continuous_limits[0]) / self.grid_z_num_cells)

        # Get start cell and goal cell
        self.start_cell = self._continuous_to_grid(
            self.sim.get_gripper_pose()[0])
        self.goal_cell = self._continuous_to_grid(
            self.goal_position)
        self.current_cell = self.start_cell.copy()

        # Initialize cost map
        self.initialize_cost_map()

    def get_obstacle_cells(self):
        if self.args.scenario in [1, 4]:
            lower_corner, upper_corner = self.sim.get_aabb(self.obstacle)
            offset = 0
            cell_offset = 1
            # # Get lower corner inflated by object shape
            lower_corner = lower_corner - self.object_dimensions/2.0 - offset

            # Get corresponding grid cell
            lower_corner_cell = self._continuous_to_grid(
                lower_corner) - cell_offset

            # # Get upper corner inflated by object shape
            upper_corner = upper_corner + self.object_dimensions/2.0 + offset

            # Get corresponding grid cell
            upper_corner_cell = self._continuous_to_grid(
                upper_corner) + cell_offset

            # Find the range of cells to be marked
            cells = set([(x, y, z)
                         for x in range(lower_corner_cell[0], upper_corner_cell[0] + 1)
                         for y in range(lower_corner_cell[1], upper_corner_cell[1] + 1)
                         for z in range(lower_corner_cell[2], upper_corner_cell[2] + 1)])
        elif self.args.scenario in [2, 3]:
            cells = set()

        return cells

    def get_table_cells(self):
        # Get aabb of the table
        lower_corner, upper_corner = self.sim.get_aabb(self.table)
        offset = 0
        cell_offset = 1

        # Inflate the aabb with the object dimensions
        lower_corner[0] = lower_corner[0] - \
            self.object_dimensions[0]/2.0 - offset
        lower_corner[1] = lower_corner[1] - \
            self.object_dimensions[1]/2.0 - offset
        lower_corner[2] = lower_corner[2] - \
            self.object_dimensions[2]/2.0 - offset

        upper_corner[0] = upper_corner[0] + \
            self.object_dimensions[0]/2.0 + offset
        upper_corner[1] = upper_corner[1] + \
            self.object_dimensions[1]/2.0 + offset
        upper_corner[2] = upper_corner[2] + \
            self.object_dimensions[2]/2.0 + offset

        # Get correspoding grid cells
        lower_corner_cell = self._continuous_to_grid(
            lower_corner) - cell_offset
        upper_corner_cell = self._continuous_to_grid(
            upper_corner) + cell_offset

        # Find the range of cells to be marked
        cells = set([(x, y, z)
                     for x in range(lower_corner_cell[0], upper_corner_cell[0] + 1)
                     for y in range(lower_corner_cell[1], upper_corner_cell[1] + 1)
                     for z in range(lower_corner_cell[2], upper_corner_cell[2] + 1)])

        if self.args.scenario in [2, 3, 4]:
            # Get aabb of the table
            lower_corner, upper_corner = self.sim.get_aabb(self.table_other)

            # Inflate the aabb with the object dimensions
            lower_corner[0] = lower_corner[0] - \
                self.object_dimensions[0]/2.0 - offset
            lower_corner[1] = lower_corner[1] - \
                self.object_dimensions[1]/2.0 - offset
            lower_corner[2] = lower_corner[2] - \
                self.object_dimensions[2]/2.0 - offset

            upper_corner[0] = upper_corner[0] + \
                self.object_dimensions[0]/2.0 + offset
            upper_corner[1] = upper_corner[1] + \
                self.object_dimensions[1]/2.0 + offset
            upper_corner[2] = upper_corner[2] + \
                self.object_dimensions[2]/2.0 + offset

            # Get correspoding grid cells
            lower_corner_cell = self._continuous_to_grid(
                lower_corner) - cell_offset
            upper_corner_cell = self._continuous_to_grid(
                upper_corner) + cell_offset

            # Add the cells
            cells_other = set([(x, y, z)
                               for x in range(lower_corner_cell[0], upper_corner_cell[0] + 1)
                               for y in range(lower_corner_cell[1], upper_corner_cell[1] + 1)
                               for z in range(lower_corner_cell[2], upper_corner_cell[2] + 1)])

            cells = cells.union(cells_other)

        return cells

    def initialize_cost_map(self):
        self.cost_map = np.ones((self.args.grid_size,
                                 self.args.grid_size,
                                 self.args.grid_size), dtype=np.int32)
        # goal = self.goal_cell
        # start = self.start_cell
        # # First column should all be low
        # self.cost_map[start[0], start[1], :] = 1
        # # Top row should all be low
        # self.cost_map[:, start[1], self.args.grid_size-1] = 1
        # # Last column should all be low
        # self.cost_map[goal[0], goal[1], :] = 1

        # Increase all cells corresponding to obstacle/table to large values
        cells = self.get_obstacle_cells()
        cells = cells.union(self.get_table_cells())
        for cell in cells:
            if self.out_of_bounds(cell):
                continue
            self.cost_map[cell] = 1e1

    def set_cell_no_dynamics(self, cell):
        # TODO: Should I just be using this, instead of storing/restoring
        # simulation state ?
        if self.no_kinematics:
            self.current_cell = cell.copy()
            return self.get_current_observation()
        position, orientation = self.sim.get_gripper_pose()
        new_position = self._grid_to_continuous(cell)
        gripper_pose = self.sim.set_gripper_pose((new_position, orientation))
        if gripper_pose is None:
            # IK failed to find a joint conf
            # Go back to old pose
            self.sim.set_gripper_pose((position, orientation))
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
        if self.no_dynamics or self.no_kinematics:
            # Teleport to commanded cell
            no_dynamics_observation = self.set_cell_no_dynamics(next_cell)
            return no_dynamics_observation, cost
        # Move to commanded cell
        self.move_to_cell(next_cell)
        # Read in the current_observation
        new_observation = self.get_current_observation()
        return new_observation, cost

    def sub_successor(self, cell, ac):
        next_cell = cell.copy()
        if ac == 0:
            next_cell = np.array((max(next_cell[0] - 1, 0),
                                  next_cell[1],
                                  next_cell[2]), dtype=np.int32)
        elif ac == 1:
            next_cell = np.array((min(next_cell[0] + 1, self.args.grid_size - 1),
                                  next_cell[1],
                                  next_cell[2]), dtype=np.int32)
        elif ac == 2:
            next_cell = np.array((next_cell[0],
                                  max(next_cell[1] - 1, 0),
                                  next_cell[2]), dtype=np.int32)
        elif ac == 3:
            next_cell = np.array((next_cell[0],
                                  min(next_cell[1] + 1,
                                      self.args.grid_size-1),
                                  next_cell[2]), dtype=np.int32)
        elif ac == 4:
            next_cell = np.array((next_cell[0],
                                  next_cell[1],
                                  max(next_cell[2] - 1, 0)), dtype=np.int32)
        elif ac == 5:
            next_cell = np.array((next_cell[0],
                                  next_cell[1],
                                  min(next_cell[2] + 1, self.args.grid_size - 1)),
                                 dtype=np.int32)
        else:
            raise Exception('Invalid action')

        return next_cell

    def _continuous_to_grid(self, position):
        x, y, z = position

        # HACK: Clip the continuous values to be within the grid
        x = min(
            max(x, self.grid_x_continuous_limits[0]), self.grid_x_continuous_limits[1])
        y = min(
            max(y, self.grid_y_continuous_limits[0]), self.grid_y_continuous_limits[1])
        z = min(
            max(z, self.grid_z_continuous_limits[0]), self.grid_z_continuous_limits[1])

        x_shifted = x - self.grid_x_continuous_limits[0]
        y_shifted = y - self.grid_y_continuous_limits[0]
        z_shifted = z - self.grid_z_continuous_limits[0]

        x_cell = x_shifted // self.grid_x_cell_size
        y_cell = y_shifted // self.grid_y_cell_size
        z_cell = z_shifted // self.grid_z_cell_size

        # HACK: Clip the discrete values to be within the grid
        x_cell = max(0, min(x_cell, self.args.grid_size - 1))
        y_cell = max(0, min(y_cell, self.args.grid_size - 1))
        z_cell = max(0, min(z_cell, self.args.grid_size - 1))

        return np.array([x_cell, y_cell, z_cell], dtype=np.int32)

    def _grid_to_continuous(self, cell):
        x_cell, y_cell, z_cell = cell

        x = x_cell * self.grid_x_cell_size + self.grid_x_cell_size / \
            2.0 + self.grid_x_continuous_limits[0]
        y = y_cell * self.grid_y_cell_size + self.grid_y_cell_size / \
            2.0 + self.grid_y_continuous_limits[0]
        z = z_cell * self.grid_z_cell_size + self.grid_z_cell_size / \
            2.0 + self.grid_z_continuous_limits[0]

        return np.array([x, y, z], dtype=np.float32)

    def move_to_cell(self, cell):
        # Get current gripper pose
        # current_position, current_orientation = self.sim.get_gripper_pose()
        # Get current grid cell
        # current_cell = self._continuous_to_grid(current_position)
        # # Check if commanded cell is one of the 4 adjacent neighbors of current
        # # grid cell
        # displacement = current_cell - cell
        # if np.count_nonzero(displacement == 0) < 2:
        #     warnings.warn(
        #         'Trying to execute the incorrect displacement'+np.array2string(displacement))
        #     return current_cell
        # if np.abs(displacement[displacement != 0]) != 1:
        #     warnings.warn(
        #         'Trying to execute the incorrect displacement'+np.array2string(displacement))
        #     return current_cell

        position = self._grid_to_continuous(cell)
        path = self.sim.compute_move_end_effector_path(
            position,
            # obstacles=self.get_relevant_obstacles(),
            obstacles=[],
            check_attachments=False)
        if path is not None:
            self.sim.execute_path(self.arm_joints, path)

        # read the current grid cell
        current_cell = self.get_current_cell()
        return current_cell

    def get_current_cell(self):
        if self.no_kinematics:
            return self.current_cell
        current_position, _ = self.sim.get_gripper_pose()
        current_cell = self._continuous_to_grid(current_position)

        return current_cell

    def get_actions(self):
        return ACTIONS

    def get_extended_actions(self):
        return list(combinations_with_replacement(ACTIONS, MPRIM_LENGTH))

    def get_repeated_actions(self):
        actions = []
        actions += ACTIONS[:5]
        actions.append((5, 5))
        # for ac in ACTIONS:
        #     actions.append((ac,)*MPRIM_LENGTH)

        return actions

    def get_cost(self, cell, ac, next_cell):
        cost = self.cost_map[tuple(next_cell)]
        return cost

    def check_goal(self, cell, goal_cell=None):
        if goal_cell is None:
            goal_cell = self.goal_cell
        # if np.array_equal(cell, goal_cell):
        #     return True
        # else:
        #     return False
        if np.sum(np.abs(cell - goal_cell)) <= self.args.goal_threshold:
            return True
        else:
            return False

    def out_of_bounds(self, cell):
        np_cell = np.array(cell)
        if np.any(np_cell < 0) or np.any(np_cell >= self.args.grid_size):
            return True
        return False
