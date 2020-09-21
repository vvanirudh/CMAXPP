import time
import copy
import numpy as np


class pr2_7d_cmax_agent:
    def __init__(self, args, env, planning_env, controller):
        self.args, self.env, self.planning_env = args, env, planning_env
        self.controller = controller

        self.goal_position = self.env.goal_position
        self.goal_orientation = self.env.goal_orientation
        self.goal_cell = self.env.goal_cell

        self.discrepancy_sets = {}
        self.value_dict = {}

    def get_cell_value(self, cell):
        # value = self.get_euclidean_heuristic(cell)
        value = self.get_manhattan_heuristic(cell)
        # Add any residual
        if tuple(cell) in self.value_dict:
            value += self.value_dict[tuple(cell)]

        return value

    def get_manhattan_heuristic(self, cell):
        return np.sum(np.abs(cell - self.goal_cell))

    def get_euclidean_heuristic(self, cell):
        # Get end-effector pose
        gripper_position, gripper_orientation = self.planning_env.get_pose(
            cell)
        gripper_position = np.array(gripper_position, dtype=np.float32)
        gripper_orientation = np.array(gripper_orientation, dtype=np.float32)
        # Use euclidean distance between gripper_position and goal_position
        euclidean_dist = np.linalg.norm(gripper_position - self.goal_position)
        euclidean_dist += np.linalg.norm(gripper_orientation -
                                         self.goal_orientation)
        return euclidean_dist

    def get_discrepancy(self, cell, ac, cost):
        if ac not in self.discrepancy_sets:
            return cost
        if tuple(cell) not in self.discrepancy_sets[ac]:
            return cost

        return 1e6

    def learn_online_in_real_world(self):
        # Reset environment
        current_observation = copy.deepcopy(self.env.get_current_observation())
        self.planning_env.set_observation(current_observation)
        self.controller.reconfigure_heuristic(self.get_cell_value)
        self.controller.reconfigure_discrepancy(self.get_discrepancy)

        total_n_steps = 0
        start = time.time()
        while True:
            print('-------------')
            print('Current cell', current_observation['observation'])
            print('Current cell heuristic',
                  self.get_cell_value(current_observation['observation']))

            ac, info = self.controller.act(copy.deepcopy(current_observation))
            print('Action', ac)
            # Step in the environment
            next_observation, cost = self.env.step(ac)
            print('True next cell', next_observation['observation'])
            total_n_steps += 1
            # Now set planning_env to the same state
            self.planning_env.set_observation(
                current_observation)
            # Step in the model
            next_sim_observation, _ = self.planning_env.step(ac)
            print('Predicted next cell', next_sim_observation['observation'])

            # Is there a discrepancy?
            if not np.array_equal(next_observation['observation'],
                                  next_sim_observation['observation']):
                if np.array_equal(next_observation['observation'],
                                  current_observation['observation']):
                    print('Blocking discrepancy!')
                else:
                    print('Non-Blocking Discrepancy!')
                cell = current_observation['observation']
                # self.discrepancy_matrix[ac, cell[0], cell[1], cell[2]] += 1
                if ac not in self.discrepancy_sets:
                    self.discrepancy_sets[ac] = set()
                self.discrepancy_sets[ac].add(tuple(cell))

            for node in info['closed']:
                observation = node.obs
                gval = node._g
                # self.value_dict[tuple(observation['observation'])
                #                 ] = info['best_node_f'] - gval - self.get_euclidean_heuristic(observation['observation'])
                self.value_dict[tuple(observation['observation'])
                                ] = info['best_node_f'] - gval - self.get_manhattan_heuristic(observation['observation'])

            # Check goal
            if self.env.check_goal(next_observation['observation']):
                self.env.execute_goal_completion()
                print('Reached goal')
                print('======================================================')
                self.env.recreate_object()
                current_observation = copy.deepcopy(
                    self.env.get_current_observation())
                continue
                # break

            # Update current observation
            current_observation = copy.deepcopy(next_observation)

        end = time.time()
        print('Finished in time', end-start, 'secs')
        self.env.wait_for_user()
        return total_n_steps
