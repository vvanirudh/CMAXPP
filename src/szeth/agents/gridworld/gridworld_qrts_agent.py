from szeth.utils.simulation_utils import set_gridworld_state_and_goal
import numpy as np
import copy


class gridworld_qrts_agent:
    def __init__(self, args, env, planning_env, controller):
        self.args, self.env = args, env
        self.planning_env, self.controller = planning_env, controller

        self.num_actions = self.env.get_actions().shape[0]
        self.state_values = np.zeros((args.grid_size, args.grid_size))
        self._fill_state_values(env)
        self.discrepancy_matrix = np.zeros((4, args.grid_size, args.grid_size))

        self.qvalues = np.tile(self.state_values, (self.num_actions, 1, 1))

    def get_state_value(self, obs):
        current_state = obs['observation']
        return self.state_values[tuple(current_state)]

    def get_discrepancy(self, obs, ac):
        current_state = obs['observation']
        return self.discrepancy_matrix[ac, current_state[0], current_state[1]] > 0

    def _fill_state_values(self, env):
        goal_state = env.goal_state
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                # Fill with manhattan distances
                self.state_values[i, j] = np.abs(
                    goal_state - np.array([i, j])).sum()

        return True

    def get_qvalue(self, obs, ac):
        current_state = obs['observation']
        return self.qvalues[ac, current_state[0], current_state[1]]

    def learn_online_in_real_world(self, max_timesteps=None):
        obs = self.env.reset()

        self.controller.reconfigure_heuristic(self.get_state_value)
        self.controller.reconfigure_discrepancy(self.get_discrepancy)
        self.controller.reconfigure_qvalue_fn(self.get_qvalue)

        total_n_steps = 0
        while True:
            current_state = obs['observation'].copy()
            ac, info_online = self.controller.act(obs)
            next_obs, cost, _, _ = self.env.step(ac)

            if self.args.verbose:
                print('t', total_n_steps)
                print('STATE', current_state,
                      self.env.grid[tuple(current_state)])
                print('ACTION', ac)
                print('VALUE PREDICTED', info_online['start_node_h'])
            if self.env.check_goal(next_obs['observation'],
                                   next_obs['desired_goal']):
                print('REACHED GOAL!')
                break
            total_n_steps += 1
            # # Get the next obs in planning env
            set_gridworld_state_and_goal(self.planning_env,
                                         obs['observation'],
                                         obs['desired_goal'])
            next_obs_sim, _, _, _ = self.planning_env.step(ac)
            if not np.array_equal(next_obs['observation'],
                                  next_obs_sim['observation']):
                # Report discrepancy
                self.discrepancy_matrix[ac,
                                        current_state[0], current_state[1]] += 1
                # Update Q-values
                self.qvalues[ac, current_state[0], current_state[1]
                             ] = cost + self.get_state_value(next_obs)
                # # Make sure Q-values and V-values are consistent
                if self.get_qvalue(obs, ac) < self.get_state_value(obs):
                    # Update state value
                    print('Making values consistent')
                    self.state_values[tuple(
                        current_state)] = self.get_qvalue(obs, ac)

            # Plan in model
            _, info = self.controller.act(obs)
            for node in info['closed']:
                state = node.obs['observation']
                gval = node._g
                self.state_values[tuple(state)] = info['best_node_f'] - gval

            # if tuple(current_state) == (4, 6) or tuple(current_state) == (4, 7):
            #     import ipdb
            #     ipdb.set_trace()
            obs = copy.deepcopy(next_obs)

            if max_timesteps and total_n_steps >= max_timesteps:
                break

        return total_n_steps
