import time
import copy
import numpy as np

from szeth.agents.pr2.pr2_7d_approximate.pr2_7d_approximate_agent import pr2_7d_approximate_agent
from szeth.agents.pr2.pr2_7d_approximate.features import compute_heuristic

from szeth.controllers.pr2.pr2_7d_controller import pr2_7d_controller

from szeth.utils.bcolors import print_warning, print_green, print_fail


class pr2_7d_approximate_cmax_agent(pr2_7d_approximate_agent):
    def __init__(self, args, env, controller):
        pr2_7d_approximate_agent.__init__(self, args, env, controller)
        assert isinstance(controller, pr2_7d_controller)

    def learn_online_in_real_world(self):
        current_observation = copy.deepcopy(
            self.env.get_current_observation(goal=True))

        total_n_steps = 0
        current_n_steps = 0
        max_attempts = self.args.max_attempts
        n_attempts = 0
        n_steps = []
        start = time.time()
        while True:
            print('-------------')
            print('Current cell', current_observation['observation'])

            ac, info = self.controller.act(copy.deepcopy(current_observation))
            print('Action', ac)
            print('Current cell value',
                  info['best_node_f'])
            print('Current cell heuristic', compute_heuristic(
                current_observation['observation'],
                current_observation['desired_goal'],
                self.args.goal_threshold))

            if ac is None or info['best_node_f'] >= 100:
                # CMAX is stuck
                print_fail("CMAX is stuck")
                n_steps.append(self.args.max_timesteps)
                current_n_steps = 0
                self.env.reset(goal=True)
                current_observation = copy.deepcopy(
                    self.env.get_current_observation(goal=True))
                n_attempts += 1
                break

            # Step in the environment
            next_observation, cost = self.env.step(ac)
            print('True next cell', next_observation['observation'])
            total_n_steps += 1
            current_n_steps += 1

            # Add to buffers
            # self.add_to_state_buffer(current_observation)

            next_sim_observation = info['successor_obs']
            print('Predicted next cell', next_sim_observation['observation'])

            # Is there a discrepancy?
            discrepancy_found = self.check_discrepancy(
                current_observation, ac, next_observation, next_sim_observation)
            if discrepancy_found:
                print_warning('Discrepancy!')
                if np.array_equal(current_observation['observation'],
                                  next_observation['observation']):
                    print_warning('BLOCKING DISCREPANCY')
                else:
                    print_warning('NON-BLOCKING DISCREPANCY')
                # Replan
                # _, info = self.controller.act(
                #     copy.deepcopy(current_observation))

            self.rollout_in_model(current_observation)
            for _ in range(self.args.num_updates):
                # TODO: Should I pass in the current_observation?
                self.update_state_value_residual()
                self.update_target_networks()

            # Check goal
            check_goal = self.env.check_goal(next_observation)
            max_timesteps = current_n_steps >= self.args.max_timesteps
            if check_goal or max_timesteps:
                n_steps.append(current_n_steps)
                current_n_steps = 0
                if check_goal:
                    print_green('Reached goal')
                    print_green('Reached goal in '+str(n_steps[-1])+' steps')
                if max_timesteps:
                    print_fail('Maxed out number of steps')
                    break
                print('======================================================')
                self.env.reset(goal=True)
                current_observation = copy.deepcopy(
                    self.env.get_current_observation(goal=True))
                n_attempts += 1
                if n_attempts == max_attempts:
                    break
                continue
                # break

            # Update current observation
            current_observation = copy.deepcopy(next_observation)

        end = time.time()
        print_green('Finished in time '+str(end-start)+' secs')
        return n_steps
