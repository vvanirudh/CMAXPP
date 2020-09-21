import time
import copy
import numpy as np

from szeth.agents.pr2.pr2_7d_approximate.pr2_7d_approximate_agent import pr2_7d_approximate_agent
from szeth.agents.pr2.pr2_7d_approximate.features import compute_heuristic

from szeth.controllers.pr2.pr2_7d_q_controller import pr2_7d_q_controller
from szeth.controllers.pr2.pr2_7d_controller import pr2_7d_controller

from szeth.utils.bcolors import print_warning, print_green, print_fail, print_blue


class pr2_7d_approximate_adaptive_cmaxpp_agent(pr2_7d_approximate_agent):
    def __init__(self, args, env, controller, controller_inflated):
        pr2_7d_approximate_agent.__init__(
            self, args, env, controller, controller_inflated)
        assert isinstance(controller, pr2_7d_q_controller)
        assert isinstance(controller_inflated, pr2_7d_controller)
        self.controller.reconfigure_qvalue_fn(self.get_qvalue)

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
            ac_inflated, info_inflated = self.controller_inflated.act(
                copy.deepcopy(current_observation))
            print('Inflated action', ac_inflated)

            print('Current cell inflated value',
                  info_inflated['best_node_f'])
            print('Current cell non-inflated value',
                  info['best_node_f'])
            print('Current cell heuristic', compute_heuristic(
                current_observation['observation'],
                current_observation['desired_goal'],
                self.args.goal_threshold))

            if (info_inflated['best_node_f'] <= (1 + self.alpha) * info['best_node_f']) and (ac_inflated is not None):
                # CMAX action
                executed_inflated_action = True
                print_blue('Following inflated cost-to-go')
                ac_chosen = ac_inflated
            else:
                # CMAX++ action
                executed_inflated_action = False
                print_green('Following non-inflated cost-to-go')
                ac_chosen = ac

            if (info['best_node_f'] >= 100 and info_inflated['best_node_f'] >= 100):
                # ADAPTIVE CMAXPP is stuck
                print_fail("ADAPTIVE CMAXPP is stuck")
                n_steps.append(self.args.max_timesteps)
                current_n_steps = 0
                self.env.reset(goal=True)
                current_observation = copy.deepcopy(
                    self.env.get_current_observation(goal=True))
                n_attempts += 1
                break
            elif (not executed_inflated_action) and (info['best_node_f'] >= 100):
                # CMAXPP is stuck
                print_fail("CMAXPP is stuck")
                n_steps.append(self.args.max_timesteps)
                current_n_steps = 0
                self.env.reset(goal=True)
                current_observation = copy.deepcopy(
                    self.env.get_current_observation(goal=True))
                n_attempts += 1
                break
            elif executed_inflated_action and (info_inflated['best_node_f'] >= 100):
                # CMAX is stuck
                print_fail("CMAX is stuck")
                n_steps.append(self.args.max_timesteps)
                current_n_steps = 0
                self.env.reset(goal=True)
                current_observation = copy.deepcopy(
                    self.env.get_current_observation(goal=True))
                n_attempts += 1
                break

            if ac_chosen is not None:
                # Step in the environment
                next_observation, cost = self.env.step(ac_chosen)
                print('True next cell', next_observation['observation'])

                # Add to buffers
                # self.add_to_state_buffer(current_observation)
                self.add_to_transition_buffer(n_attempts, current_observation,
                                              ac_chosen, cost, next_observation)

                if not executed_inflated_action:
                    # CMAXPP
                    next_sim_observation = info['successor_obs']
                    if next_sim_observation is None:
                        # The next successor is unknown to the model
                        # Already discovered discrepancy
                        print_warning(
                            'Executed a previously known to be incorrect transition')
                        # self.add_to_transition_buffer(n_attempts, current_observation,
                        #                               ac_chosen, cost, next_observation)
                    else:
                        print('Predicted next cell',
                              next_sim_observation['observation'])
                        # Is there a discrepancy?
                        discrepancy_found = self.check_discrepancy(
                            current_observation, ac_chosen, next_observation, next_sim_observation)
                        if discrepancy_found:
                            print_warning('Discrepancy!')
                            # self.add_to_transition_buffer(n_attempts, current_observation,
                            #                               ac_chosen, cost, next_observation)
                            if np.array_equal(current_observation['observation'],
                                              next_observation['observation']):
                                print_warning('BLOCKING DISCREPANCY')
                            else:
                                print_warning('NON-BLOCKING DISCREPANCY')

                    # self.rollout_in_model(current_observation, inflated=False)
                    # for _ in range(self.args.num_updates):
                    #     for _ in range(self.args.num_updates_q):
                    #         self.update_state_action_value_residual()
                    #     self.update_state_value_residual(inflated=False)
                    #     self.update_target_networks(inflated=False)

                if executed_inflated_action:
                    # CMAX
                    next_sim_observation = info_inflated['successor_obs']
                    print('Predicted next cell',
                          next_sim_observation['observation'])
                    discrepancy_found = self.check_discrepancy(
                        current_observation, ac_chosen, next_observation, next_sim_observation)
                    if discrepancy_found:
                        print_warning('Discrepancy!')
                        # self.add_to_transition_buffer(n_attempts, current_observation,
                        #                               ac_chosen, cost, next_observation)
                        if np.array_equal(current_observation['observation'],
                                          next_observation['observation']):
                            print_warning('BLOCKING DISCREPANCY')
                        else:
                            print_warning('NON-BLOCKING DISCREPANCY')

                    # self.rollout_in_model(current_observation, inflated=True)
                    # for _ in range(self.args.num_updates):
                    #     self.update_state_value_residual(inflated=True)
                    #     self.update_target_networks(inflated=True)
            else:
                next_observation = copy.deepcopy(current_observation)

            self.rollout_in_model(current_observation, inflated=False)
            self.rollout_in_model(current_observation, inflated=True)
            for _ in range(self.args.num_updates):
                self.update_state_action_value_residual()
                self.update_state_value_residual(inflated=False)
                self.update_state_value_residual(inflated=True)
                self.update_target_networks()

            total_n_steps += 1
            current_n_steps += 1

            # Check goal
            check_goal = self.env.check_goal(next_observation)
            max_timesteps = current_n_steps >= self.args.max_timesteps
            if check_goal or max_timesteps:
                n_steps.append(current_n_steps)
                current_n_steps = 0
                if check_goal:
                    # Decrease alpha
                    self.alpha = self.alpha * 0.5
                    print_blue('Changed alpha to '+str(self.alpha))
                    print_green('Reached goal in '+str(n_steps[-1])+' steps')
                    print_green('Steps so far '+str(n_steps))
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
