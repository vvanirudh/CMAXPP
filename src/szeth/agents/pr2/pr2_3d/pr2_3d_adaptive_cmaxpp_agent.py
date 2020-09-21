import time
import copy

from szeth.agents.pr2.pr2_3d.pr2_3d_agent import pr2_3d_agent

from szeth.controllers.pr2.pr2_3d_q_controller import pr2_3d_q_controller
from szeth.controllers.pr2.pr2_3d_controller import pr2_3d_controller

from szeth.utils.bcolors import bcolors


class pr2_3d_adaptive_cmaxpp_agent(pr2_3d_agent):
    def __init__(self, args, env, controller, controller_inflated):
        pr2_3d_agent.__init__(self, args, env, controller, controller_inflated)
        assert isinstance(controller, pr2_3d_q_controller)
        assert isinstance(controller_inflated, pr2_3d_controller)
        self.controller.reconfigure_qvalue_fn(self.get_qvalue)

    def learn_online_in_real_world(self):
        # Reset environment
        current_observation = copy.deepcopy(self.env.get_current_observation())

        total_n_steps = 0
        max_attempts = self.args.max_attempts
        n_attempts = 0
        n_steps = []
        start = time.time()
        while True:
            print('-------------')
            print('Current cell', current_observation['observation'])
            print('Current cell heuristic',
                  self.get_state_value(current_observation['observation'], inflated=False))
            print('Current cell inflated heuristic',
                  self.get_state_value(current_observation['observation'], inflated=True))

            # Get action according to inflated value function
            ac_inflated, info_inflated = self.controller_inflated.act(
                copy.deepcopy(current_observation))
            # Get action according to non-inflated value function
            ac, info = self.controller.act(copy.deepcopy(current_observation))
            print('Inflated Action', ac_inflated)
            print('Non-Inflated Action', ac)
            executed_inflated_action = True

            if info_inflated['best_node_f'] <= (1 + self.alpha) * info['best_node_f']:
                # CMAX action
                # Step in the environment
                print(bcolors.OKGREEN+'Following inflated cost-to-go'+bcolors.ENDC)
                executed_inflated_action = True
                ac_chosen = ac_inflated
            else:
                # inflated value has a better cost-to-go than non-inflated value
                # Step in the environment
                print(bcolors.OKBLUE +
                      'Following non-inflated cost-to-go'+bcolors.ENDC)
                executed_inflated_action = False
                ac_chosen = ac

            # Step in the environment
            next_observation, cost = self.env.step(ac_chosen)
            print('True next cell', next_observation['observation'])
            total_n_steps += 1

            # Update qvalues based on the online execution
            self.update_qvalue(current_observation,
                               ac_chosen, cost, next_observation)

            # Add to buffers
            self.add_to_transition_buffer(
                current_observation, ac_chosen, cost, next_observation)
            self.add_to_state_buffer(current_observation)

            # INFLATED
            if executed_inflated_action:
                next_sim_observation = info_inflated['successor_obs']
                if next_sim_observation is None:
                    print('Executed a previously known to be incorrect transition')
                    # The next successor is unknown to the model
                    # Already discovered discrepancy
                    pass
                else:
                    print('Predicted next cell',
                          next_sim_observation['observation'])
                    discrepancy_found = self.check_discrepancy(
                        current_observation, ac_chosen, next_observation, next_sim_observation)
                    if discrepancy_found:
                        print('Discrepancy!')
                        # Replan to account for discrepancy updates
                        _, info_inflated = self.controller_inflated.act(
                            copy.deepcopy(current_observation))

            # Update all nodes on closed list
            self.update_state_value(info_inflated, inflated=True)

            # NON-INFLATED
            if not executed_inflated_action:
                next_sim_observation = info['successor_obs']
                if next_sim_observation is None:
                    print('Executed a previously known to be incorrect transition')
                    # The next successor is unknown to the model
                    # Already discovered discrepancy
                    pass
                else:
                    print('Predicted next cell',
                          next_sim_observation['observation'])
                    discrepancy_found = self.check_discrepancy(
                        current_observation, ac_chosen, next_observation, next_sim_observation)
                    if discrepancy_found:
                        print('Discrepancy!')
                        # Replan to account for discrepancy updates
                        _, info = self.controller.act(
                            copy.deepcopy(current_observation))

            # Update all nodes on closed list
            self.update_state_value(info, inflated=False)
            # Update qvalues based on the online execution
            self.update_buffer_qvalues()

            num_iterative_updates = self.args.num_updates
            for _ in range(num_iterative_updates):
                sampled_obs = self.sample_state_buffer()
                _, info = self.controller.act(
                    copy.deepcopy(sampled_obs))
                _, info_inflated = self.controller_inflated.act(
                    copy.deepcopy(sampled_obs))
                self.update_state_value(info_inflated, inflated=True)
                self.update_state_value(info, inflated=False)
                self.update_buffer_qvalues()

            # Check goal
            check_goal = self.env.check_goal(next_observation['observation'])
            max_timesteps = total_n_steps >= self.args.max_timesteps
            if check_goal or max_timesteps:
                n_steps.append(total_n_steps)
                total_n_steps = 0
                if check_goal:
                    self.env.execute_goal_completion()
                    # Decrease alpha
                    self.alpha = self.alpha / 2
                    print(bcolors.HEADER+'Changed alpha to ' +
                          str(self.alpha)+bcolors.ENDC)
                    print('Reached goal')
                if max_timesteps:
                    print('Maxed out number of steps')
                    break
                print('======================================================')
                self.env.recreate_object(place_failed=max_timesteps)
                current_observation = copy.deepcopy(
                    self.env.get_current_observation())
                n_attempts += 1
                if n_attempts == max_attempts:
                    break
                continue
                # break

            # Update current observation
            current_observation = copy.deepcopy(next_observation)

        end = time.time()
        print('Finished in time', end-start, 'secs')
        # self.env.wait_for_user()
        return n_steps
