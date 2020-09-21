import time
import copy

from szeth.agents.pr2.pr2_3d_agent import pr2_3d_agent
from szeth.controllers.pr2.pr2_3d_q_controller import pr2_3d_q_controller


class pr2_3d_cmaxppeff_agent(pr2_3d_agent):
    def __init__(self, args, env, controller):
        pr2_3d_agent.__init__(self, args, env, controller)
        # TODO: assert controller is of the right type
        assert isinstance(controller, pr2_3d_q_controller)
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
                  self.get_state_value(current_observation['observation']))

            ac, info = self.controller.act(copy.deepcopy(current_observation))
            print('Action', ac)
            # Step in the environment
            next_observation, cost = self.env.step(ac)
            print('True next cell', next_observation['observation'])
            total_n_steps += 1

            # Can update state values since we already planned once
            self.update_state_value(info)

            # Add to buffers
            self.add_to_state_buffer(current_observation)
            self.add_to_transition_buffer(
                current_observation, ac, cost, next_observation)

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
                    current_observation, ac, next_observation, next_sim_observation)
                if discrepancy_found:
                    print('Discrepancy!')

            # # Replan to account for discrepancy updates
            # _, info = self.controller.act(
            #     copy.deepcopy(current_observation))
            # # Update all nodes on closed list
            # self.update_state_value(info)
            # # Update qvalues based on the new updated state values
            # self.update_buffer_qvalues()
            num_iterative_updates = 10
            for _ in range(num_iterative_updates):
                # Sample a state from the state buffer
                # (can be multiple states, and can be done in parallel)
                sampled_obs = self.sample_state_buffer()
                # Replan from that sampled state
                _, info = self.controller.act(copy.deepcopy(sampled_obs))
                # Update all state values
                self.update_state_value(info)
                # Update qvalues based on the updated state values
                self.update_buffer_qvalues()

            # Check goal
            if self.env.check_goal(next_observation['observation']):
                n_steps.append(total_n_steps)
                total_n_steps = 0
                self.env.execute_goal_completion()
                print('Reached goal')
                print('======================================================')
                self.env.recreate_object()
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
