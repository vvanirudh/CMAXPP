import time
import copy

from szeth.agents.pr2.pr2_3d_agent import pr2_3d_agent


class pr2_3d_cmaxeff_agent(pr2_3d_agent):
    def __init__(self, args, env, controller):
        pr2_3d_agent.__init__(self, args, env, controller)

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

            next_sim_observation = info['successor_obs']
            print('Predicted next cell', next_sim_observation['observation'])

            # Is there a discrepancy?
            discrepancy_found = self.check_discrepancy(current_observation, ac, next_observation,
                                                       next_sim_observation)
            if discrepancy_found:
                print('Discrepancy!')

            num_iterative_updates = 10
            for _ in range(num_iterative_updates):
                sampled_obs = self.sample_state_buffer()
                _, info = self.controller.act(copy.deepcopy(sampled_obs))
                self.update_state_value(info)

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
