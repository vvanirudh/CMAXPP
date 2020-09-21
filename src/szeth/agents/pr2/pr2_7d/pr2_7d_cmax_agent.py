import time
import copy
import numpy as np

from szeth.agents.pr2.pr2_7d.pr2_7d_agent import pr2_7d_agent

from szeth.controllers.pr2.pr2_7d_controller import pr2_7d_controller


class pr2_7d_cmax_agent(pr2_7d_agent):
    def __init__(self, args, env, controller):
        pr2_7d_agent.__init__(self, args, env, controller)
        assert isinstance(controller, pr2_7d_controller)

    def learn_online_in_real_world(self):
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

            # Add to buffers
            self.add_to_state_buffer(current_observation)

            next_sim_observation = info['successor_obs']
            print('Predicted next cell', next_sim_observation['observation'])

            # Is there a discrepancy?
            discrepancy_found = self.check_discrepancy(
                current_observation, ac, next_observation, next_sim_observation)
            if discrepancy_found:
                print('Discrepancy!')
                if np.array_equal(current_observation['observation'],
                                  next_observation['observation']):
                    print('BLOCKING DISCREPANCY')
                else:
                    print('NON-BLOCKING DISCREPANCY')
                # Replan
                _, info = self.controller.act(
                    copy.deepcopy(current_observation))

            # Update all nodes on closed list
            self.update_state_value(info)

            num_iterative_updates = self.args.num_updates
            for _ in range(num_iterative_updates):
                sampled_obs = self.sample_state_buffer()
                _, info = self.controller.act(copy.deepcopy(sampled_obs))
                self.update_state_value(info)

            # Check goal
            check_goal = self.env.check_goal(next_observation['observation'])
            max_timesteps = total_n_steps >= self.args.max_timesteps
            if check_goal or max_timesteps:
                n_steps.append(total_n_steps)
                total_n_steps = 0
                if check_goal:
                    self.env.execute_goal_completion()
                    print('Reached goal')
                if max_timesteps:
                    print('Maxed out number of steps')
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
        return n_steps
