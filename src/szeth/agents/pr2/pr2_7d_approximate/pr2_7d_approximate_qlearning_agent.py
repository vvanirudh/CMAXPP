import time
import copy
import numpy as np

from szeth.agents.pr2.pr2_7d_approximate.pr2_7d_approximate_agent import pr2_7d_approximate_agent
from szeth.agents.pr2.pr2_7d_approximate.features import compute_heuristic

from szeth.controllers.pr2.pr2_7d_qlearning_controller import pr2_7d_qlearning_controller

from szeth.utils.bcolors import print_warning, print_green, print_fail


class pr2_7d_approximate_qlearning_agent(pr2_7d_approximate_agent):
    def __init__(self, args, env, controller):
        pr2_7d_approximate_agent.__init__(self, args, env, controller)
        assert isinstance(controller, pr2_7d_qlearning_controller)
        self.controller.reconfigure_qvalues_fn(self.get_qvalues_lookahead)

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

            ac, _ = self.controller.act(copy.deepcopy(current_observation))
            if self.rng_exploration.rand() < self.args.epsilon:
                ac_idx = self.rng.randint(len(self.actions))
                ac = self.actions[ac_idx]
            print('Action', ac)
            print('Current state-action value',
                  self.get_qvalue(current_observation, ac))
            print('Current cell heuristic', compute_heuristic(
                current_observation['observation'],
                current_observation['desired_goal'],
                self.args.goal_threshold))

            # Step in the environment
            next_observation, cost = self.env.step(ac)
            print('True next cell', next_observation['observation'])
            total_n_steps += 1
            current_n_steps += 1

            self.add_to_transition_buffer(n_attempts, current_observation,
                                          ac, cost, next_observation)

            if (current_n_steps + 1) % self.args.qlearning_update_freq == 0:
                for _ in range(self.args.num_updates * self.args.qlearning_update_freq):
                    self.update_state_action_value_residual_qlearning()
                    self.update_target_networks()

            # Check goal
            check_goal = self.env.check_goal(next_observation)
            max_timesteps = current_n_steps >= self.args.max_timesteps
            if check_goal or max_timesteps:
                n_steps.append(current_n_steps)
                current_n_steps = 0
                if check_goal:
                    print_green('Reached goal in '+str(n_steps)+' steps')
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
