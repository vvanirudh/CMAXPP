import copy
import numpy as np
import pickle
import os

from szeth.agents.car_racing.car_racing_agent import car_racing_agent


class car_racing_dts_agent(car_racing_agent):
    def __init__(self, args, env, controller):
        car_racing_agent.__init__(self, args, env, controller)
        self.initial_state_values = copy.deepcopy(self.state_values)
        self.best_state_value = np.inf
        # 0 - qlearning, 1 - cmax
        self.alphas = np.array([1, 1], dtype=np.float32)
        self.alphas[1] = 4
        # 0 - qlearning, 1 - cmax
        self.betas = np.array([1, 1], dtype=np.float32)
        self.C = 20.0

    def learn_online_in_real_world(self,
                                   max_timesteps=None,
                                   max_laps=None,
                                   cmax=True,
                                   render=True,
                                   save_value_fn=False):
        obs = self.env.get_observation()
        total_n_steps = 0
        total_return = 0
        cmax_steps = 0
        lap_returns = []
        lap_t_steps = []
        percentage_cmax = []
        while True:
            current_obs = copy.deepcopy(obs)
            mprims = self.mprims[current_obs['observation'][2]]
            mprim, info = self.controller.act(obs)
            idx = mprims.index(mprim)
            # sample from the beta distribution
            qlearning_expected_improvement = np.random.beta(self.alphas[0],
                                                            self.betas[0])
            cmax_expected_improvement = np.random.beta(self.alphas[1],
                                                       self.betas[1])
            qlearning_mprim = False
            if qlearning_expected_improvement > cmax_expected_improvement:
                qlearning_mprim = True
                # Q-learning mprim
                qvalues = self.get_qvalues(current_obs)
                min_qvalue = min(qvalues)
                idx = qvalues.index(min_qvalue)
                toss = self.rng.rand()
                if toss < self.epsilon:
                    idx = self.rng.randint(len(mprims))
                mprim = mprims[idx]

            next_obs, reward, _, _ = self.env.step_mprim(
                mprim, render=render)
            total_return = total_return - reward
            total_n_steps += 1

            # Check if we have discovered a better
            # state based on optimal values acc. to model
            if self.initial_state_values[
                    self.current_checkpoint_index
            ][tuple(next_obs['observation'])] < self.best_state_value:
                # Update posterior of the chosen strategy
                self.best_state_value = self.initial_state_values[
                    self.current_checkpoint_index
                ][tuple(next_obs['observation'])]
                if qlearning_mprim:
                    self.alphas[0] += 1
                else:
                    self.alphas[1] += 1
            else:
                if qlearning_mprim:
                    self.betas[0] += 1
                else:
                    self.betas[1] += 1

            # Normalize alphas and betas
            if self.alphas[0] + self.betas[0] > self.C:
                self.alphas[0] = (self.C / (self.C + 1)) * self.alphas[0]
                self.betas[0] = (self.C / (self.C + 1)) * self.betas[0]
            if self.alphas[1] + self.betas[1] > self.C:
                self.alphas[1] = (self.C / (self.C + 1)) * self.alphas[1]
                self.betas[1] = (self.C / (self.C + 1)) * self.betas[1]

            # CMAX
            if cmax and (not qlearning_mprim):
                cmax_steps += 1.0
                successor_obs_sim = info['successor_obs']['observation']
                # Check for discrepancy
                discrepancy_found = self.check_discrepancy(
                    current_obs,
                    mprim,
                    successor_obs_sim,
                    next_obs['observation']
                )
                if discrepancy_found:
                    # Do a simple RTAA* update
                    _, info = self.controller.act(current_obs)
            # Update heuristic for all states on closed list
            self.update_value_fn(info)
            # Update qvalues based on the online execution
            self.update_qvalue(current_obs, idx, -reward, next_obs)

            # Check goal
            if self.env.check_goal(next_obs['observation'],
                                   self.checkpoints[
                                       self.current_checkpoint_index]):
                print('t', total_n_steps)
                print('REACHED CHECKPOINT', self.current_checkpoint_index)
                finished_lap = self.update_checkpoint()
                self.best_state_value = np.inf
                if finished_lap:
                    lap_returns.append(total_return)
                    lap_t_steps.append(total_n_steps)
                    percentage_cmax.append(cmax_steps / total_n_steps)

            obs = copy.deepcopy(next_obs)

            timestep_condition = (
                max_timesteps and total_n_steps >= max_timesteps)
            lap_condition = (max_laps and self.current_lap >= max_laps)

            if timestep_condition or lap_condition:
                self.env.close()
                break

        return total_n_steps, lap_t_steps, lap_returns, percentage_cmax
