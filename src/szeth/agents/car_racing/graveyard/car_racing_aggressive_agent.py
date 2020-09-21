import copy
import numpy as np
import pickle
import os

from szeth.agents.car_racing.car_racing_agent import car_racing_agent


class car_racing_aggressive_agent(car_racing_agent):
    def __init__(self, args, env, controller):
        car_racing_agent.__init__(self, args, env, controller)

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
            qvalues = self.get_qvalues(current_obs)
            min_qvalue = min(qvalues)
            qlearning_mprim = False
            if min_qvalue < info['best_node_f']:
                # Execute Q learning action
                idx = qvalues.index(min_qvalue)
                mprim = mprims[idx]
                qlearning_mprim = True

            next_obs, reward, _, _ = self.env.step_mprim(
                mprim, render=render)
            total_return = total_return - reward
            total_n_steps += 1

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
