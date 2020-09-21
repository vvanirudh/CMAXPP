import pickle
import os
import numpy as np
import copy

from szeth.agents.car_racing.car_racing_agent import car_racing_agent


class car_racing_qlearning_agent(car_racing_agent):
    def __init__(self, args, env, controller):
        car_racing_agent.__init__(self, args, env, controller)

    def learn_online_in_real_world(self,
                                   max_timesteps=None,
                                   max_laps=None,
                                   render=True,
                                   save_value_fn=False):
        obs = self.env.get_observation()
        total_n_steps = 0
        n_steps = 0
        lap_return = 0
        lap_returns = []
        lap_t_steps = []
        current_obs = copy.deepcopy(obs)
        while True:
            current_heading = current_obs['observation'][2]
            # Find the qvalues
            qvalues = self.get_qvalues(current_obs)
            # Find the min qvalue
            min_qvalue = min(qvalues)
            idx = qvalues.index(min_qvalue)
            mprims = self.mprims[current_heading]
            # toss = self.rng.rand()
            # if toss < self.epsilon:
            #     idx = self.rng.randint(len(mprims))
            mprim = mprims[idx]
            next_obs, reward, _, _ = self.env.step_mprim(mprim, render=render)
            lap_return = lap_return - reward
            total_n_steps += 1
            n_steps += 1

            # Update q value fn
            self.update_qvalue(current_obs, idx, -reward, next_obs)

            # Check goal
            if self.env.check_goal(next_obs['observation'],
                                   self.checkpoints[
                                       self.current_checkpoint_index]):
                print('t', total_n_steps)
                print('REACHED CHECKPOINT', self.current_checkpoint_index)
                finished_lap = self.update_checkpoint()
                if finished_lap:
                    lap_returns.append(lap_return)
                    lap_t_steps.append(n_steps)
                    n_steps = 0
                    lap_return = 0

            current_obs = copy.deepcopy(next_obs)

            lap_condition = (max_laps and self.current_lap >= max_laps)
            timestep_condition = (
                max_timesteps and n_steps >= max_timesteps)
            if lap_condition or timestep_condition:
                if save_value_fn:
                    path = os.path.join(os.environ['HOME'],
                                        'workspaces/szeth_ws/src/szeth/save/qvalue_fn.pkl')
                    pickle.dump(self.qvalues, open(path, 'wb'))
                if timestep_condition:
                    lap_returns.append(lap_return)
                    lap_t_steps.append(n_steps)
                self.env.close()
                break

        return total_n_steps, lap_t_steps, lap_returns
