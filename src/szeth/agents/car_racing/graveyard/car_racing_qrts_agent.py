import copy
import numpy as np
import pickle
import os

from szeth.agents.car_racing.car_racing_agent import car_racing_agent
from szeth.envs.car_racing.car_racing_env_revised import THETA_DISCRETIZATION
from szeth.controllers.car_racing.car_q_controller import CarRacingQController


class car_racing_qrts_agent(car_racing_agent):
    def __init__(self, args, env, controller):
        car_racing_agent.__init__(self, args, env, controller)
        assert isinstance(controller, CarRacingQController)
        self.controller.reconfigure_qvalue_fn(self.get_qvalue)

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
            # print('STATE', current_obs['observation'],
            #       'MPRIM', mprim.id, info['dummy'])
            next_obs, reward, _, _ = self.env.step_mprim(
                mprim, render=render)
            total_return = total_return - reward
            total_n_steps += 1

            # Update qvalues based on the online execution
            self.update_qvalue(current_obs, idx, -reward, next_obs,
                               use_state_value=True)

            cmax_steps += 1.0
            successor_obs_sim = info['successor_obs']
            if successor_obs_sim is None:
                # The next successor is unknown to the model
                # Already discovered discrepancy
                pass
            else:
                # Check for discrepancy
                discrepancy_found = self.check_discrepancy(
                    current_obs,
                    mprim,
                    successor_obs_sim['observation'],
                    next_obs['observation']
                )
                if discrepancy_found:
                    _, info = self.controller.act(obs)

            # Update heuristic for all states on closed list
            self.update_value_fn(info, consistent=False)

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
                if save_value_fn:
                    # Save the value function
                    path = os.path.join(
                        os.environ['HOME'],
                        'workspaces/szeth_ws/src/szeth/save/value_fn.pkl')
                    pickle.dump(self.state_values, open(path, 'wb'))
                self.env.close()
                break

        return total_n_steps, lap_t_steps, lap_returns, percentage_cmax
