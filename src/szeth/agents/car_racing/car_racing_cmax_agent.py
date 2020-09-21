import copy
import pickle
import os

from szeth.agents.car_racing.car_racing_agent import car_racing_agent
from szeth.controllers.car_racing.car_controller import CarRacingController


class car_racing_cmax_agent(car_racing_agent):
    def __init__(self, args, env, controller):
        car_racing_agent.__init__(self, args, env, controller)
        assert isinstance(controller, CarRacingController)

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
            mprim, info = self.controller.act(copy.deepcopy(current_obs))

            print('Current cell inflated value',
                  info['best_node_f'])
            print('Current cell heuristic', self.get_heuristic(current_obs))
            print('=============================================')

            next_obs, reward, _, _ = self.env.step_mprim(
                mprim, render=render)
            lap_return = lap_return - reward
            total_n_steps += 1
            n_steps += 1

            # CMAX
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
                _, info = self.controller.act(copy.deepcopy(current_obs))

            # Update heuristic for all states on closed list
            self.update_value_fn(info)

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

            timestep_condition = (
                max_timesteps and n_steps >= max_timesteps)
            lap_condition = (max_laps and self.current_lap >= max_laps)

            if timestep_condition or lap_condition:
                if save_value_fn:
                    # Save the value function
                    path = os.path.join(
                        os.environ['HOME'],
                        'workspaces/szeth_ws/src/szeth/save/value_fn.pkl')
                    pickle.dump(self.state_values, open(path, 'wb'))
                if timestep_condition:
                    lap_returns.append(lap_return)
                    lap_t_steps.append(n_steps)
                self.env.close()
                break

        return total_n_steps, lap_t_steps, lap_returns
