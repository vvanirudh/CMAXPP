import copy
import pickle
import os

from szeth.agents.car_racing.car_racing_agent import car_racing_agent
from szeth.controllers.car_racing.car_q_controller import CarRacingQController
from szeth.controllers.car_racing.car_controller import CarRacingController


class car_racing_adaptive_cmaxpp_agent(car_racing_agent):
    def __init__(self, args, env, controller, controller_inflated):
        car_racing_agent.__init__(
            self, args, env, controller, controller_inflated)
        assert isinstance(controller, CarRacingQController)
        assert isinstance(controller_inflated, CarRacingController)
        self.controller.reconfigure_qvalue_fn(self.get_qvalue)

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
            mprims = self.mprims[current_obs['observation'][2]]
            mprim, info = self.controller.act(copy.deepcopy(current_obs))
            mprim_inflated, info_inflated = self.controller_inflated.act(
                copy.deepcopy(current_obs))

            # print('Current cell inflated value',
            #       info_inflated['best_node_f'])
            # print('Current cell non-inflated value',
            #       info['best_node_f'])
            # print('Current cell heuristic', self.get_heuristic(current_obs))
            # print('=============================================')
            # input()

            # if info_inflated['best_node_f'] < info['best_node_f']:
            #     import ipdb
            #     ipdb.set_trace()

            idx = mprims.index(mprim)
            idx_inflated = mprims.index(mprim_inflated)

            executed_inflated_action = False
            if info_inflated['best_node_f'] <= (1 + self.alpha) * info['best_node_f']:
                # Execute CMAX action
                # print('Executed CMAX action')
                mprim_chosen = mprim_inflated
                idx_chosen = idx_inflated
                executed_inflated_action = True
            else:
                # Execute CMAX++ action
                # print('Executed CMAXPP action')
                mprim_chosen = mprim
                idx_chosen = idx
                executed_inflated_action = False

            next_obs, reward, _, _ = self.env.step_mprim(
                mprim_chosen, render=render)
            assert reward == - \
                self.controller.get_cost(current_obs, mprim_chosen)
            assert reward == - \
                self.controller_inflated.get_cost(current_obs, mprim_chosen)
            lap_return = lap_return - reward
            total_n_steps += 1
            n_steps += 1

            # Update qvalues based on the online execution
            self.update_qvalue(current_obs, idx_chosen, -reward, next_obs)

            if not executed_inflated_action:
                successor_obs_sim = info['successor_obs']
                if successor_obs_sim is None:
                    # The next successor is unknown to the model
                    # Already discovered discrepancy
                    pass
                else:
                    # Check for discrepancy
                    discrepancy_found = self.check_discrepancy(
                        current_obs,
                        mprim_chosen,
                        successor_obs_sim['observation'],
                        next_obs['observation']
                    )
                    if discrepancy_found:
                        # print('Discrepancy!')
                        _, info = self.controller.act(
                            copy.deepcopy(current_obs))

            # Update heuristic for all states on closed list
            self.update_value_fn(info, consistent=False, inflated=False)
            # TODO: Could update qvalues again to reflect changes in state values

            if executed_inflated_action:
                successor_obs_sim = info_inflated['successor_obs']
                discrepancy_found = self.check_discrepancy(
                    current_obs,
                    mprim_chosen,
                    successor_obs_sim['observation'],
                    next_obs['observation'])
                if discrepancy_found:
                    # print('Discrepancy!')
                    _, info_inflated = self.controller_inflated.act(
                        copy.deepcopy(current_obs))

            # Update heuristic for all states on closed list
            self.update_value_fn(info_inflated, inflated=True)

            # Update qvalues in the buffer
            # self.update_qvalues_in_buffer()

            # Check goal
            if self.env.check_goal(next_obs['observation'],
                                   self.checkpoints[
                                       self.current_checkpoint_index]):
                print('t', total_n_steps)
                print('REACHED CHECKPOINT', self.current_checkpoint_index)
                finished_lap = self.update_checkpoint()
                if finished_lap:
                    # Update alpha
                    # self.alpha = self.alpha * 0.9
                    self.schedule_next_alpha()
                    print('CHANGING ALPHA TO', self.alpha)
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
