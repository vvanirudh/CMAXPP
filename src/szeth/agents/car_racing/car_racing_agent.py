import os
import pickle
import copy
import numpy as np
from collections import deque
from szeth.envs.car_racing.car_racing_env_revised import (X_DISCRETIZATION,
                                                          Y_DISCRETIZATION,
                                                          THETA_DISCRETIZATION)


class car_racing_agent:
    def __init__(self, args, env, controller, controller_inflated=None):
        self.args = args
        self.env = env
        self.controller = controller
        self.controller_inflated = controller_inflated

        self.env.reset()
        self.mprims = self.env.get_motion_primitives()
        self.cost_map = self.env.cost_map.copy()

        self.checkpoints = env.checkpoints
        self.num_checkpoints = len(env.checkpoints)
        self.last_checkpoint_index = None
        self.current_checkpoint_index = 0
        self.current_lap = 0

        # Load value functions
        path = os.path.join(os.environ['HOME'],
                            'workspaces/szeth_ws/src/szeth/save/value_fn.pkl')
        self.initial_state_values = pickle.load(open(path, 'rb'))
        self.state_values = pickle.load(open(path, 'rb'))
        self.inflated_state_values = pickle.load(open(path, 'rb'))
        path = os.path.join(os.environ['HOME'],
                            'workspaces/szeth_ws/src/szeth/save/qvalue_fn.pkl')
        self.qvalues = pickle.load(open(path, 'rb'))

        self.buffer = deque([])
        self.max_buffer_size = 1000

        self.discrepancy_dict = [{} for _ in range(self.num_checkpoints)]

        self.rng = np.random.RandomState(args.seed)

        # Set controller functions
        def get_state_value(obs):
            return self.get_state_value(obs, inflated=False)
        self.controller.reconfigure_heuristic(get_state_value)
        self.controller.set_checkpoint(
            self.checkpoints[self.current_checkpoint_index])
        self.controller.reconfigure_discrepancy(self.get_discrepancy)

        if controller_inflated is not None:
            def get_state_value_inflated(obs):
                return self.get_state_value(obs, inflated=True)
            self.controller_inflated.reconfigure_heuristic(
                get_state_value_inflated)
            self.controller_inflated.set_checkpoint(
                self.checkpoints[self.current_checkpoint_index])
            self.controller_inflated.reconfigure_discrepancy(
                self.get_discrepancy)

        # Anytime parameter
        self.alpha = args.alpha
        self.initial_alpha = args.alpha
        self.schedule = args.schedule
        self.time_step = 0  # Only for scheduling alpha in case of time schedule

    def _initialize_q_values(self):
        self.qvalues = []
        for checkpoint_idx in range(len(self.state_values)):
            state_values = self.state_values[checkpoint_idx]
            qvalues = {}
            for x in range(X_DISCRETIZATION):
                for y in range(Y_DISCRETIZATION):
                    for theta in range(THETA_DISCRETIZATION):
                        qvalues[x, y, theta] = [
                            state_values[x, y, theta]
                            for _ in range(len(self.mprims[theta]))]

            self.qvalues.append(qvalues)

        return

    def get_state_value(self, obs, inflated=False):
        current_obs = obs['observation']
        state_values = self.state_values if not inflated else self.inflated_state_values
        return state_values[
            self.current_checkpoint_index][tuple(current_obs)]

    def get_heuristic(self, obs):
        current_obs = obs['observation']
        return self.initial_state_values[
            self.current_checkpoint_index][tuple(current_obs)]

    def get_qvalues(self, obs):
        current_obs = obs['observation']
        current_heading = current_obs[2]
        mprims = self.mprims[current_heading]
        qvalues = []
        for mprim in mprims:
            qvalues.append(self.get_qvalue(obs, mprim))
        return qvalues

    def get_qvalue(self, obs, mprim):
        current_obs = obs['observation']
        current_heading = current_obs[2]
        mprim_index = self.mprims[current_heading].index(mprim)
        return self.qvalues[
            self.current_checkpoint_index][tuple(current_obs)][mprim_index]

    def get_qvalues_lookahead(self, obs):
        current_obs = obs['observation']
        current_heading = current_obs[2]
        mprims = self.mprims[current_heading]
        qvalues = []
        for mprim in mprims:
            qvalues.append(self.get_qvalue_lookahead(obs, mprim))
        return qvalues

    def get_qvalue_lookahead(self, obs, mprim):
        raise NotImplementedError
        cost, next_state_value = self.one_step_lookahead(obs, mprim)
        return cost + next_state_value

    def one_step_lookahead(self, obs, mprim):
        state = obs['observation']
        cost = 0
        for discrete_state in mprim.discrete_states:
            xd, yd, thetad = discrete_state
            state = np.array(
                [max(min(state[0] + xd, X_DISCRETIZATION-1), 0),
                 max(min(state[1] + yd, Y_DISCRETIZATION-1), 0),
                 thetad], dtype=int)
            cost += self.cost_map[state[0], state[1]]
        next_state_value = self.initial_state_values[
            self.current_checkpoint_index][tuple(state)]
        return cost, next_state_value

    def get_discrepancy(self, obs, mprim, cost):
        current_state = obs['observation']
        if mprim.id not in self.discrepancy_dict[
                self.current_checkpoint_index]:
            return cost
        else:
            if tuple(current_state[:2]) in self.discrepancy_dict[
                    self.current_checkpoint_index][mprim.id]:
                return 1e6
            else:
                return cost

    def update_value_fn(self, info, consistent=True, inflated=False):
        state_values = self.state_values if not inflated else self.inflated_state_values
        for node in info['closed']:
            state = node.obs['observation']
            gval = node._g

            if consistent and state_values[
                    self.current_checkpoint_index][
                        tuple(state)] > info['best_node_f'] - gval:
                print('Heuristic is decreasing', state_values[
                    self.current_checkpoint_index][tuple(
                        state)], info['best_node_f'] - gval)
                raise Exception('Heuristic has decreased. Should not happen')

            if not inflated:
                self.state_values[self.current_checkpoint_index][tuple(
                    state)] = info['best_node_f'] - gval
            else:
                self.inflated_state_values[self.current_checkpoint_index][tuple(
                    state)] = info['best_node_f'] - gval

        return

    def update_qvalue(self, obs, mprim_idx, cost, obs_next,
                      backup_trajectory=True):
        # Update the last transition first
        current_state = obs['observation'].copy()
        next_obs = copy.deepcopy(obs_next)

        new_qvalue = cost + self.get_state_value(next_obs, inflated=False)

        if new_qvalue > self.qvalues[
                self.current_checkpoint_index][
                    tuple(current_state)][mprim_idx]:
            self.qvalues[self.current_checkpoint_index][tuple(
                current_state)][mprim_idx] = new_qvalue

        if backup_trajectory:
            self.update_qvalues_in_buffer()

        # Add the transition to the buffer
        self.add_to_buffer(obs, mprim_idx, cost, obs_next)

        return

    def update_qvalues_in_buffer(self):
        for (o, midx, c, on) in reversed(self.buffer):
            current_state = o['observation'].copy()
            next_obs = copy.deepcopy(on)

            new_qvalue = c + self.get_state_value(next_obs, inflated=False)

            if new_qvalue > self.qvalues[
                    self.current_checkpoint_index][tuple(current_state)][midx]:
                self.qvalues[self.current_checkpoint_index][tuple(
                    current_state)][midx] = new_qvalue
        return

    def add_to_buffer(self, obs, mprim_idx, cost, obs_next):
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.popleft()
        self.buffer.append((obs, mprim_idx, cost, obs_next))
        return

    def update_checkpoint(self):
        finished_lap = False
        if self.current_checkpoint_index == 0:
            if self.last_checkpoint_index == self.num_checkpoints - 1:
                self.current_lap += 1
                finished_lap = True
                print('FINISHED LAP', self.current_lap)
        self.last_checkpoint_index = self.current_checkpoint_index
        self.current_checkpoint_index += 1
        if self.current_checkpoint_index >= self.num_checkpoints:
            self.current_checkpoint_index = 0

        self.controller.set_checkpoint(
            self.checkpoints[self.current_checkpoint_index])
        if self.controller_inflated is not None:
            self.controller_inflated.set_checkpoint(
                self.checkpoints[self.current_checkpoint_index])

        # Clear buffer
        # TODO: Can keep separate buffers for each checkpoint
        self.buffer = deque([])

        return finished_lap

    def check_deviation(self, obs_sim, obs):
        x_sim, y_sim, theta_sim = obs_sim
        x, y, theta = obs
        deviation = not np.array_equal(obs, obs_sim)
        return deviation

    def add_discrepancy(self, obs, mprim):
        current_state = obs['observation'].copy()
        if mprim.id not in self.discrepancy_dict[
                self.current_checkpoint_index]:
            self.discrepancy_dict[
                self.current_checkpoint_index][mprim.id] = set()

        self.discrepancy_dict[
            self.current_checkpoint_index][mprim.id].add(
                tuple(current_state[:2]))

        return

    def check_discrepancy(self, obs, mprim, next_obs_sim, next_obs):
        discrepancy = False
        if self.check_deviation(next_obs_sim, next_obs):
            # print('DISCREPANCY DETECTED', current_state, mprim.id, flush=True)
            discrepancy = True
            self.add_discrepancy(obs, mprim)

        return discrepancy

    def schedule_next_alpha(self):
        if self.schedule == 'linear':
            linear_step = (self.args.alpha) / self.args.max_laps
            self.alpha = self.alpha - linear_step
        elif self.schedule == 'exp':
            exp_step = self.args.exp_step
            self.alpha = self.alpha * exp_step
        elif self.schedule == 'time':
            self.time_step += 1
            self.alpha = self.initial_alpha / self.time_step
        elif self.schedule == 'step':
            self.step_freq = self.args.step_freq
            self.time_step += 1
            if (self.time_step) % self.step_freq == 0:
                step_size = (self.args.alpha) * \
                    self.step_freq / self.args.max_laps
                self.alpha = self.alpha - step_size
        else:
            raise NotImplementedError

    def learn_online_in_real_world(self,
                                   max_timesteps=None,
                                   max_laps=None,
                                   cmax=True,
                                   render=True,
                                   save_value_fn=False):
        raise NotImplementedError
