import copy
import ray
import numpy as np
import torch
import pickle
from collections import deque

from szeth.agents.pr2.pr2_7d_approximate.approximators import (
    StateValueResidual, get_state_value_residual,
    StateActionValueResidual, get_state_action_value_residual,
    DynamicsResidual, get_dynamics_residual,
    KNNDynamicsResidual, get_knn_dynamics_residual)
from szeth.agents.pr2.pr2_7d_approximate.features import (
    compute_features, compute_heuristic, FeatureNormalizer,
    compute_representation, compute_lookahead_heuristic)
from szeth.agents.pr2.pr2_7d_approximate.worker import (
    LookaheadWorker, compute_lookahead_heuristic_using_workers)

from szeth.utils.bcolors import print_bold, print_underline

from sklearn.neighbors import KDTree


class pr2_7d_approximate_agent:
    def __init__(self, args, env, controller, controller_inflated=None):
        self.args, self.env = args, env
        self.controller = controller
        self.controller_inflated = controller_inflated
        self.seed = self.args.seed
        if self.seed is None:
            self.seed = 0
        self.rng = np.random.RandomState(self.seed)
        self.rng_inflated = np.random.RandomState(self.seed)
        self.rng_q = np.random.RandomState(self.seed)
        self.rng_dyn = np.random.RandomState(self.seed)
        self.rng_exploration = np.random.RandomState(self.seed)

        self.start_cell = self.env.start_cell
        self.goal_cells = self.env.goal_cells
        self.actions = self.controller.actions
        self.actions_index = {}
        for ac_idx in range(len(self.actions)):
            self.actions_index[self.actions[ac_idx]] = ac_idx

        # Global function approximator for state value
        self.state_value_residual = StateValueResidual(in_dim=self.get_num_features(),
                                                       out_dim=1)
        self.state_value_target_residual = StateValueResidual(in_dim=self.get_num_features(),
                                                              out_dim=1)
        self.state_value_target_residual.load_state_dict(
            self.state_value_residual.state_dict())

        # Global function approximator for state action value
        self.state_action_value_residual = StateActionValueResidual(
            in_dim=self.get_num_features(),
            out_dim=len(self.actions))
        self.state_action_value_target_residual = StateActionValueResidual(
            in_dim=self.get_num_features(),
            out_dim=len(self.actions))
        self.state_action_value_target_residual.load_state_dict(
            self.state_action_value_residual.state_dict())

        # Global function approximator for inflated state value
        self.inflated_state_value_residual = StateValueResidual(
            in_dim=self.get_num_features(),
            out_dim=1)
        self.inflated_state_value_target_residual = StateValueResidual(
            in_dim=self.get_num_features(),
            out_dim=1)
        self.inflated_state_value_target_residual.load_state_dict(
            self.inflated_state_value_residual.state_dict())

        if self.args.agent == 'model':
            # Global function approximator for dynamics residual
            self.dynamics_residual = DynamicsResidual(in_dim=self.get_size_representation(),
                                                      num_actions=len(
                                                          self.actions),
                                                      out_dim=7)

        if self.args.agent == 'knn':
            # Local function approximators for dynamics residual
            self.knn_dynamics_residuals = [
                KNNDynamicsResidual(in_dim=7,
                                    radius=self.args.knn_radius,
                                    out_dim=7)
                for _ in range(len(self.actions))]

        # Local function approximator for discrepancy
        self.kdtrees = {}
        self.discrepancy_sets = {}
        for ac in self.actions:
            self.kdtrees[ac] = None
            self.discrepancy_sets[ac] = set()
        self.delta = self.args.delta
        self.inflated_states = set()

        # Optimizers
        self.state_value_residual_optim = torch.optim.Adam(
            self.state_value_residual.parameters(),
            lr=self.args.lr_value_residual,
            weight_decay=self.args.l2_reg_value_residual)
        self.state_action_value_residual_optim = torch.optim.Adam(
            self.state_action_value_residual.parameters(),
            lr=self.args.lr_value_residual,
            weight_decay=self.args.l2_reg_value_residual)
        self.inflated_state_value_residual_optim = torch.optim.Adam(
            self.inflated_state_value_residual.parameters(),
            lr=self.args.lr_value_residual,
            weight_decay=self.args.l2_reg_value_residual)
        if self.args.agent == 'model':
            self.dynamics_residual_optim = torch.optim.Adam(
                self.dynamics_residual.parameters(),
                lr=self.args.lr_dynamics_residual,
                weight_decay=self.args.l2_reg_dynamics_residual)

        # Normalizer
        self.feature_normalizer = FeatureNormalizer(self.get_num_features())
        self.feature_normalizer_q = FeatureNormalizer(self.get_num_features())
        if self.args.agent == 'model':
            self.representation_normalizer_dyn = FeatureNormalizer(
                self.get_size_representation())

        # Buffers
        self.max_buffer_size = 1000
        self.rollout_buffer = deque([], maxlen=self.max_buffer_size)
        self.rollout_buffer_inflated = deque([], maxlen=self.max_buffer_size)
        self.transition_buffer = []
        self.dynamics_transition_buffer = []
        self.batch_size = self.args.batch_size
        self.batch_size_q = self.args.batch_size_q
        self.batch_size_dyn = self.args.batch_size_dyn

        # Configure heuristic and discrepancy for controller
        def get_state_value(obs):
            return self.get_state_value(obs, inflated=False)
        self.controller.reconfigure_heuristic(get_state_value)
        self.controller.reconfigure_discrepancy(self.get_discrepancy)
        # if self.args.agent in ['cmaxpp', 'adaptive_cmaxpp']:
        #     self.controller.reconfigure_qvalue_fn(self.get_qvalue)

        # Configure heuristic and discrepancy for inflated controller
        if controller_inflated is not None:
            def get_state_value_inflated(obs):
                return self.get_state_value(obs, inflated=True)
            self.controller_inflated.reconfigure_heuristic(
                get_state_value_inflated)
            self.controller_inflated.reconfigure_discrepancy(
                self.get_discrepancy)

        # Workers
        self.workers = [LookaheadWorker.remote(
            self.args) for i in range(self.args.n_workers)]
        self.rollout_length = self.args.rollout_length

        # HER
        # self.future_p = 0.5
        self.future_p = self.args.her_ratio
        self.future_q_p = self.args.her_ratio

        # Anytime parameter
        self.alpha = self.args.alpha

    def get_state_value(self, obs, inflated=False):
        if self.env.check_goal(obs):
            return 0
        cell = obs['observation'].copy()
        goal_cell = obs['desired_goal'].copy()

        value = compute_heuristic(
            cell, goal_cell, self.args.goal_threshold)
        features = compute_features(
            cell, goal_cell, self.env.carry_cell,
            self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
            self.args.grid_size, self.env._grid_to_continuous)
        features_norm = self.feature_normalizer.normalize(features)

        # Use inflated if need be
        if inflated:
            state_value_residual = self.inflated_state_value_residual
        else:
            state_value_residual = self.state_value_residual

        residual_value = get_state_value_residual(
            features_norm, state_value_residual)
        return value + residual_value
        # return value

    def get_heuristic(self, obs):
        if self.env.check_goal(obs):
            return 0
        cell = obs['observation'].copy()
        goal_cell = obs['desired_goal'].copy()

        return compute_heuristic(cell, goal_cell, self.args.goal_threshold)

    def get_qvalues(self, obs):
        qvalues = []
        for ac in self.actions:
            qvalues.append(self.get_qvalue(obs, ac))
        return qvalues

    def get_qvalue(self, obs, ac):
        if self.env.check_goal(obs):
            return 0
        cell = obs['observation'].copy()
        goal_cell = obs['desired_goal'].copy()
        value = compute_heuristic(cell, goal_cell, self.args.goal_threshold)
        features = compute_features(
            cell, goal_cell, self.env.carry_cell,
            self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
            self.args.grid_size, self.env._grid_to_continuous)
        features_norm = self.feature_normalizer_q.normalize(features)
        ac_idx = self.actions_index[ac]
        residual_state_action_value = get_state_action_value_residual(
            features_norm, ac_idx, self.state_action_value_residual)
        return value + residual_state_action_value

    def get_qvalues_lookahead(self, obs):
        qvalues = []
        for ac in self.actions:
            qvalues.append(self.get_qvalue_lookahead(obs, ac))
        return qvalues

    def get_qvalue_lookahead(self, obs, ac):
        if self.env.check_goal(obs):
            return 0
        cell = obs['observation'].copy()
        goal_cell = obs['desired_goal'].copy()
        value = compute_lookahead_heuristic(cell, goal_cell, ac, self.controller,
                                            self.args.goal_threshold)
        features = compute_features(
            cell, goal_cell, self.env.carry_cell,
            self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
            self.args.grid_size, self.env._grid_to_continuous)
        features_norm = self.feature_normalizer_q.normalize(features)
        ac_idx = self.actions_index[ac]
        residual_state_action_value = get_state_action_value_residual(
            features_norm, ac_idx, self.state_action_value_residual)
        return value + residual_state_action_value

    def get_dynamics_residual(self, obs, ac):
        cell = obs['observation'].copy()

        representation = compute_representation(
            cell, self.args.grid_size, self.env._grid_to_continuous)
        representation_norm = self.representation_normalizer_dyn.normalize(
            representation)

        ac_idx = self.actions_index[ac]

        residual_dynamics = get_dynamics_residual(representation_norm,
                                                  ac_idx,
                                                  self.dynamics_residual)
        return residual_dynamics

    def get_knn_dynamics_residual(self, obs, ac):
        cell = obs['observation'].copy()

        # representation = compute_representation(
        #     cell, self.args.grid_size, self.env._grid_to_continuous)

        ac_idx = self.actions_index[ac]

        residual_dynamics = get_knn_dynamics_residual(
            # representation,
            cell,
            self.knn_dynamics_residuals[ac_idx])

        return residual_dynamics

    def get_num_features(self):
        start_cell_features = compute_features(
            self.start_cell, self.goal_cells[0], self.env.carry_cell,
            self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
            self.args.grid_size, self.env._grid_to_continuous)
        return start_cell_features.shape[0]

    def get_size_representation(self):
        start_cell_representation = compute_representation(
            self.start_cell, self.args.grid_size, self.env._grid_to_continuous)
        return start_cell_representation.shape[0]

    def get_discrepancy(self, cell, ac, cost):
        if self.has_discrepancy(cell, ac):
            return 100
        return cost

    def has_discrepancy(self, cell, ac):
        if self.kdtrees[ac] is None:
            return False

        num_neighbors = self.kdtrees[ac].query_radius(
            cell.reshape(1, -1),
            self.delta,
            count_only=True).squeeze()

        return num_neighbors > 0

    def update_state_value_residual(self, inflated=False):
        # Sample batch of states
        observations = self._sample_batch(inflated)
        batch_size = len(observations)

        num_workers = self.args.n_workers
        if batch_size < num_workers:
            num_workers = batch_size
        num_per_worker = batch_size // num_workers
        # Put state value target residual in object store
        state_value_residual_state_dict_id = ray.put(
            self.state_value_target_residual.state_dict())
        # Put kdtrees in object store
        kdtrees_serialized_id = ray.put(pickle.dumps(self.kdtrees))
        # Put feature normalizer in object store
        feature_normalizer_state_dict_id = ray.put(
            self.feature_normalizer.state_dict())

        if self.args.agent in ['cmaxpp', 'adaptive_cmaxpp']:
            # Put feature normalizer q in object store
            feature_normalizer_q_state_dict_id = ray.put(
                self.feature_normalizer_q.state_dict())
            # Put state action value target residual in object store
            state_action_value_residual_state_dict_id = ray.put(
                self.state_action_value_target_residual.state_dict())
        else:
            feature_normalizer_q_state_dict_id = None
            state_action_value_residual_state_dict_id = None

        if self.args.agent == 'adaptive_cmaxpp':
            # Put inflated state value target residual in object store
            inflated_state_value_residual_state_dict_id = ray.put(
                self.inflated_state_value_target_residual.state_dict())
        else:
            inflated_state_value_residual_state_dict_id = None

        if self.args.agent == 'model':
            dynamics_residual_state_dict_id = ray.put(
                self.dynamics_residual.state_dict())
            representation_normalizer_dyn_state_dict_id = ray.put(
                self.representation_normalizer_dyn.state_dict())
        else:
            dynamics_residual_state_dict_id = None
            representation_normalizer_dyn_state_dict_id = None

        if self.args.agent == 'knn':
            knn_dynamics_residuals_serialized_id = ray.put(
                pickle.dumps(self.knn_dynamics_residuals))
        else:
            knn_dynamics_residuals_serialized_id = None

        results, count = [], 0
        for worker_id in range(num_workers):
            if worker_id == num_workers - 1:
                # last worker takes the remaining load
                num_per_worker = batch_size - count

            # Set parameters
            ray.get(self.workers[worker_id].set_worker_params.remote(
                state_value_residual_state_dict_id,
                kdtrees_serialized_id,
                feature_normalizer_state_dict_id,
                state_action_value_residual_state_dict_id,
                feature_normalizer_q_state_dict_id,
                inflated_state_value_residual_state_dict_id,
                dynamics_residual_state_dict_id,
                knn_dynamics_residuals_serialized_id,
                representation_normalizer_dyn_state_dict_id
            ))

            # send job
            results.append(self.workers[worker_id].lookahead_batch.remote(
                observations[count:count+num_per_worker], inflated))
            # Increment count
            count += num_per_worker
        # Check if all observations have been accounted for
        assert count == batch_size
        # Get all targets
        results = ray.get(results)
        target_infos = [item for sublist in results for item in sublist]

        cells = [k.obs['observation'].copy()
                 for info in target_infos for k in info['closed']]
        intended_goals = [k.obs['desired_goal'].copy()
                          for info in target_infos for k in info['closed']]
        assert len(cells) == len(intended_goals)
        heuristics = np.array([compute_heuristic(
            cells[i], intended_goals[i], self.args.goal_threshold) for i in range(len(cells))],
            dtype=np.float32)
        targets = np.array([info['best_node_f'] -
                            k._g for info in target_infos for k in info['closed']],
                           dtype=np.float32)
        residual_targets = targets - heuristics
        # Clip the residual targets such that the residual is always positive
        residual_targets = np.maximum(residual_targets, 0)
        # Clip the residual targets so that the residual is not super big
        residual_targets = np.minimum(residual_targets, 20)

        # Compute features of the cell
        features = np.array([compute_features(
            cells[i], intended_goals[i], self.env.carry_cell,
            self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
            self.args.grid_size, self.env._grid_to_continuous
        ) for i in range(len(cells))], dtype=np.float32)
        features_norm = self.feature_normalizer.normalize(features)

        loss = self._fit_state_value_residual(
            features_norm, residual_targets, inflated)
        # Update target network
        # if not inflated:
        #     self._update_target_network(self.state_value_target_residual,
        #                                 self.state_value_residual)
        # else:
        #     self._update_target_network(self.inflated_state_value_target_residual,
        #                                 self.inflated_state_value_residual)
        # Update normalizer
        self.feature_normalizer.update_normalizer(features)
        return loss

    def update_state_action_value_residual(self):
        if len(self.transition_buffer) == 0:
            # No transitions yet
            return
        # Sample a batch of transitions
        transitions = self._sample_transition_batch()

        cells = [transition['obs']['observation']
                 for transition in transitions]
        goal_cells = [transition['obs']['desired_goal']
                      for transition in transitions]
        actions = [transition['ac'] for transition in transitions]
        ac_idxs = np.array([self.actions_index[ac]
                            for ac in actions], dtype=np.int32)
        costs = np.array([transition['cost']
                          for transition in transitions], dtype=np.float32)
        cells_next = [transition['obs_next']['observation']
                      for transition in transitions]
        goal_cells_next = [transition['obs_next']['desired_goal']
                           for transition in transitions]
        heuristics = np.array(
            [compute_heuristic(cells[i],
                               goal_cells[i],
                               self.args.goal_threshold) for i in range(len(cells))],
            dtype=np.float32)
        heuristics_next = np.array(
            [compute_heuristic(cells_next[i],
                               goal_cells_next[i],
                               self.args.goal_threshold) for i in range(len(cells))],
            dtype=np.float32)
        features = np.array(
            [compute_features(cells[i], goal_cells[i], self.env.carry_cell,
                              self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
                              self.args.grid_size, self.env._grid_to_continuous)
             for i in range(len(cells))],
            dtype=np.float32)
        features_norm = self.feature_normalizer_q.normalize(features)

        features_next = np.array(
            [compute_features(cells_next[i], goal_cells_next[i], self.env.carry_cell,
                              self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
                              self.args.grid_size, self.env._grid_to_continuous)
             for i in range(len(cells))],
            dtype=np.float32)
        features_next_norm = self.feature_normalizer.normalize(features_next)

        # Compute next state value
        features_next_norm_tensor = torch.from_numpy(features_next_norm)
        with torch.no_grad():
            residual_next_tensor = self.state_value_target_residual(
                features_next_norm_tensor)
            residual_next = residual_next_tensor.detach().numpy().squeeze()
        value_next = residual_next + heuristics_next

        # Compute targets
        targets = costs + value_next
        residual_targets = targets - heuristics
        # Clip the residual targets such that the residual is always positive
        residual_targets = np.maximum(residual_targets, 0)
        # Clip the residual targets so that the residual is not super big
        residual_targets = np.minimum(residual_targets, 20)

        loss = self._fit_state_action_value_residual(
            features_norm, ac_idxs, residual_targets)
        # Update normalizer
        self.feature_normalizer_q.update_normalizer(features)
        self.feature_normalizer.update_normalizer(features_next)

        return loss

    def update_state_action_value_residual_qlearning(self):
        if len(self.transition_buffer) == 0:
            # No transitions yet
            return

        # Sample a batch of transitions
        transitions = self._sample_transition_batch()

        cells = [transition['obs']['observation']
                 for transition in transitions]
        goal_cells = [transition['obs']['desired_goal']
                      for transition in transitions]
        actions = [transition['ac'] for transition in transitions]
        ac_idxs = np.array([self.actions_index[ac]
                            for ac in actions], dtype=np.int32)
        costs = np.array([transition['cost']
                          for transition in transitions], dtype=np.float32)
        cells_next = [transition['obs_next']['observation']
                      for transition in transitions]
        goal_cells_next = [transition['obs_next']['desired_goal']
                           for transition in transitions]
        heuristics = np.array(
            compute_lookahead_heuristic_using_workers(self.workers, cells,
                                                      goal_cells, actions),
            dtype=np.float32)
        # heuristics = np.array(
        #     [compute_lookahead_heuristic(cells[i], goal_cells[i], actions[i],
        #                                  self.controller, self.args.goal_threshold)
        #      for i in range(len(cells))]
        # )

        # heuristics_next = np.array(
        #     [compute_heuristic(cells_next[i],
        #                        goal_cells_next[i],
        #                        self.args.goal_threshold) for i in range(len(cells))],
        #     dtype=np.float32)
        heuristics_next = []
        for ac in self.actions:
            heuristics_next.append(
                compute_lookahead_heuristic_using_workers(
                    self.workers, cells_next,
                    goal_cells_next,
                    [ac for _ in range(len(cells_next))])
            )
            # heuristics_next.append(
            #     [compute_lookahead_heuristic(cells_next[i], goal_cells_next[i],
            #                                  ac, self.controller, self.args.goal_threshold)
            #      for i in range(len(cells_next))]
            # )
        heuristics_next = np.transpose(
            np.array(heuristics_next, dtype=np.float32))
        features = np.array(
            [compute_features(cells[i], goal_cells[i], self.env.carry_cell,
                              self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
                              self.args.grid_size, self.env._grid_to_continuous)
             for i in range(len(cells))],
            dtype=np.float32)
        features_norm = self.feature_normalizer_q.normalize(features)

        features_next = np.array(
            [compute_features(cells_next[i], goal_cells_next[i], self.env.carry_cell,
                              self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
                              self.args.grid_size, self.env._grid_to_continuous)
             for i in range(len(cells))],
            dtype=np.float32)
        features_next_norm = self.feature_normalizer_q.normalize(features_next)

        # Compute next state value using the target state action value residual
        features_next_norm_tensor = torch.from_numpy(features_next_norm)
        with torch.no_grad():
            qvalues_target_residual_next = self.state_action_value_target_residual(
                features_next_norm_tensor).detach().numpy()
            # Double Q-learning update
            qvalues_residual_next = self.state_action_value_residual(
                features_next_norm_tensor).detach().numpy()
            target_ac = np.argmin(
                qvalues_residual_next + heuristics_next, axis=1)
            qvalues_target_residual_next_chosen = np.take_along_axis(
                qvalues_target_residual_next, target_ac.reshape(-1, 1), axis=1).squeeze()
            heuristics_next_chosen = np.take_along_axis(
                heuristics_next, target_ac.reshape(-1, 1), axis=1).squeeze()
            qvalues_target_next = qvalues_target_residual_next_chosen + \
                heuristics_next_chosen

        # Compute targets
        targets = costs + qvalues_target_next
        residual_targets = targets - heuristics
        # Clip the residual targets such that the residual is always positive
        residual_targets = np.maximum(residual_targets, 0)
        # Clip the residual targets so that the residual is not super big
        residual_targets = np.minimum(residual_targets, 20)

        loss = self._fit_state_action_value_residual(
            features_norm, ac_idxs, residual_targets)
        # Update normalizer
        self.feature_normalizer_q.update_normalizer(features)
        self.feature_normalizer_q.update_normalizer(features_next)

        return loss

    def update_dynamics_residual(self):
        if len(self.dynamics_transition_buffer) == 0:
            # No transitions yet
            return

        # Sample a batch of transitions
        transitions = self._sample_dynamics_transition_batch()
        observations = [transition['obs'] for transition in transitions]
        cells = [obs['observation'] for obs in observations]
        representations = np.array([compute_representation(
            cell, self.args.grid_size, self.env._grid_to_continuous) for cell in cells],
            dtype=np.float32)

        observations_next = [transition['obs_next']
                             for transition in transitions]
        observations_next_sim = [transition['obs_next_sim']
                                 for transition in transitions]
        actions = [transition['ac'] for transition in transitions]
        ac_idxs = np.array([self.actions_index[ac]
                            for ac in actions], dtype=np.int32)
        continuous_states_next = [self.env._grid_to_continuous(
            obs['observation']) for obs in observations_next]

        model_continuous_states_next = [
            self.env._grid_to_continuous(obs['observation']) for obs in observations_next_sim]

        # Compute targets
        targets = np.array(continuous_states_next, dtype=np.float32) - \
            np.array(model_continuous_states_next, dtype=np.float32)

        # Compute normalized states
        representations_norm = self.representation_normalizer_dyn.normalize(
            representations)
        # Fit residual
        loss = self._fit_dynamics_residual(
            representations_norm, ac_idxs, targets)
        # Update normalizer
        self.representation_normalizer_dyn.update_normalizer(representations)

        return loss

    def update_knn_dynamics_residual(self):
        if len(self.dynamics_transition_buffer) == 0:
            # no transitions yet
            return

        transitions = self.dynamics_transition_buffer
        observations = [transition['obs'] for transition in transitions]
        cells = np.array([obs['observation'] for obs in observations])

        observations_next = [transition['obs_next']
                             for transition in transitions]
        observations_next_sim = [transition['obs_next_sim']
                                 for transition in transitions]
        actions = [transition['ac'] for transition in transitions]
        ac_idxs = np.array([self.actions_index[ac]
                            for ac in actions], dtype=np.int32)
        continuous_states_next = [self.env._grid_to_continuous(
            obs['observation']) for obs in observations_next]

        model_continuous_states_next = [
            self.env._grid_to_continuous(obs['observation']) for obs in observations_next_sim]

        # Compute targets
        targets = np.array(continuous_states_next, dtype=np.float32) - \
            np.array(model_continuous_states_next, dtype=np.float32)

        loss = 0
        for i in range(len(self.actions)):
            ac_mask = ac_idxs == i
            cells_mask = cells[ac_mask]
            target_mask = targets[ac_mask]

            if cells_mask.shape[0] == 0:
                # No data points for this action
                continue
            loss += self.knn_dynamics_residuals[i].fit(cells_mask, target_mask)

        print_underline('Dynamics Loss ' +
                        str(loss))
        return loss

    def _sample_batch(self, inflated=False, her=True):
        rollout_buffer = self.rollout_buffer if not inflated else self.rollout_buffer_inflated
        rng = self.rng if not inflated else self.rng_inflated
        rollout_buffer_size = len(rollout_buffer)
        observations = []
        count = 0
        # for i in range(self.batch_size):
        while True:
            idx = rng.randint(0, rollout_buffer_size)
            rollout_length = len(rollout_buffer[idx])
            # rollout_length = self.args.rollout_length
            t_sample = rng.randint(
                rollout_length)
            observation = copy.deepcopy(rollout_buffer[idx][t_sample])
            if rng.uniform() < self.future_p and her:
                # Fake goal
                future_offset = rng.uniform(
                ) * (rollout_length - t_sample)
                future_offset = int(future_offset)
                future_t = t_sample + future_offset
                observation['desired_goal'] = rollout_buffer[
                    idx][future_t]['observation'].copy()

            # if self.env.check_goal(observation):
            #     continue

            # if not np.array_equal(observation['desired_goal'], observation['observation']):
            # Making sure that the current state is never the goal

            observations.append(copy.deepcopy(observation))
            count += 1

            if count == self.batch_size:
                break

        return observations

    def _sample_transition_batch(self, her=True):
        transition_buffer_size = len(self.transition_buffer)

        transitions = []
        count = 0
        # for i in range(self.batch_size):
        while True:
            idx = self.rng_q.randint(0, transition_buffer_size)
            trial_length = len(self.transition_buffer[idx])
            t_sample = self.rng_q.randint(trial_length)
            transition = copy.deepcopy(self.transition_buffer[idx][t_sample])
            if self.rng_q.uniform() < self.future_q_p and her:
                # Fake goal
                future_offset = self.rng_q.uniform() * (trial_length - t_sample)
                future_offset = int(future_offset)
                future_t = t_sample + future_offset
                # Change desired goal of obs, obs_next
                fake_goal = self.transition_buffer[idx][future_t]['obs_next']['observation']
                transition['obs']['desired_goal'] = fake_goal.copy()
                transition['obs_next']['desired_goal'] = fake_goal.copy()
                # Rewrite cost if necessary
                if self.env.check_goal(transition['obs']):
                    transition['cost'] = 0
                elif isinstance(transition['ac'], tuple):
                    # Need to check if cost will be 1 or 2
                    next_cell = self.env.sub_successor(transition['obs']['observation'],
                                                       transition['ac'][0])
                    next_obs = {'observation': next_cell,
                                'desired_goal': transition['obs']['desired_goal']}
                    if self.env.check_goal(next_obs):
                        transition['cost'] = 1

            transitions.append(copy.deepcopy(transition))
            count += 1

            if count == self.batch_size_q:
                break

        return transitions

    def _sample_dynamics_transition_batch(self):
        transition_buffer_size = len(self.dynamics_transition_buffer)

        transitions = []
        count = 0
        while True:
            idx = self.rng_dyn.randint(transition_buffer_size)
            transitions.append(self.dynamics_transition_buffer[idx])
            count += 1

            if count == self.batch_size_dyn:
                break

        return transitions

    def _fit_state_value_residual(self, features, targets, inflated=False):
        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        targets_tensor = torch.from_numpy(targets)
        # Compute predictions
        if not inflated:
            residual_tensor = self.state_value_residual(
                features_tensor).squeeze()
        else:
            residual_tensor = self.inflated_state_value_residual(
                features_tensor).squeeze()
        # Compute loss
        state_value_residual_loss = (
            residual_tensor - targets_tensor).pow(2).mean()
        # Backprop and step
        if not inflated:
            self.state_value_residual_optim.zero_grad()
            state_value_residual_loss.backward()
            self.state_value_residual_optim.step()
        else:
            self.inflated_state_value_residual_optim.zero_grad()
            state_value_residual_loss.backward()
            self.inflated_state_value_residual_optim.step()
        if not inflated:
            print_bold('State Value Loss ' +
                       str(state_value_residual_loss.detach().numpy()))
        else:
            print_bold('Inflated State Value Loss ' +
                       str(state_value_residual_loss.detach().numpy()))
        return state_value_residual_loss.detach().numpy()

    def _fit_state_action_value_residual(self, features, ac_idxs, targets):
        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        targets_tensor = torch.from_numpy(targets)
        ac_idxs_tensor = torch.as_tensor(ac_idxs, dtype=torch.long).view(-1, 1)
        # Compute predictions
        residual_tensor = self.state_action_value_residual(features_tensor)
        residual_tensor = residual_tensor.gather(1, ac_idxs_tensor).squeeze()
        # Compute loss
        state_action_value_residual_loss = (
            residual_tensor - targets_tensor).pow(2).mean()
        # Backprop and step
        self.state_action_value_residual_optim.zero_grad()
        state_action_value_residual_loss.backward()
        self.state_action_value_residual_optim.step()
        print_underline(
            'State Action Value Loss ' + str(state_action_value_residual_loss.detach().numpy()))
        return state_action_value_residual_loss.detach().numpy()

    def _fit_dynamics_residual(self, states, ac_idxs, targets):
        # Convert to tensors
        states_tensor = torch.from_numpy(states)
        targets_tensor = torch.from_numpy(targets)
        # Compute predictions
        residual_tensor = self.dynamics_residual(states_tensor, ac_idxs)
        # Compute loss
        dynamics_residual_loss = (
            residual_tensor - targets_tensor).pow(2).mean()
        # Backprop and step
        self.dynamics_residual_optim.zero_grad()
        dynamics_residual_loss.backward()
        self.dynamics_residual_optim.step()
        print_underline('Dynamics Loss ' +
                        str(dynamics_residual_loss.detach().numpy()))
        return dynamics_residual_loss.detach().numpy()

    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)
        return

    def update_target_networks(self, inflated=None):
        if inflated is None:
            self._update_target_network(
                self.state_value_target_residual, self.state_value_residual)
            self._update_target_network(
                self.inflated_state_value_target_residual, self.inflated_state_value_residual)
            self._update_target_network(
                self.state_action_value_target_residual, self.state_action_value_residual)
        elif inflated:
            self._update_target_network(
                self.inflated_state_value_target_residual, self.inflated_state_value_residual)
        else:
            self._update_target_network(
                self.state_value_target_residual, self.state_value_residual)
            self._update_target_network(
                self.state_action_value_target_residual, self.state_action_value_residual)
        return

    def add_to_rollout_buffer(self, rollout, inflated=False):
        if not inflated:
            self.rollout_buffer.append(rollout)
        else:
            self.rollout_buffer_inflated.append(rollout)
        return

    def add_to_transition_buffer(self, trial_num, obs, ac, cost, obs_next):
        # Assuming trial_num starts at 0
        if len(self.transition_buffer) <= trial_num:
            self.transition_buffer.append(
                deque([], maxlen=self.max_buffer_size))
        transition = {'obs': obs, 'ac': ac,
                      'cost': cost, 'obs_next': obs_next}
        self.transition_buffer[trial_num].append(copy.deepcopy(transition))
        return

    def add_to_dynamics_transition_buffer(self, obs, ac, cost, obs_next, obs_next_sim):
        transition = {'obs': obs, 'ac': ac,
                      'cost': cost, 'obs_next': obs_next,
                      'obs_next_sim': obs_next_sim}
        self.dynamics_transition_buffer.append(
            copy.deepcopy(transition))
        return

    def check_discrepancy(self, obs, ac, next_obs_sim, next_obs):
        if not np.array_equal(next_obs['observation'][:6], next_obs_sim['observation'][:6]):
            self.add_discrepancy(obs, ac)
            return True
        return False

    def add_discrepancy(self, obs, ac):
        cell = obs['observation'].copy()
        self.discrepancy_sets[ac].add(tuple(cell))
        self.kdtrees[ac] = KDTree(
            np.array(list(self.discrepancy_sets[ac])),
            metric='manhattan')
        return

    def rollout_in_model(self, observation, inflated=False):
        controller = self.controller if not inflated else self.controller_inflated
        rollout = []
        rollout.append(copy.deepcopy(observation))
        for _ in range(self.rollout_length):
            _, info = controller.act(copy.deepcopy(observation))
            next_observation = info['successor_obs']
            if next_observation is None:
                break
            if self.env.check_goal(next_observation):
                break
            observation = copy.deepcopy(next_observation)
            rollout.append(copy.deepcopy(observation))
        self.add_to_rollout_buffer(rollout, inflated)
        return

    def learn_online_in_real_world(self):
        raise NotImplementedError

    def update_state_action_value_residual_workers(self):
        if len(self.transition_buffer) == 0:
            # No incorrect transitions yet
            return

        # Sample a batch of transitions
        transitions = self._sample_transition_batch()
        # Get all the next observations as we need to query the controller
        # for their best estimate of cost-to-go
        observations_next = [transition['obs_next']
                             for transition in transitions]
        batch_size = len(observations_next)

        # Split jobs among workers
        num_workers = self.args.n_workers
        if batch_size < num_workers:
            num_workers = batch_size
        num_per_worker = batch_size // num_workers
        # Put state value residual in object store
        state_value_residual_state_dict_id = ray.put(
            self.state_value_target_residual.state_dict())
        # Put kdtrees in object store
        kdtrees_serialized_id = ray.put(pickle.dumps(self.kdtrees))
        # Put feature normalizer in object store
        feature_normalizer_state_dict_id = ray.put(
            self.feature_normalizer.state_dict())
        # Put feature normalizer q in object store
        feature_normalizer_q_state_dict_id = ray.put(
            self.feature_normalizer_q.state_dict())
        # Put state action value target residual in object store
        state_action_value_residual_state_dict_id = ray.put(
            self.state_action_value_target_residual.state_dict())

        results, count = [], 0
        for worker_id in range(num_workers):
            if worker_id == num_workers - 1:
                # last worker takes the remaining load
                num_per_worker = batch_size - count

            # Set parameters
            ray.get(self.workers[worker_id].set_worker_params.remote(
                state_value_residual_state_dict_id,
                kdtrees_serialized_id,
                feature_normalizer_state_dict_id,
                state_action_value_residual_state_dict_id,
                feature_normalizer_q_state_dict_id))

            # send job
            results.append(self.workers[worker_id].lookahead_batch.remote(
                observations_next[count:count+num_per_worker]))
            # Increment count
            count += num_per_worker
        # Check if all observations have been accounted for
        assert count == batch_size
        # Get all targets
        results = ray.get(results)
        target_infos = [item for sublist in results for item in sublist]

        cells = [transition['obs']['observation']
                 for transition in transitions]
        goal_cells = [transition['obs']['desired_goal']
                      for transition in transitions]
        actions = [transition['ac'] for transition in transitions]
        ac_idxs = np.array([self.actions_index[ac]
                            for ac in actions], dtype=np.int32)
        costs = np.array([transition['cost']
                          for transition in transitions], dtype=np.float32)
        heuristics = np.array(
            [compute_heuristic(cells[i],
                               goal_cells[i],
                               self.args.goal_threshold) for i in range(len(cells))],
            dtype=np.float32)
        features = np.array(
            [compute_features(cells[i], goal_cells[i], self.env.carry_cell,
                              self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
                              self.args.grid_size, self.env._grid_to_continuous)
             for i in range(len(cells))],
            dtype=np.float32)
        features_norm = self.feature_normalizer_q.normalize(features)

        # Get next state value
        value_next = np.array([info['best_node_f']
                               for info in target_infos], dtype=np.float32)
        assert value_next.shape[0] == heuristics.shape[0]

        # Compute targets
        targets = costs + value_next
        residual_targets = targets - heuristics
        # Clip the residual targets such that the residual is always positive
        residual_targets = np.maximum(residual_targets, 0)
        # Clip the residual targets so that the residual is not super big
        residual_targets = np.minimum(residual_targets, 20)

        loss = self._fit_state_action_value_residual(
            features_norm, ac_idxs, residual_targets)
        # Update normalizer
        self.feature_normalizer_q.update_normalizer(features)
        return loss
