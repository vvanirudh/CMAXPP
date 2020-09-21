import numpy as np
import torch
import ray
import pickle

from szeth.agents.pr2.pr2_7d_approximate.approximators import (
    StateValueResidual, get_state_value_residual,
    StateActionValueResidual, get_state_action_value_residual,
    DynamicsResidual, get_dynamics_residual,
    KNNDynamicsResidual, get_knn_dynamics_residual)
from szeth.agents.pr2.pr2_7d_approximate.features import (
    compute_features, compute_heuristic, FeatureNormalizer,
    compute_representation, compute_lookahead_heuristic)


from szeth.controllers.pr2.pr2_7d_controller import pr2_7d_controller
from szeth.controllers.pr2.pr2_7d_q_controller import pr2_7d_q_controller
from szeth.controllers.pr2.pr2_7d_model_controller import pr2_7d_model_controller
from szeth.controllers.pr2.pr2_7d_qlearning_controller import pr2_7d_qlearning_controller
from szeth.envs.pr2.pr2_7d_xyzrpy_env import pr2_7d_xyzrpy_env


@ray.remote
class LookaheadWorker:
    def __init__(self, args):
        self.args = args
        self.env = pr2_7d_xyzrpy_env(
            args, mass=0.01, use_gui=False, no_dynamics=True)

        self.start_cell = self.env.start_cell
        self.goal_cells = self.env.goal_cells
        self.num_features = compute_features(
            self.start_cell, self.goal_cells[0], self.env.carry_cell,
            self.env.obstacle_cell_aa, self.env.obstacle_cell_bb,
            self.args.grid_size, self.env._grid_to_continuous).shape[0]
        self.representation_size = compute_representation(
            self.start_cell, self.args.grid_size, self.env._grid_to_continuous).shape[0]

        if self.args.agent == 'cmax':
            self.controller = pr2_7d_controller(
                self.env, num_expansions=self.args.num_expansions)
        elif self.args.agent == 'cmaxpp':
            self.controller = pr2_7d_q_controller(
                self.env, num_expansions=self.args.num_expansions)
        elif self.args.agent == 'adaptive_cmaxpp':
            self.controller = pr2_7d_q_controller(
                self.env, num_expansions=self.args.num_expansions)
            self.controller_inflated = pr2_7d_controller(
                self.env, num_expansions=self.args.num_expansions)
        elif self.args.agent in ['model', 'knn']:
            self.controller = pr2_7d_model_controller(
                self.env, num_expansions=self.args.num_expansions)
        elif self.args.agent == 'qlearning':
            self.controller = pr2_7d_qlearning_controller(self.env)

        self.actions = self.controller.actions
        self.actions_index = {}
        for ac_idx in range(len(self.actions)):
            self.actions_index[self.actions[ac_idx]] = ac_idx

        self.state_value_residual = StateValueResidual(in_dim=self.num_features,
                                                       out_dim=1)
        self.state_action_value_residual = StateActionValueResidual(in_dim=self.num_features,
                                                                    out_dim=len(self.actions))
        self.inflated_state_value_residual = StateValueResidual(
            in_dim=self.num_features,
            out_dim=1)

        if self.args.agent == 'model':
            # Global function approximator for dynamics residual
            self.dynamics_residual = DynamicsResidual(in_dim=self.representation_size,
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

        self.kdtrees = {}
        for ac in self.actions:
            self.kdtrees[ac] = None
        self.delta = self.args.delta

        self.feature_normalizer = FeatureNormalizer(self.num_features)
        self.feature_normalizer_q = FeatureNormalizer(self.num_features)
        self.representation_normalizer_dyn = FeatureNormalizer(
            self.representation_size)

        # Configure heuristic and discrepancy for controller
        def get_state_value(obs):
            return self.get_state_value(obs, inflated=False)
        self.controller.reconfigure_heuristic(get_state_value)
        self.controller.reconfigure_discrepancy(self.get_discrepancy)
        if self.args.agent in ['cmaxpp', 'adaptive_cmaxpp']:
            self.controller.reconfigure_qvalue_fn(self.get_qvalue)

        if self.args.agent == 'model':
            self.controller.reconfigure_residual_dynamics(
                self.get_dynamics_residual)
        if self.args.agent == 'knn':
            self.controller.reconfigure_residual_dynamics(
                self.get_knn_dynamics_residual)

        # Configure heuristic and discrepancy for controller_inflated
        if self.args.agent == 'adaptive_cmaxpp':
            def get_state_value_inflated(obs):
                return self.get_state_value(obs, inflated=True)
            self.controller_inflated.reconfigure_heuristic(
                get_state_value_inflated)
            self.controller_inflated.reconfigure_discrepancy(
                self.get_discrepancy)

    def set_worker_params(self, state_value_residual_state_dict,
                          kdtrees_serialized,
                          feature_normalizer_state_dict,
                          state_action_value_residual_state_dict=None,
                          feature_normalizer_q_state_dict=None,
                          inflated_state_value_residual_state_dict=None,
                          dynamics_residual_state_dict=None,
                          knn_dynamics_residuals_serialized=None,
                          representation_normalizer_dyn_state_dict=None):
        self.state_value_residual.load_state_dict(
            state_value_residual_state_dict)
        self.kdtrees = pickle.loads(kdtrees_serialized)
        self.feature_normalizer.load_state_dict(feature_normalizer_state_dict)
        if state_action_value_residual_state_dict is not None:
            self.state_action_value_residual.load_state_dict(
                state_action_value_residual_state_dict)
        if feature_normalizer_q_state_dict is not None:
            self.feature_normalizer_q.load_state_dict(
                feature_normalizer_q_state_dict)
        if inflated_state_value_residual_state_dict is not None:
            self.inflated_state_value_residual.load_state_dict(
                inflated_state_value_residual_state_dict)
        if dynamics_residual_state_dict is not None:
            self.dynamics_residual.load_state_dict(
                dynamics_residual_state_dict)
        if knn_dynamics_residuals_serialized is not None:
            self.knn_dynamics_residuals = pickle.loads(
                knn_dynamics_residuals_serialized)
        if representation_normalizer_dyn_state_dict is not None:
            self.representation_normalizer_dyn.load_state_dict(
                representation_normalizer_dyn_state_dict)
        return

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

    def get_discrepancy(self, cell, ac, cost):
        if self.has_discrepancy(cell, ac):
            return 1e6
        return cost

    def has_discrepancy(self, cell, ac):
        if self.kdtrees[ac] is None:
            return False

        num_neighbors = self.kdtrees[ac].query_radius(
            cell.reshape(1, -1),
            self.delta,
            count_only=True).squeeze()

        return num_neighbors > 0

    def lookahead(self, obs, inflated=False):
        controller = self.controller if not inflated else self.controller_inflated
        _, info = controller.act(obs)
        return info

    def lookahead_batch(self, observations, inflated=False):
        infos = []
        batch_size = len(observations)
        for i in range(batch_size):
            infos.append(self.lookahead(observations[i], inflated=inflated))
        return infos

    def lookahead_heuristic(self, cell, goal_cell, ac):
        return compute_lookahead_heuristic(cell, goal_cell, ac,
                                           self.controller, self.args.goal_threshold)

    def lookahead_heuristic_batch(self, cells, goal_cells, acs):
        batch_size = len(cells)
        heuristics = []
        for i in range(batch_size):
            heuristics.append(self.lookahead_heuristic(
                cells[i], goal_cells[i], acs[i]))

        return heuristics

    def rollout(self, observation, rollout_length):
        pass


def compute_lookahead_heuristic_using_workers(workers, cells, goal_cells, actions):
    batch_size = len(cells)
    num_workers = len(workers)
    if batch_size < num_workers:
        num_workers = batch_size
    num_per_worker = batch_size // num_workers

    results, count = [], 0
    for worker_id in range(num_workers):
        if worker_id == num_workers - 1:
            num_per_worker = batch_size - count
        results.append(
            workers[worker_id].lookahead_heuristic_batch.remote(
                cells[count:count+num_per_worker],
                goal_cells[count:count+num_per_worker],
                actions[count:count+num_per_worker]
            ))
        count += num_per_worker

    results = ray.get(results)
    heuristics = [h for sublist in results for h in sublist]
    return heuristics
