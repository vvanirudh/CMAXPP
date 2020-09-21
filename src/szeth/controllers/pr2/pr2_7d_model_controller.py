import numpy as np
import copy

from szeth.utils.node import Node
from szeth.utils.astar import Astar


class pr2_7d_model_controller:
    def __init__(self, model, num_expansions=3):
        self.model, self.num_expansions = model, num_expansions
        self.actions = self.model.get_repeated_actions()

        self.astar = Astar(self.heuristic,
                           self.get_successors,
                           self.check_goal,
                           num_expansions,
                           self.actions)

        self.heuristic_fn = None
        self.residual_dynamics_fn = None

    def heuristic(self, node):
        if isinstance(node, Node):
            obs = node.obs
        else:
            obs = node

        if self.heuristic_fn is not None:
            value = self.heuristic_fn(obs)
        else:
            raise Exception('Heuristic fn not set')

        return value

    def get_successors(self, node, ac):
        obs = node.obs
        # Set the model to the sim state
        self.model.set_observation(obs, goal=True)
        # Step the model
        next_obs, cost = self.model.step(ac)
        # Check if residual dynamics is set
        if self.residual_dynamics_fn is not None:
            residual_correction = self.residual_dynamics_fn(obs, ac)
            # Convert next obs acc. to model to continuous
            continuous_next_state = self.model._grid_to_continuous(
                next_obs['observation'])
            # Add correction
            corrected_next_state = continuous_next_state + residual_correction
            # Convert continuous corrected next state to discrete
            next_obs['observation'] = self.model._continuous_to_grid(
                corrected_next_state)

        # Create a node
        next_node = Node(next_obs)

        return next_node, cost

    def check_goal(self, node):
        obs = node.obs
        return self.model.check_goal(obs)

    def act(self, obs):
        start_node = Node(obs)
        best_action, info = self.astar.act(start_node)
        return best_action, info

    def reconfigure_heuristic(self, heuristic_fn):
        self.heuristic_fn = heuristic_fn
        return

    def reconfigure_num_expansions(self, num_expansions):
        self.num_expansions = num_expansions
        self.astar.num_expansions = num_expansions
        return

    def reconfigure_residual_dynamics(self, residual_dynamics_fn):
        self.residual_dynamics_fn = residual_dynamics_fn
        return

    # dummy functions
    def reconfigure_discrepancy(self, discrepancy_fn):
        return
