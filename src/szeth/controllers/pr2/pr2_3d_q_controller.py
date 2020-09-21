import numpy as np
import copy

from szeth.utils.node import QNode, Node
from szeth.utils.qastar import QAstar


class pr2_3d_q_controller:
    def __init__(self, model, num_expansions=3):
        self.model, self.num_expansions = model, num_expansions
        # self.model.reset()
        # self.actions = self.model.get_actions()
        # self.actions = self.model.get_extended_actions()
        self.actions = self.model.get_repeated_actions()

        self.astar = QAstar(self.heuristic,
                            self.get_qvalue,
                            # self.get_successors_no_kinematics,
                            self.check_discrepancy_fn,
                            self.get_successors,
                            self.check_goal,
                            num_expansions,
                            self.actions)

        self.heuristic_fn = None
        self.discrepancy_fn = None
        self.get_qvalue_fn = None

    def check_discrepancy_fn(self, obs, ac):
        if self.discrepancy_fn(obs['observation'], ac, 0) != 0:
            return True
        return False

    def get_qvalue(self, obs, ac):
        return self.get_qvalue_fn(obs['observation'], ac)

    def heuristic(self, node):
        if isinstance(node, QNode):
            obs = node.obs['observation']
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
        self.model.set_observation(obs)
        # Step the model
        next_obs, cost = self.model.step(ac)
        # Check if it was a previously known incorrect transition
        if self.discrepancy_fn is not None:
            cost = self.discrepancy_fn(obs['observation'],
                                       ac, cost)
            # We should not be querying successors for an already known to be incorrect transition
            assert not self.check_discrepancy_fn(obs, ac)

        # Create a node
        next_node = QNode(next_obs)
        # print(obs['observation'], ac, next_obs['observation'])

        return next_node, cost

    def get_successors_no_kinematics(self, node, ac):
        if isinstance(node, Node):
            obs = node.obs
        else:
            obs = node
        next_cell = self.model.successor(obs['observation'], ac)
        next_obs = {'observation': next_cell, 'sim_state': None}
        cost = self.model.get_cost(obs['observation'], ac, next_cell)

        # Create a node
        next_node = Node(next_obs)
        # Check if it was a previously known incorrect transition
        if self.discrepancy_fn is not None:
            cost = self.discrepancy_fn(obs['observation'],
                                       ac, cost)

        if isinstance(node, Node):
            return next_node, cost
        return next_obs, cost

    def check_goal(self, node):
        obs = node.obs['observation']
        return self.model.check_goal(obs)

    def act(self, obs):
        start_node = QNode(obs)
        best_action, info = self.astar.act(start_node)
        return best_action, info

    def reconfigure_heuristic(self, heuristic_fn):
        self.heuristic_fn = heuristic_fn
        return

    def reconfigure_discrepancy(self, discrepancy_fn):
        self.discrepancy_fn = discrepancy_fn
        return

    def reconfigure_qvalue_fn(self, get_qvalue_fn):
        self.get_qvalue_fn = get_qvalue_fn
        return
