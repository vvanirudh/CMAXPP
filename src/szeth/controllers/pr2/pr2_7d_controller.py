import numpy as np
import copy

from szeth.utils.node import Node
from szeth.utils.astar import Astar


class pr2_7d_controller:
    def __init__(self, model, num_expansions=3):
        self.model, self.num_expansions = model, num_expansions
        # self.model.reset()
        # self.actions = self.model.get_actions()
        # self.actions = self.model.get_extended_actions()
        self.actions = self.model.get_repeated_actions()

        self.astar = Astar(self.heuristic,
                           # self.get_successors_no_kinematics,
                           self.get_successors,
                           self.check_goal,
                           num_expansions,
                           self.actions)

        self.heuristic_fn = None
        self.discrepancy_fn = None

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
        # Check if it was a previously known incorrect transition
        if self.discrepancy_fn is not None:
            cost = self.discrepancy_fn(obs['observation'],
                                       ac, cost)

        # Create a node
        next_node = Node(next_obs)
        # print(obs['observation'], ac, next_obs['observation'])

        return next_node, cost

    # def get_successors_no_kinematics(self, node, ac):
    #     assert False, "Did not check this"
    #     if isinstance(node, Node):
    #         obs = node.obs
    #     else:
    #         obs = node
    #     next_cell = self.model.successor(obs['observation'], ac)
    #     next_obs = {'observation': next_cell, 'sim_state': None}
    #     cost = self.model.get_cost(obs['observation'], ac, next_cell)

    #     # Create a node
    #     next_node = Node(next_obs)
    #     # Check if it was a previously known incorrect transition
    #     if self.discrepancy_fn is not None:
    #         cost = self.discrepancy_fn(obs['observation'],
    #                                    ac, cost)

    #     if isinstance(node, Node):
    #         return next_node, cost
    #     return next_obs, cost

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

    def reconfigure_discrepancy(self, discrepancy_fn):
        self.discrepancy_fn = discrepancy_fn
        return

    def reconfigure_num_expansions(self, num_expansions):
        self.num_expansions = num_expansions
        self.astar.num_expansions = num_expansions
        return
