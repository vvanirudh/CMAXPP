from szeth.envs.car_racing.car_racing_env_revised import make_car_racing_env
from szeth.envs.car_racing.car_racing_env_revised import X_DISCRETIZATION, Y_DISCRETIZATION
from szeth.utils.astar_mprim import Astar_mprim
from szeth.utils.node import Node
from szeth.controllers.controller import Controller
import numpy as np


class CarRacingController(Controller):
    def __init__(self,
                 model,
                 num_expansions=3):
        super(CarRacingController, self).__init__()
        self.model = model
        self.model.reset()
        self.cost_map = self.model.cost_map
        self.num_expansions = num_expansions
        # self.actions = self.model.get_actions()
        self.mprims = self.model.get_motion_primitives()

        self.astar = Astar_mprim(self.heuristic,
                                 self.get_successors,
                                 self.check_goal,
                                 num_expansions,
                                 self.get_mprims)

        self.heuristic_fn = None
        self.discrepancy_fn = None
        self.checkpoint = None

    def heuristic(self, node):
        if isinstance(node, Node):
            obs = node.obs
        else:
            obs = node

        steps_to_goal = 0

        if self.heuristic_fn is not None:
            steps_to_goal = self.heuristic_fn(obs)

        return steps_to_goal

    def get_successors(self, node, mprim):
        # obs = node.obs
        # self.model.set_sim_state(copy.deepcopy(obs['true_state']))

        # next_obs, reward, _, _ = self.model.step_mprim(mprim)

        # cost = -reward
        # if self.discrepancy_fn is not None:
        #     cost = self.discrepancy_fn(obs, mprim, cost)

        # next_node = Node(next_obs)

        obs = node.obs
        cost = 0
        # Step through all discrete states that the mprim goes through
        # and sum up the cost
        for discrete_state in mprim.discrete_states:
            xd, yd, thetad = discrete_state
            current_observation = np.array(
                [max(min(obs['observation'][0] + xd, X_DISCRETIZATION-1), 0),
                 max(min(obs['observation'][1] + yd, Y_DISCRETIZATION-1), 0),
                 thetad], dtype=int)
            cost_step = self.cost_map[current_observation[0],
                                      current_observation[1]]
            # if self.discrepancy_fn is not None and (not discrepancy_state):
            #     if self.discrepancy_fn(current_obs, mprim, 0) != 0:
            #         discrepancy_state = True
            cost += cost_step

        if self.discrepancy_fn is not None:
            cost = self.discrepancy_fn(obs, mprim, cost)
        # if discrepancy_state:
        #     cost = 1e6
        next_obs = {'observation': current_observation}
        next_node = Node(next_obs)

        return next_node, cost

    def get_cost(self, obs, mprim):
        cost = 0
        for discrete_state in mprim.discrete_states:
            xd, yd, thetad = discrete_state
            current_observation = np.array(
                [max(min(obs['observation'][0] + xd, X_DISCRETIZATION-1), 0),
                 max(min(obs['observation'][1] + yd, Y_DISCRETIZATION-1), 0),
                 thetad], dtype=int)
            cost_step = self.cost_map[current_observation[0],
                                      current_observation[1]]
            cost += cost_step

        return cost

    def get_successors_obs(self, obs, mprim):
        node = Node(obs)
        next_node, cost = self.get_successors(node, mprim)
        return next_node.obs, cost

    def check_goal(self, node):
        obs = node.obs
        return self.model.check_goal(obs['observation'], self.checkpoint)

    def get_mprims(self, node):
        obs = node.obs
        heading = obs['observation'][2]
        assert heading in self.mprims, "heading not found in mprims"
        return self.mprims[heading]

    def act(self, obs):
        start_node = Node(obs)
        best_mprim, info = self.astar.act(start_node)
        return best_mprim, info

    def reconfigure_heuristic(self, heuristic_fn):
        self.heuristic_fn = heuristic_fn
        return True

    def reconfigure_discrepancy(self, discrepancy_fn):
        self.discrepancy_fn = discrepancy_fn
        return True

    def set_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint


def get_car_racing_controller(seed, n_expansions, friction_params=None):
    return CarRacingController(
        make_car_racing_env(seed=seed,
                            friction_params=friction_params),
        num_expansions=n_expansions)
