from szeth.controllers.controller import Controller
from szeth.utils.qastar import QAstar
from szeth.utils.node import QNode
from szeth.utils.simulation_utils import set_gridworld_state_and_goal

from szeth.envs.gridworld_env import make_gridworld_env


class GridWorldQrtsController(Controller):
    def __init__(self, model, num_expansions=3):
        super(GridWorldQrtsController, self).__init__()
        self.model = model
        self.model.reset()
        self.num_expansions = num_expansions
        self.actions = self.model.get_actions()

        self.qastar = QAstar(self.heuristic,
                             self.qvalue,
                             self.discrepancy,
                             self.get_successors,
                             self.check_goal,
                             num_expansions,
                             self.actions)

        self.heuristic_fn = None
        self.discrepancy_fn = None
        self.qvalue_fn = None

    def heuristic(self, node):
        if isinstance(node, QNode):
            obs = node.obs
        else:
            obs = node

        if self.heuristic_fn:
            steps_to_goal = self.heuristic_fn(obs)
        else:
            raise Exception('Heuristic function not defined')

        return steps_to_goal

    def qvalue(self, node, ac):
        if isinstance(node, QNode):
            assert not node.dummy, "Querying qvalue of a dummy state"
            obs = node.obs
        else:
            obs = node

        if self.qvalue_fn:
            qvalue = self.qvalue_fn(obs, ac)
        else:
            raise Exception('Qvalue function not defined')

        return qvalue

    def discrepancy(self, obs, ac):
        return self.discrepancy_fn(obs, ac)

    def get_successors(self, node, action):
        obs = node.obs
        set_gridworld_state_and_goal(
            self.model,
            obs['observation'].copy(),
            obs['desired_goal'].copy(),
        )

        next_obs, cost, _, _ = self.model.step(action)

        next_node = QNode(obs=next_obs, dummy=False)

        return next_node, cost

    def check_goal(self, node):
        obs = node.obs
        current_state = obs['observation'].copy()
        goal_state = obs['desired_goal'].copy()
        return self.model.check_goal(current_state, goal_state)

    def act(self, obs):
        start_node = QNode(obs=obs, dummy=False)
        best_action, info = self.qastar.act(start_node)
        return best_action, info

    def reconfigure_heuristic(self, heuristic_fn):
        self.heuristic_fn = heuristic_fn
        return True

    def reconfigure_discrepancy(self, discrepancy_fn):
        self.discrepancy_fn = discrepancy_fn
        return True

    def reconfigure_qvalue_fn(self, qvalue_fn):
        self.qvalue_fn = qvalue_fn
        return True


def get_gridworld_qrts_controller(env, grid_size, n_expansions):
    return GridWorldQrtsController(make_gridworld_env(env, grid_size), n_expansions)
