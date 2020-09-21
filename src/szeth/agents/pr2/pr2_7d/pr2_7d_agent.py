import numpy as np
from collections import deque


class pr2_7d_agent:
    def __init__(self, args, env, controller, controller_inflated=None):
        self.args, self.env = args, env
        self.controller = controller
        self.controller_inflated = controller_inflated

        self.start_cell = self.env.start_cell
        self.goal_cells = self.env.goal_cells
        self.actions = self.controller.actions

        self.values_dict = {}
        self.qvalues_dict = {}
        for ac in self.actions:
            self.qvalues_dict[ac] = {}

        self.discrepancy_sets = {}

        # Configure heuristic for controller
        def get_state_value(cell):
            return self.get_state_value(cell, inflated=False)
        self.controller.reconfigure_heuristic(get_state_value)
        # Configure discrepancy
        self.controller.reconfigure_discrepancy(self.get_discrepancy)

        self.max_buffer_size = 1000
        self.transition_buffer = deque([], maxlen=self.max_buffer_size)
        self.state_buffer = deque([], maxlen=self.max_buffer_size)

        # Inflated value functions
        self.values_dict_inflated = {}

        # Configure heuristic and discrepancy for controller_inflated
        if self.controller_inflated is not None:
            def get_state_value_inflated(cell):
                return self.get_state_value(cell, inflated=True)
            self.controller_inflated.reconfigure_heuristic(
                get_state_value_inflated)
            self.controller_inflated.reconfigure_discrepancy(
                self.get_discrepancy)

        # Anytime parameter
        self.alpha = 5

    def get_state_value(self, cell, inflated=False):
        value = self.compute_heuristic(cell)
        values_dict = self.values_dict_inflated if inflated else self.values_dict
        if tuple(cell) in values_dict:
            value += values_dict[tuple(cell)]
        return value

    def get_qvalues(self, cell):
        qvalues = []
        for ac in self.actions:
            qvalues.append(self.get_qvalue(cell, ac))
        return qvalues

    def get_qvalue(self, cell, ac):
        qvalue = self.compute_heuristic(cell)
        if tuple(cell) in self.qvalues_dict[ac]:
            qvalue += self.qvalues_dict[ac][tuple(cell)]
        return qvalue

    def set_qvalue(self, cell, ac, new_qvalue):
        self.qvalues_dict[ac][tuple(cell)] = new_qvalue - \
            self.compute_heuristic(cell)
        return

    def get_discrepancy(self, cell, ac, cost):
        if self.has_discrepancy(cell, ac):
            return 1e6
        return cost

    def has_discrepancy(self, cell, ac):
        if ac not in self.discrepancy_sets:
            return False
        if tuple(cell) not in self.discrepancy_sets[ac]:
            return False
        return True

    def compute_heuristic(self, cell):
        # Get the 6D gripper pose corresponding to the cell
        cell_6d = cell[:6]
        # Get the 6D gripper pose corresponding to the goal
        goal_cell_6d = self.goal_cells[0][:6]
        # Compute the manhattan heuristic
        heuristic = np.sum(np.abs(cell_6d - goal_cell_6d))
        # Account for any goal threshold
        if heuristic <= self.args.goal_threshold:
            heuristic = 0
        else:
            heuristic -= self.args.goal_threshold
        return heuristic

    def update_state_value(self, info, inflated=False):
        for node in info['closed']:
            cell = node.obs['observation']
            gval = node._g
            if not inflated:
                self.values_dict[tuple(cell)] = info['best_node_f'] - \
                    gval - self.compute_heuristic(cell)
            else:
                self.values_dict_inflated[tuple(cell)] = info['best_node_f'] - \
                    gval - self.compute_heuristic(cell)

        return

    def update_qvalue(self, obs, ac, cost, obs_next):
        cell = obs['observation'].copy()
        new_qvalue = cost + \
            self.get_state_value(obs_next['observation'], inflated=False)

        if new_qvalue >= self.get_qvalue(cell, ac):
            # raise Exception('Why did this happen?')
            self.set_qvalue(cell, ac, new_qvalue)

        # Backup trajectory so far
        self.update_buffer_qvalues()
        return

    def update_buffer_qvalues(self):
        for (o, a, c, on) in reversed(self.transition_buffer):
            cell = o['observation'].copy()
            new_qvalue = c + \
                self.get_state_value(on['observation'], inflated=False)
            if new_qvalue >= self.get_qvalue(cell, a):
                # raise Exception('Why did this happen?')
                self.set_qvalue(cell, a, new_qvalue)
        return

    def add_to_transition_buffer(self, obs, ac, cost, obs_next):
        self.transition_buffer.append((obs, ac, cost, obs_next))
        return

    def add_to_state_buffer(self, obs):
        self.state_buffer.append(obs)
        return

    def sample_state_buffer(self):
        state_buffer_size = len(self.state_buffer)
        idx = np.random.randint(state_buffer_size)
        return self.state_buffer[idx]

    def check_discrepancy(self, obs, ac, next_obs_sim, next_obs):
        if not np.array_equal(next_obs['observation'], next_obs_sim['observation']):
            self.add_discrepancy(obs, ac)
            return True
        return False

    def add_discrepancy(self, obs, ac):
        cell = obs['observation'].copy()
        if ac not in self.discrepancy_sets:
            self.discrepancy_sets[ac] = set()
        self.discrepancy_sets[ac].add(tuple(cell))
        return

    def learn_online_in_real_world(self):
        raise NotImplementedError
