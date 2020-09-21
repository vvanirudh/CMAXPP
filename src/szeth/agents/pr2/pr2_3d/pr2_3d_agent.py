import time
import copy
import numpy as np
from szeth.utils.node import Node
from szeth.utils.dijkstra import Dijkstra
from collections import deque


class pr2_3d_agent:
    def __init__(self, args, env, controller, controller_inflated=None):
        self.args, self.env = args, env
        self.controller = controller
        self.controller_inflated = controller_inflated

        self.start_cell = self.env.start_cell
        self.goal_cell = self.env.goal_cell
        self.actions = self.controller.actions

        self.cell_values = np.zeros((args.grid_size,
                                     args.grid_size,
                                     args.grid_size), dtype=np.float32)
        self.qvalues = np.zeros(
            (len(self.actions), args.grid_size, args.grid_size, args.grid_size),
            dtype=np.float32)
        if self.args.precomputed_heuristic:
            self._fill_cell_values_dijkstra(controller.model)
            self._initialize_qvalues()

        self.discrepancy_sets = {}
        # Configure heuristic for controller
        def get_state_value(cell): return self.get_state_value(
            cell, inflated=False)
        self.controller.reconfigure_heuristic(get_state_value)
        # Configure discrepancy
        self.controller.reconfigure_discrepancy(self.get_discrepancy)

        self.max_buffer_size = 1000
        self.transition_buffer = deque([], maxlen=self.max_buffer_size)
        self.state_buffer = deque([], maxlen=self.max_buffer_size)

        # Inflated value functions
        self.cell_values_inflated = self.cell_values.copy()

        # Configure heuristic and discrepancy for controller_inflated
        if self.controller_inflated is not None:
            def get_state_value_inflated(
                cell): return self.get_state_value(cell, inflated=True)
            self.controller_inflated.reconfigure_heuristic(
                get_state_value_inflated)
            self.controller_inflated.reconfigure_discrepancy(
                self.get_discrepancy)

        # Anytime parameter
        self.alpha = self.args.alpha

    def get_state_value(self, cell, inflated=False):
        cell_values = self.cell_values_inflated if inflated else self.cell_values
        if self.args.precomputed_heuristic:
            return cell_values[tuple(cell)]
        else:
            return cell_values[tuple(cell)] + self.compute_heuristic(cell)

    def get_qvalues(self, cell):
        qvalues = []
        for ac in self.actions:
            qvalues.append(self.get_qvalue(cell, ac))
        return qvalues

    def get_qvalue(self, cell, ac):
        ac_idx = self.actions.index(ac)
        if self.args.precomputed_heuristic:
            return self.qvalues[ac_idx, cell[0], cell[1], cell[2]]
        else:
            return self.qvalues[ac_idx, cell[0], cell[1], cell[2]] + self.compute_heuristic(cell)

    def set_qvalue(self, cell, ac, new_qvalue):
        ac_idx = self.actions.index(ac)
        if self.args.precomputed_heuristic:
            self.qvalues[ac_idx, cell[0], cell[1], cell[2]] = new_qvalue
        else:
            self.qvalues[ac_idx, cell[0], cell[1], cell[2]
                         ] = new_qvalue - self.compute_heuristic(cell)
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

    def _fill_cell_values_dijkstra(self, env):
        '''
        Do Dijkstra and get good heuristic
        '''
        goal_cell = env.goal_cell
        goal_observation = {'observation': goal_cell, 'sim_state': None}
        goal_node = Node(goal_observation)
        # TODO: Getting initial values by not using kinematics
        # Using kinematics is slow, since we have to do IK
        dijkstra_search = Dijkstra(self.controller.get_successors_no_kinematics,
                                   self.controller.actions)

        closed_set = dijkstra_search.get_dijkstra_heuristic(goal_node)

        for node in closed_set:
            cell = node.obs['observation']
            self.cell_values[tuple(cell)] = node._g

        return

    def _initialize_qvalues(self):
        for i in range(len(self.actions)):
            self.qvalues[i] = self.cell_values.copy()
        return

    def compute_heuristic(self, cell):
        '''
        Compute manhattan heuristic
        '''
        heuristic = np.sum(np.abs(cell - self.goal_cell))
        if heuristic <= self.args.goal_threshold:
            heuristic = 0
        else:
            heuristic -= self.args.goal_threshold
        return heuristic

    def update_state_value(self, info, inflated=False):
        for node in info['closed']:
            cell = node.obs['observation']
            gval = node._g

            if self.args.precomputed_heuristic:
                if not inflated:
                    self.cell_values[tuple(cell)] = info['best_node_f'] - gval
                else:
                    self.cell_values_inflated[tuple(
                        cell)] = info['best_node_f'] - gval
            else:
                if not inflated:
                    self.cell_values[tuple(cell)] = info['best_node_f'] - \
                        gval - self.compute_heuristic(cell)
                else:
                    self.cell_values_inflated[tuple(cell)] = info['best_node_f'] - \
                        gval - self.compute_heuristic(cell)

        return

    def update_qvalue(self, obs, ac, cost, obs_next):
        cell = obs['observation'].copy()
        new_qvalue = cost + \
            self.get_state_value(obs_next['observation'], inflated=False)
        # HACK: Updating qvalue only if it increases; this is needed since
        # the dynamics can be stochastic as a result of using IK for 3D control
        if new_qvalue > self.get_qvalue(cell, ac):
            self.set_qvalue(cell, ac, new_qvalue)

        # backup trajectory so far
        self.update_buffer_qvalues()
        return

    def update_buffer_qvalues(self):
        for (o, a, c, on) in reversed(self.transition_buffer):
            cll = o['observation'].copy()
            new_qvalue = c + \
                self.get_state_value(on['observation'], inflated=False)
            # HACK: Updating qvalue only if it increases; this is needed since
            # the dynamics can be stochastic as a result of using IK for 3D control
            if new_qvalue > self.get_qvalue(cll, a):
                self.set_qvalue(cll, a, new_qvalue)
        return

    def add_to_transition_buffer(self, obs, ac, cost, obs_next):
        self.transition_buffer.append((obs, ac, cost, obs_next))
        return

    def add_to_state_buffer(self, obs):
        self.state_buffer.append(obs)
        return

    def increment_visit_counter(self, obs):
        cell = obs['observation']
        self.visit_counts[tuple(cell)] += 1
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
