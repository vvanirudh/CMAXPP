import heapq
from szeth.utils.node import QNode

counter = 0


class QAstar:
    def __init__(self,
                 heuristic_fn,
                 qvalue_fn,
                 discrepancy_fn,
                 successors_fn,
                 check_goal_fn,
                 num_expansions,
                 actions):
        self.heuristic_fn = heuristic_fn
        self.qvalue_fn = qvalue_fn
        self.discrepancy_fn = discrepancy_fn
        self.successors_fn = successors_fn
        self.check_goal_fn = check_goal_fn
        self.num_expansions = num_expansions
        self.actions = actions

    def act(self, start_node):
        closed_set = set()
        inconsistent_set = set()
        open = []

        if hasattr(start_node, '_came_from'):
            # Ensure start node has no parent
            del start_node._came_from

        reached_goal = False
        popped_dummy = False

        start_node._g = 0
        h = start_node._h = self.heuristic_fn(start_node)
        f = start_node._g + start_node._h

        count = 0
        start_triplet = [f, h, count, start_node]
        heapq.heappush(open, start_triplet)
        count += 1
        open_d = {start_node: start_triplet}

        node = None
        for _ in range(self.num_expansions):
            if len(open) == 0:
                # if the open list is empty!
                best_node = node
                break
            f, h, _, node = heapq.heappop(open)
            del open_d[node]

            # Check if the node is dummy, if so pop it and stop
            if node.dummy:
                popped_dummy = True
                best_node = node
                break

            closed_set.add(node)

            # CHeck if node is goal
            if self.check_goal_fn(node):
                reached_goal = True
                best_node = node
                break

            for action in self.actions:
                # Check if this transition has any known discrepancy
                if self.discrepancy_fn(node.obs, action):
                    # This transition is known to be incorrect
                    # Create a dummy node and add to open list
                    new_node = QNode(obs=None, dummy=True)
                    new_node._g = node._g
                    new_node._h = self.qvalue_fn(node.obs, action)
                    new_node._came_from = node
                    new_node._action = action
                    new_f = new_node._g + new_node._h

                    d = open_d[new_node] = [
                        new_f, new_node._h, count, new_node]
                    heapq.heappush(open, d)
                    count += 1
                    # Add this node to inconsistent set
                    inconsistent_set.add(new_node)
                else:
                    # No known discrepancy for this transition
                    neighbor, cost = self.successors_fn(node, action)
                    if neighbor in closed_set:
                        continue

                    tentative_g = node._g + cost
                    if neighbor not in open_d:
                        neighbor._came_from = node
                        neighbor._action = action
                        neighbor._g = tentative_g
                        h = neighbor._h = self.heuristic_fn(
                            neighbor)
                        f = neighbor._g + neighbor._h
                        d = open_d[neighbor] = [tentative_g +
                                                h, h, count, neighbor]
                        heapq.heappush(open, d)
                        count += 1
                    else:
                        neighbor = open_d[neighbor][3]
                        if tentative_g < neighbor._g:
                            neighbor._came_from = node
                            neighbor._action = action
                            neighbor._g = tentative_g
                            open_d[neighbor][0] = tentative_g + neighbor._h
                            heapq.heapify(open)

        # Check if we either reached the goal or popped a dummy
        if (not reached_goal) and (not popped_dummy) and (len(open) > 0):
            # Pop the open list again
            best_node_f, best_node_h, _, best_node = heapq.heappop(open)
            del open_d[best_node]

        info = {'best_node_f': best_node._g + best_node._h,
                'start_node_h': start_node._h,
                'best_node': best_node,
                'closed': closed_set,
                'open': open_d.keys,
                'start_node': start_node,
                'reached_goal': reached_goal}

        best_action, path = self.get_best_action(start_node, best_node)
        info['path'] = path
        info['successor_obs'] = path[1]
        return best_action, info

    def get_best_action(self, start_node, best_node):
        if start_node == best_node:
            # raise Exception('Start node is the best node!')
            # Already at the goal, should be at the goal
            # raise Exception('Should not happen')
            return None, [start_node.obs, start_node.obs]
        node = best_node._came_from
        action = best_node._action
        path = [best_node.obs]
        while True:
            if hasattr(node, '_came_from'):
                path.append(node.obs)
                # Not the start node
                next_node = node._came_from
                action = node._action
                if not hasattr(next_node, '_came_from'):
                    # NExt node is start node
                    break
                else:
                    node = next_node
            else:
                # Start node
                break

        path.append(start_node.obs)
        path = path[::-1]
        return action, path
