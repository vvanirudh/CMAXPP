import numpy as np

counter = 0


class Node:
    def __init__(self, obs):
        self.obs = obs
        self._g = None
        self._h = None
        self._came_from = None
        self._action = None

    def __eq__(self, other):
        if isinstance(self.obs, dict):
            if np.array_equal(self.obs['observation'],
                              other.obs['observation']):
                return True
            return False
        else:
            if np.array_equal(self.obs, other.obs):
                return True
            return False

    def __hash__(self):
        if isinstance(self.obs, dict):
            return hash(tuple(self.obs['observation']))
        else:
            return hash(tuple(self.obs))


class QNode:
    def __init__(self, obs=None, dummy=False):
        self.obs = obs
        self.dummy = dummy
        global counter
        self.j = counter
        counter = counter + 1
        self._g = None
        self._h = None
        self._came_from = None
        self._action = None

    def __eq__(self, other):
        if self.dummy:
            # Dummy node can't be the same as any other node that is not dummy
            if not other.dummy:
                return False
            else:
                # Two dummy nodes can be the same iff they have
                # the same j value
                return self.j == other.j
        else:
            # This node is not a dummy
            # If the other node is dummy then they can't be same
            if other.dummy:
                return False
            else:
                # The other node is not dummy, so check obs
                if isinstance(self.obs, dict):
                    if np.array_equal(self.obs['observation'],
                                      other.obs['observation']):
                        return True
                    return False
                else:
                    if np.array_equal(self.obs, other.obs):
                        return True
                    return False

    def __hash__(self):
        # If this is a dummy node, then hash based on j value
        if self.dummy:
            return hash(self.j)
        else:
            # Not a dummy
            if isinstance(self.obs, dict):
                return hash(tuple(self.obs['observation']))
            else:
                return hash(tuple(self.obs))

    def __print__(self):
        if not self.dummy:
            print('state', self.obs['observation'], 'dummy', self.dummy)
        else:
            print('state', None, 'dummy', self.dummy)
