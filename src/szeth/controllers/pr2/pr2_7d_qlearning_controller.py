import numpy as np
import copy


class pr2_7d_qlearning_controller:
    def __init__(self, model):
        self.model = model
        self.actions = self.model.get_repeated_actions()

        self.qvalues_fn = None

    def get_qvalues(self, obs):
        assert self.qvalues_fn is not None
        return self.qvalues_fn(obs)

    def reconfigure_qvalues_fn(self, qvalues_fn):
        self.qvalues_fn = qvalues_fn
        return

    def act(self, obs):
        # Get qvalues
        qvalues = self.get_qvalues(obs)
        # Choose the min qvalue
        ac_idx = np.argmin(qvalues)
        # Return the corresponding action
        return self.actions[ac_idx], None

    # dummy functions
    def reconfigure_heuristic(self, heuristic_fn):
        return

    def reconfigure_discrepancy(self, discrepancy_fn):
        return

    def reconfigure_num_expansions(self, num_expansions):
        return
