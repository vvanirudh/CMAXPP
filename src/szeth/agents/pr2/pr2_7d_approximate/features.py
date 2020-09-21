import numpy as np
from szeth.agents.pr2.pr2_7d_approximate.normalizer import normalizer


def stable_divide(a, b):
    if b < 1e-6:
        return a / 1e-6
    return a / b


def norm(a):
    return np.sqrt(np.sum(a**2))


def compute_features(cell, goal_cell, carry_cell, obstacle_cell_aa,
                     obstacle_cell_bb, grid_size, grid_to_continuous_fn):
    cell_representation = compute_representation(
        cell, grid_size, grid_to_continuous_fn)
    goal_cell_representation = compute_representation(
        goal_cell, grid_size, grid_to_continuous_fn)
    carry_cell_representation = compute_representation(
        carry_cell, grid_size, grid_to_continuous_fn)

    # relative position of goal w.r.t current state
    diff_position_goal = goal_cell_representation[:-
                                                  1] - cell_representation[:-1]

    # relative position of cell w.rt. carry cell
    diff_position_carry = carry_cell_representation - cell_representation

    # relative position of goal cell w.r.t carry cell
    diff_position_carry_goal = goal_cell_representation[:-1] - \
        carry_cell_representation[:-1]

    # relative position of cell w.r.t obstacle cell aa
    obstacle_cell_representation_aa = compute_representation_3d(
        obstacle_cell_aa, grid_size)
    diff_position_obstacle_aa = obstacle_cell_representation_aa - \
        cell_representation[:3]

    # relative position of cell w.r.t obstacle cell bb
    obstacle_cell_representation_bb = compute_representation_3d(
        obstacle_cell_bb, grid_size)
    diff_position_obstacle_bb = obstacle_cell_representation_bb - \
        cell_representation[:3]

    features = np.concatenate([diff_position_goal, # 9
                               diff_position_carry, # 10
                               diff_position_carry_goal, # 9
                               diff_position_obstacle_aa, # 3
                               diff_position_obstacle_bb]) # 3

    # features = np.concatenate(
    #     [cell_representation, goal_cell_representation[:-1]])

    return features


# def compute_features(cell, goal_cell, grid_size, grid_to_continuous_fn):
#     # TODO: Add obstacle specific features
#     cell_representation = compute_representation(
#         cell, grid_size, grid_to_continuous_fn)
#     goal_cell_representation = compute_representation(
#         goal_cell, grid_size, grid_to_continuous_fn)
#     # relative position of goal w.r.t current state
#     diff_position = goal_cell_representation[:6] - cell_representation[:6]
#     rel_position = diff_position / (np.linalg.norm(diff_position) + 1e-6)
#     distance = np.linalg.norm(diff_position)
#     features = np.concatenate([cell_representation, rel_position, [distance]])
#     return features


def compute_representation(cell, grid_size, grid_to_continuous_fn):
    # For x, y, z, redundant joint we simply normalize it to lie between [-1, 1]
    # For r, p, y we use the corresponding sin and cos values
    x, y, z, _, _, _, rjoint = cell
    _, _, _, rcont, pcont, ycont, _ = grid_to_continuous_fn(cell)
    xn = (x / grid_size) * 2 - 1
    yn = (y / grid_size) * 2 - 1
    zn = (z / grid_size) * 2 - 1
    rjointn = (rjoint / grid_size) * 2 - 1

    sinr, cosr = np.sin(rcont), np.cos(rcont)
    sinp, cosp = np.sin(pcont), np.cos(pcont)
    siny, cosy = np.sin(ycont), np.cos(ycont)

    return np.array([xn, yn, zn, sinr, cosr,
                     sinp, cosp, siny, cosy, rjointn],
                    dtype=np.float32)


def compute_representation_3d(cell_3d, grid_size):
    x, y, z = cell_3d
    xn = (x / grid_size) * 2 - 1
    yn = (y / grid_size) * 2 - 1
    zn = (z / grid_size) * 2 - 1

    return np.array([xn, yn, zn], dtype=np.float32)


def compute_heuristic(cell, goal_cell, goal_threshold):
    # Get the 6D gripper pose corresponding to the cell
    cell_6d = cell[:6]
    # Get the 6D gripper pose corresponding to the goal
    goal_cell_6d = goal_cell[:6]
    # Compute the manhattan heuristic
    heuristic = np.sum(np.abs(cell_6d - goal_cell_6d))
    # Account for any goal threshold
    if heuristic <= goal_threshold:
        heuristic = 0
    else:
        heuristic -= goal_threshold
    return heuristic


def compute_lookahead_heuristic(cell, goal_cell, ac, controller, goal_threshold):
    obs = {'observation': cell, 'desired_goal': goal_cell}
    # Set controller to obs
    controller.model.set_observation(obs, goal=True)
    # Step the model
    next_obs, cost = controller.model.step(ac)
    # Get next obs heuristic
    heuristic_next = compute_heuristic(next_obs['observation'],
                                       next_obs['desired_goal'],
                                       goal_threshold)

    return cost + heuristic_next


class Normalizer:
    def __init__(self, norm):
        '''
        Base class
        '''
        self.norm = norm

    def normalize(self, x):
        return self.norm.normalize(x)

    def update(self, x):
        return self.norm.update(x)

    def recompute_stats(self):
        return self.norm.recompute_stats()

    def get_mean(self):
        return self.norm.mean.copy()

    def get_std(self):
        return self.norm.std.copy()

    def state_dict(self):
        return {'mean': self.get_mean(),
                'std': self.get_std()}

    def load_state_dict(self, state_dict):
        self.norm.mean = state_dict['mean'].copy()
        self.norm.std = state_dict['std'].copy()


class FeatureNormalizer(Normalizer):
    def __init__(self, num_features):
        self.num_features = num_features
        f_norm = normalizer(size=num_features)
        Normalizer.__init__(self, f_norm)

    def update_normalizer(self, features):
        self.update(features)
        self.recompute_stats()
        return True
