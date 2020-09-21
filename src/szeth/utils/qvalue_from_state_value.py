'''
Script to generate Q* of the model from V* of the model
'''
import os
import ray
import pickle
import numpy as np
from szeth.envs.car_racing.car_racing_env_revised import (X_DISCRETIZATION,
                                                          Y_DISCRETIZATION,
                                                          THETA_DISCRETIZATION,
                                                          make_car_racing_env)


@ray.remote
def one_step_lookahead(x, y, theta, mprim, state_values, cost_map):
    # env = make_car_racing_env()
    # env.reset()
    # env.reset_car_position(np.array([x, y, theta]))
    # obs, reward, _, _ = env.step_mprim(mprim)

    state = np.array([x, y, theta], dtype=int)
    cost = 0
    for discrete_state in mprim.discrete_states:
        xd, yd, thetad = discrete_state
        state = np.array(
            [max(min(state[0] + xd, X_DISCRETIZATION-1), 0),
             max(min(state[1] + yd, Y_DISCRETIZATION-1), 0),
             thetad], dtype=int)
        cost += cost_map[state[0], state[1]]

    next_state_value = state_values[tuple(state)]
    return cost + next_state_value


def get_qvalue_from_state_value():
    path = os.path.join(os.environ['HOME'],
                        'workspaces/szeth_ws/src/szeth/save/value_fn.pkl')
    state_values = pickle.load(open(path, 'rb'))
    env = make_car_racing_env()
    env.reset()
    mprims = env.get_motion_primitives()
    cost_map = env.cost_map
    cost_map_id = ray.put(cost_map)
    qvalues = []
    for checkpoint_idx in range(len(state_values)):
        state_values_checkpoint = state_values[checkpoint_idx]
        state_values_checkpoint_id = ray.put(state_values_checkpoint)
        qvalues_checkpoint = {}
        for x in range(X_DISCRETIZATION):
            for y in range(Y_DISCRETIZATION):
                for theta in range(THETA_DISCRETIZATION):
                    print(x, y, theta)
                    mprims_theta = mprims[theta]
                    results = [one_step_lookahead.remote(
                        x, y, theta, mprim,
                        state_values_checkpoint_id, cost_map_id)
                        for mprim in mprims_theta]
                    qvalues_checkpoint[x, y, theta] = ray.get(results)

        qvalues.append(qvalues_checkpoint)

    # save qvalues
    path = os.path.join(os.environ['HOME'],
                        'workspaces/szeth_ws/src/szeth/save/qvalue_fn.pkl')
    pickle.dump(qvalues, open(path, 'wb'))
    return qvalues


if __name__ == '__main__':
    ray.init()
    get_qvalue_from_state_value()
