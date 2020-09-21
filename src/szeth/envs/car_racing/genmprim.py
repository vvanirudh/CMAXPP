import ray
import math
from itertools import combinations_with_replacement
import numpy as np
from szeth.envs.car_racing_env_revised import CarRacing
from szeth.envs.car_racing_env_revised import (X_DISCRETIZATION,
                                               Y_DISCRETIZATION,
                                               THETA_DISCRETIZATION,
                                               X_BOUNDS,
                                               Y_BOUNDS)
from szeth.utils.mprim import write_primitives, MPrim


X_CELL_SIZE = (X_BOUNDS[1] - X_BOUNDS[0]) / X_DISCRETIZATION
Y_CELL_SIZE = (Y_BOUNDS[1] - Y_BOUNDS[0]) / Y_DISCRETIZATION
THETA_CELL_SIZE = (2 * math.pi) / THETA_DISCRETIZATION
# THRESHOLD = min(X_CELL_SIZE / 20, Y_CELL_SIZE / 20, THETA_CELL_SIZE / 20)
X_THRESHOLD = X_CELL_SIZE / 40
Y_THRESHOLD = Y_CELL_SIZE / 40
THETA_THRESHOLD = THETA_CELL_SIZE / 40
print(X_CELL_SIZE, Y_CELL_SIZE, THETA_CELL_SIZE,
      X_THRESHOLD, Y_THRESHOLD, THETA_THRESHOLD)


CONTROLS = [np.array([0, 0]), np.array([0, 1]), np.array([0, 2]),
            np.array([1, 0]), np.array([1, 1]), np.array([1, 2])]

COSTS = {tuple(CONTROLS[0]): -2,
         tuple(CONTROLS[1]): -1,
         tuple(CONTROLS[2]): -2,
         tuple(CONTROLS[3]): -1,
         tuple(CONTROLS[4]): 0,
         tuple(CONTROLS[5]): -1}

Headings = np.arange(THETA_DISCRETIZATION, dtype=np.int32)
Ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
mprims_all_headings = []

ray.init()


@ray.remote
def generate_mprims_for_heading(heading):
    env = CarRacing(seed=0, verbose=0, read_mprim=False, no_boundary=True)
    obs = env.reset()
    origin = np.array(
        [X_DISCRETIZATION//2, Y_DISCRETIZATION//2, THETA_DISCRETIZATION//2])
    mprims = []
    for T in Ts:
        # Rollout with length T
        for seq in combinations_with_replacement(CONTROLS, T):
            # A control sequence of length T
            # Reset env
            origin[2] = heading
            obs = env.reset_car_position(origin)
            start_lattice_center = obs['observation'].copy()
            start_lattice_center[2] = 0
            start_state = obs['true_state'].copy()
            start_state[2] = 0
            true_states = []
            true_states.append(obs['true_state'] - start_state)
            discrete_states = []
            # Add the start discrete state
            discrete_states.append(obs['observation'] - start_lattice_center)
            for t in range(T):
                obs, _, _, _ = env.step(seq[t])
                true_states.append(obs['true_state'] - start_state)
                discrete_states.append(
                    obs['observation'] - start_lattice_center)
                if np.array_equal(np.array([start_lattice_center[0],
                                            start_lattice_center[1]]),
                                  obs['observation'][:2]):
                    # Still in the start cell
                    continue
                discretization_error = obs['discretization_error']
                closeness_condition = (
                    discretization_error['x'] <= X_THRESHOLD)
                closeness_condition = closeness_condition and (
                    discretization_error['y'] <= Y_THRESHOLD)
                closeness_condition = closeness_condition and (
                    discretization_error['theta'] <= THETA_THRESHOLD)
                # Not the last timestep
                if t != T-1 and closeness_condition:
                    print('Intermediate point close to lattice center',
                          obs['observation'] - start_lattice_center)
                    print('Discarding motion primitive')
                    break
                # The last timestep
                if t == T-1 and closeness_condition:
                    print('Final point close to lattice center',
                          obs['observation'] - start_lattice_center)
                    print('Storing motion primitive')
                    mprim = MPrim(heading, T, seq,
                                  true_states,
                                  discrete_states,
                                  obs['observation'] - start_lattice_center)
                    mprims.append(mprim)
                    print(mprim)

    env.close()
    return mprims


mprims_all_headings = ray.get(
    [generate_mprims_for_heading.remote(heading) for heading in Headings])


# There can be more than one mprim with the same initial heading and final point
# We need to discard the duplicate mprims with larger T
# Among the mprims with the same T, same initial heading and same final point
# We need to discard mprims with higher cost
# At the end, we should have only one mprim for every (initial_heading, final_point) pair
mprims_all_headings_no_duplicates = []
for initial_heading, mprims in enumerate(mprims_all_headings):
    # Same initial heading
    final_points = {}
    # Get all final points
    for mprim in mprims:
        if tuple(mprim.final_point) not in final_points:
            final_points[tuple(mprim.final_point)] = [mprim]
        else:
            final_points[tuple(mprim.final_point)].append(mprim)

    # Remove mprims with larger T
    for final_point in final_points.keys():
        # Get all mprims corresponding to the same final point
        mprims_final_point = final_points[final_point]
        # Get the value of least T among the mprims
        least_T = np.inf
        for mprim in mprims_final_point:
            least_T = min(least_T, mprim.T)
        # Get all mprims with the least T
        least_mprims = []
        for mprim in mprims_final_point:
            if mprim.T == least_T:
                least_mprims.append(mprim)

        final_points[final_point] = least_mprims

    # Remove mprims with larger cost
    for final_point in final_points.keys():
        # Get all mprims corresponding to the same final point
        mprims_final_point = final_points[final_point]
        # Get the value of the least cost among the mprims
        least_cost = np.inf
        least_mprim = None
        for mprim in mprims_final_point:
            mprim_cost = mprim.compute_cost(COSTS)
            if mprim_cost <= least_cost:
                least_cost = mprim_cost
                least_mprim = mprim

        final_points[final_point] = [least_mprim]

    # Combine all mprims
    mprims_no_duplicates = []
    for final_point in final_points.keys():
        mprims_no_duplicates += final_points[final_point]

    mprims_all_headings_no_duplicates.append(mprims_no_duplicates)

heading = 0
number_of_primitives = 0
for mprims in mprims_all_headings_no_duplicates:
    print('-------------------------------')
    print('-------------------------------')
    print('HEADING', heading)

    for mprim in mprims:
        print('--------------------------------')
        print(mprim)
        number_of_primitives += 1

    heading += 1

# Store params used in generating motion primitives
params = {
    'X_BOUNDS': np.array(X_BOUNDS),
    'Y_BOUNDS': np.array(Y_BOUNDS),
    'X_DISCRETIZATION': X_DISCRETIZATION,
    'Y_DISCRETIZATION': Y_DISCRETIZATION,
    'THETA_DISCRETIZATION': THETA_DISCRETIZATION,
    'X_THRESHOLD': X_THRESHOLD,
    'Y_THRESHOLD': Y_THRESHOLD,
    'THETA_THRESHOLD': THETA_THRESHOLD,
    'number_of_primitives': number_of_primitives
}
# Write primitives into file
write_primitives(mprims_all_headings_no_duplicates, params)
