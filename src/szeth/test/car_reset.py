import numpy as np
import time
from szeth.envs.car_racing_env_revised import CarRacing
from szeth.envs.car_racing_env_revised import (X_DISCRETIZATION,
                                               Y_DISCRETIZATION,
                                               THETA_DISCRETIZATION,
                                               X_BOUNDS,
                                               Y_BOUNDS)

origin = np.array(
    [X_DISCRETIZATION//2, Y_DISCRETIZATION//2, THETA_DISCRETIZATION//2])

env = CarRacing(seed=0, verbose=0, read_mprim=False, no_boundary=True)
obs = env.reset()
control = np.array([1, 1])

for i in range(10):
    obs = env.reset_car_position(origin)
    print(obs['observation'], obs['true_state'])
    obs, _, _, _ = env.step(control)
    print(obs['observation'], obs['true_state'])
    print('------')

env.close()
