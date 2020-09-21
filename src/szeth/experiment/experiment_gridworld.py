import ray
import numpy as np
import pickle
import os

from szeth.agents.arguments import get_args
from szeth.experiment.train_gridworld import launch

import matplotlib.pyplot as plt

# Configure the experiment
np.random.seed(0)
incorrectness = [0, 0.2, 0.4, 0.6, 0.8]
seeds = np.random.randint(100000, size=5)
grid_size = 100
n_expansions = 5
epsilon = [0.1, 0.3, 0.5]

random_obstacle_results = {}

ray.init()

# Random obstacle experiments
args = get_args()
args.env = 'random_obstacle'
args.planning_env = 'empty'
args.grid_size = grid_size
args.max_timesteps = 100000

# RTS
args.agent = 'rts'
random_obstacle_results['rts'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    random_obstacle_results['rts'][incorrect] = result

# QRTS
args.agent = 'qrts'
random_obstacle_results['qrts'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    random_obstacle_results['qrts'][incorrect] = result

random_slip_results = {}

# Random slip state experiments
args = get_args()
args.env = 'random_slip'
args.planning_env = 'empty'
args.grid_size = grid_size
args.max_timesteps = 100000

# RTS
args.agent = 'rts'
random_slip_results['rts'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    random_slip_results['rts'][incorrect] = result

# QRTS
args.agent = 'qrts'
random_slip_results['qrts'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    random_slip_results['qrts'][incorrect] = result


slip_results = {}
# Slip experiments
args = get_args()
args.env = 'random_slip'
args.planning_env = 'empty'
args.grid_size = grid_size
args.max_timesteps = 100000

# RTS
args.agent = 'rts'
slip_results['rts'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    slip_results['rts'][incorrect] = result

# QRTS
args.agent = 'qrts'
slip_results['qrts'] = {}
for incorrect in incorrectness:
    args.incorrectness = incorrect
    result = []
    for seed in seeds:
        args.seed = seed
        n_steps = launch.remote(args)
        result.append(n_steps)
    result = ray.get(result)
    slip_results['qrts'][incorrect] = result


pickle.dump([random_obstacle_results, random_slip_results, slip_results], open(os.path.join(os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/save/gridworld_experiments_results.pkl'), 'wb'))
