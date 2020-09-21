import ray
import logging
import argparse
import numpy as np
import pickle
import os
import os.path as osp

from szeth.experiment.train_pr2_3d import launch

ray.init(logging_level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--grid_size', type=int, default=30)
parser.add_argument('--true_mass', type=float, default=2)
parser.add_argument('--num_expansions', type=int, default=30)
parser.add_argument('--precomputed_heuristic', action='store_true')
parser.add_argument('--scenario', type=int,
                    default=4, choices=[1, 2, 3, 4])
parser.add_argument(
    '--agent', choices=['cmax', 'cmaxpp', 'adaptive_cmaxpp'], default='cmax')
parser.add_argument('--goal_threshold', type=int, default=1)
parser.add_argument('--gui', action='store_true')
parser.add_argument('--max_attempts', type=int, default=20)
parser.add_argument('--num_updates', type=int, default=0)
parser.add_argument('--max_timesteps', type=int, default=1000)
parser.add_argument('--alpha', type=int, default=4)

args = parser.parse_args()

rng = np.random.RandomState(0)
seeds = rng.randint(10000, size=5)

if args.agent == 'adaptive_cmaxpp':
    alphas = [4]
else:
    alphas = [1]

results = {}
for alpha in alphas:
    results[alpha] = {}
    args.alpha = alpha
    ray_results = []
    for seed in seeds:
        args.seed = seed
        ray_results.append(launch.remote(args))

    ray_results = ray.get(ray_results)
    seed_idx = 0
    for seed in seeds:
        results[alpha][seed_idx] = ray_results[seed_idx]
        seed_idx += 1

path = osp.join(os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/data')
if args.agent == 'adaptive_cmaxpp':
    path = osp.join(path, 'adaptive_cmaxpp_3d_results.pkl')
elif args.agent == 'cmaxpp':
    path = osp.join(path, 'cmaxpp_3d_results.pkl')
elif args.agent == 'cmax':
    path = osp.join(path, 'cmax_3d_results.pkl')

pickle.dump(results, open(path, 'wb'))

ray.shutdown()
