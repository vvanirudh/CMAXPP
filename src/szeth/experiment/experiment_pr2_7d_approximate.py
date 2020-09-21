import ipdb
import argparse
import os
import os.path as osp
import pickle
import numpy as np
import ray
import logging

from szeth.experiment.train_pr2_7d_approximate import launch

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--grid_size', type=int, default=10)
parser.add_argument('--true_mass', type=float, default=4)
parser.add_argument(
    '--agent', choices=['cmax', 'cmaxpp', 'adaptive_cmaxpp',
                        'model', 'knn', 'qlearning'], default='cmax')
parser.add_argument('--num_expansions', type=int, default=5)
parser.add_argument('--scenario', type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument('--goal_threshold', type=int, default=0)
parser.add_argument('--gui', action='store_true')
parser.add_argument('--max_attempts', type=int, default=20)
parser.add_argument('--num_updates', type=int, default=3)
parser.add_argument('--num_updates_q', type=int, default=5)
parser.add_argument('--max_timesteps', type=int, default=500)

parser.add_argument('--lr_value_residual', type=float, default=0.001)
parser.add_argument('--l2_reg_value_residual', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_size_q', type=int, default=128)
parser.add_argument('--polyak', type=float, default=0.5)

parser.add_argument('--delta', type=int, default=3)

parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--rollout_length', type=int, default=10)

parser.add_argument('--her_ratio', type=float, default=0.7)

parser.add_argument('--alpha', type=float, default=4)

# Model
parser.add_argument('--lr_dynamics_residual', type=float, default=0.001)
parser.add_argument('--l2_reg_dynamics_residual',
                    type=float, default=0.001)
parser.add_argument('--batch_size_dyn', type=int, default=128)

# KNN
parser.add_argument('--knn_radius', type=float, default=3)

# Qlearning
parser.add_argument('--qlearning_update_freq', type=int, default=1)
parser.add_argument('--epsilon', type=float, default=0.1)

args = parser.parse_args()

ray.init(logging_level=logging.ERROR)

# Variables
# The following seeds were generated using np.random.randint(1000, size=4)
seeds = [None, 684, 835, 763, 9]

alphas = [args.alpha]
results = {}


# Model baseline
if args.agent in ['model', 'knn', 'qlearning']:
    ray_results = []
    for seed in seeds:
        args.seed = seed
        ray_results.append(launch.remote(args))

    ray_results = ray.get(ray_results)
    for seed_idx in range(len(seeds)):
        results[seed_idx] = ray_results[seed_idx]

    path = osp.join(os.environ['HOME'],
                    'workspaces/szeth_ws/src/szeth/data')
    if args.agent == 'model':
        model_path = osp.join(path, 'model_7d_approximate_results.pkl')
        pickle.dump(results, open(model_path, 'wb'))
    elif args.agent == 'knn':
        knn_path = osp.join(path, 'knn_7d_approximate_results.pkl')
        pickle.dump(results, open(knn_path, 'wb'))
    elif args.agent == 'qlearning':
        qlearning_path = osp.join(path, 'qlearning_7d_approximate_results.pkl')
        pickle.dump(results, open(qlearning_path, 'wb'))
else:
    # args.n_workers = 4
    for alpha in alphas:
        args.alpha = alpha
        results[alpha] = {}
        ray_results = []
        for seed in seeds:
            args.seed = seed
            agent = args.agent
            args.agent = 'adaptive_cmaxpp'
            if agent == 'cmax':
                args.alpha = 1e8
            elif agent == 'cmaxpp':
                args.alpha = -1e8
            ray_results.append(launch.remote(args))
            args.agent = agent

        ray_results = ray.get(ray_results)
        for seed_idx in range(len(seeds)):
            results[alpha][seed_idx] = ray_results[seed_idx]
        path = osp.join(os.environ['HOME'],
                        'workspaces/szeth_ws/src/szeth/data')
        if args.agent == 'cmax':
            cmax_path = osp.join(path, 'cmax_7d_approximate_results.pkl')
            pickle.dump(results, open(cmax_path, 'wb'))
        elif args.agent == 'cmaxpp':
            cmaxpp_path = osp.join(
                path, 'cmaxpp_7d_approximate_results.pkl')
            pickle.dump(results, open(cmaxpp_path, 'wb'))
        else:
            adaptive_cmaxpp_path = osp.join(
                path, 'adaptive_cmaxpp_7d_approximate_results.pkl')
            pickle.dump(results, open(adaptive_cmaxpp_path, 'wb'))

ray.shutdown()
