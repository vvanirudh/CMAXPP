import argparse
import os
import pickle
import numpy as np
import ray
import logging

from szeth.experiment.train_car_racing import launch, construct_friction_params

ray.init(logging_level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--ice_friction', type=float, default=0.1)
parser.add_argument('--variable_speed', action='store_true')
parser.add_argument('--n_expansions', type=int, default=100)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--no_cmax', action='store_true')
parser.add_argument('--max_timesteps', type=int, default=None)
parser.add_argument('--max_laps', type=int, default=None)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--load_value_fn', action='store_true')
parser.add_argument('--save_value_fn', action='store_true')
parser.add_argument(
    '--agent', choices=['cmax',
                        'cmaxpp',
                        'adaptive_cmaxpp',
                        'qlearning',
                        'model'], default='cmax')
parser.add_argument('--alpha', type=float, default=100)

# Scheduling alpha
parser.add_argument('--schedule', type=str,
                    choices=['exp', 'linear', 'time', 'step'],
                    default='step')
parser.add_argument('--exp_step', type=float, default=0.9)
parser.add_argument('--step_freq', type=float, default=5)

# Sensitivity expts
parser.add_argument('--alpha_expt', action='store_true')
args = parser.parse_args()

if args.ice_friction is not None and args.variable_speed:
    raise Exception('Both ice and variable speed are enabled')

np.random.seed(0)
seeds = np.random.randint(1000, size=10)

if args.agent != 'adaptive_cmaxpp':
    alphas = [1]
else:
    alphas = [args.alpha]

results = {}

for alpha in alphas:
    args.alpha = alpha
    n_steps = []
    lap_t_steps = []
    lap_returns = []
    successes = []

    ray_results = []
    for seed in seeds:
        args.seed = int(seed)
        args.friction_params = construct_friction_params(args)
        args.max_laps = 200
        args.headless = True
        args.load_value_fn = True
        args.max_timesteps = 1e4

        ray_results.append(launch.remote(args))

    ray_results = ray.get(ray_results)
    for r in ray_results:
        n_steps_seed, lap_t_steps_seed, lap_returns_seed = r
        if len(lap_t_steps_seed) != args.max_laps:
            success = False
        else:
            success = True
        n_steps.append(n_steps_seed)
        lap_t_steps.append(lap_t_steps_seed)
        lap_returns.append(lap_returns_seed)
        successes.append(success)

    results[alpha] = {
        'n_steps': n_steps,
        'lap_t_steps': lap_t_steps,
        'lap_returns': lap_returns,
        'successes': successes
    }

path = os.path.join(
    os.environ['HOME'],
    'workspaces/szeth_ws/src/szeth/data/')

if args.agent == 'cmax' and (not args.no_cmax):
    if args.ice_friction is not None:
        path += 'cmax_ice_friction_results.pkl'
    if args.variable_speed:
        path += 'cmax_variable_speed_results.pkl'
elif args.agent == 'qlearning':
    if args.ice_friction is not None:
        path += 'qlearning_ice_friction_results.pkl'
    if args.variable_speed:
        path += 'qlearning_variable_speed_results.pkl'
elif args.agent == 'model':
    if args.ice_friction is not None:
        path += 'model_ice_friction_results.pkl'
    if args.variable_speed:
        path += 'model_variable_speed_results.pkl'
elif args.agent == 'cmaxpp':
    if args.ice_friction is not None:
        path += 'cmaxpp_ice_friction_results.pkl'
    if args.variable_speed:
        path += 'cmaxpp_variable_speed_results.pkl'
elif args.agent == 'adaptive_cmaxpp' and (not args.alpha_expt):
    if args.ice_friction is not None:
        path += 'adaptive_cmaxpp_ice_friction_results.pkl'
    if args.variable_speed:
        path += 'adaptive_cmaxpp_variable_speed_results.pkl'
elif args.agent == 'adaptive_cmaxpp' and args.alpha_expt:
    path += 'adaptive_cmaxpp_ice_friction_results_'
    path += str(args.alpha)+'_'
    path += args.schedule+'_'
    if args.schedule == 'exp':
        path += str(args.exp_step)
    elif args.schedule == 'step':
        path += str(args.step_freq)
    path += '.pkl'
pickle.dump(results, open(path, 'wb'))

ray.shutdown()
