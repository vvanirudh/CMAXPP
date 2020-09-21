import argparse
import numpy as np
import pickle
import os
import os.path as osp
import signal
from szeth.envs.pr2.pr2_7d_xyzrpy_env import pr2_7d_xyzrpy_env


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--grid_size', type=int, default=20)
parser.add_argument('--true_mass', type=float, default=1)
parser.add_argument(
    '--agent', choices=['cmax', 'cmaxpp', 'adaptive_cmaxpp'], default='cmax')
parser.add_argument('--num_expansions', type=int, default=3)
parser.add_argument('--scenario', type=int, default=3, choices=[1, 2, 3])
parser.add_argument('--goal_threshold', type=int, default=0)
parser.add_argument('--gui', action='store_true')
parser.add_argument('--max_attempts', type=int, default=10)
parser.add_argument('--num_updates', type=int, default=10)
parser.add_argument('--max_timesteps', type=int, default=500)

parser.add_argument('--lr_value_residual', type=float, default=0.001)
parser.add_argument('--l2_reg_value_residual', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--polyak', type=float, default=0.5)

parser.add_argument('--delta', type=int, default=1)

parser.add_argument('--n_workers', type=int, default=16)
parser.add_argument('--rollout_length', type=int, default=10)

parser.add_argument('--her_ratio', type=float, default=0.5)

parser.add_argument('--alpha', type=float, default=2.0)

args = parser.parse_args()
env = pr2_7d_xyzrpy_env(args, mass=args.true_mass,
                        use_gui=args.gui, no_dynamics=False)


cell_to_joint_conf_cache = pickle.load(
    open(osp.join(os.environ['HOME'],
                  'workspaces/szeth_ws/src/szeth/data/cell_to_joint_conf.pkl'), 'rb'))

print('Number of keys', len(cell_to_joint_conf_cache.keys()))

count = 0
num_samples = 1e5
while count < num_samples:
    # Construct cell
    cell = np.random.randint(low=0, high=args.grid_size, size=7)
    if tuple(cell) in cell_to_joint_conf_cache:
        continue
    # Get conf
    conf = env._grid_to_continuous(cell)
    # Get joint conf
    joint_conf = env.sim.get_joint_conf_from_xyz_rpy(conf)
    # Add to cache
    cell_to_joint_conf_cache[tuple(cell)] = joint_conf
    # Increment count
    count += 1
    print(count)

pickle.dump(cell_to_joint_conf_cache, open(
    osp.join(os.environ['HOME'],
             'workspaces/szeth_ws/src/szeth/data/cell_to_joint_conf.pkl'),
    'wb'))
