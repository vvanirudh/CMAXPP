import ray
import logging
import argparse
import numpy as np
import torch

from szeth.envs.pr2.pr2_7d_xyzrpy_env import pr2_7d_xyzrpy_env

from szeth.controllers.pr2.pr2_7d_controller import pr2_7d_controller
from szeth.controllers.pr2.pr2_7d_q_controller import pr2_7d_q_controller
from szeth.controllers.pr2.pr2_7d_model_controller import pr2_7d_model_controller
from szeth.controllers.pr2.pr2_7d_qlearning_controller import pr2_7d_qlearning_controller

from szeth.agents.pr2.pr2_7d_approximate.pr2_7d_approximate_cmax_agent import pr2_7d_approximate_cmax_agent
from szeth.agents.pr2.pr2_7d_approximate.pr2_7d_approximate_cmaxpp_agent import pr2_7d_approximate_cmaxpp_agent
from szeth.agents.pr2.pr2_7d_approximate.pr2_7d_approximate_adaptive_cmaxpp_agent import pr2_7d_approximate_adaptive_cmaxpp_agent
from szeth.agents.pr2.pr2_7d_approximate.pr2_7d_approximate_model_agent import pr2_7d_approximate_model_agent
from szeth.agents.pr2.pr2_7d_approximate.pr2_7d_approximate_knn_agent import pr2_7d_approximate_knn_agent
from szeth.agents.pr2.pr2_7d_approximate.pr2_7d_approximate_qlearning_agent import pr2_7d_approximate_qlearning_agent

import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    ray.shutdown()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


@ray.remote
def launch(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    else:
        torch.manual_seed(0)
        np.random.seed(0)

    env = pr2_7d_xyzrpy_env(args, mass=args.true_mass,
                            use_gui=args.gui, no_dynamics=False)
    planning_env = pr2_7d_xyzrpy_env(
        args, mass=0.01, use_gui=False, no_dynamics=True)
    if args.agent == 'cmax':
        controller = pr2_7d_controller(planning_env,
                                       num_expansions=args.num_expansions)
        pr2_trainer = pr2_7d_approximate_cmax_agent(args, env, controller)
    elif args.agent == 'cmaxpp':
        controller = pr2_7d_q_controller(
            planning_env, num_expansions=args.num_expansions)
        pr2_trainer = pr2_7d_approximate_cmaxpp_agent(args, env, controller)
    elif args.agent == 'adaptive_cmaxpp':
        controller = pr2_7d_q_controller(
            planning_env, num_expansions=args.num_expansions)
        controller_inflated = pr2_7d_controller(
            planning_env, num_expansions=args.num_expansions)
        pr2_trainer = pr2_7d_approximate_adaptive_cmaxpp_agent(
            args, env, controller, controller_inflated)
    elif args.agent == 'model':
        controller = pr2_7d_model_controller(
            planning_env, num_expansions=args.num_expansions)
        pr2_trainer = pr2_7d_approximate_model_agent(args, env, controller)
    elif args.agent == 'knn':
        controller = pr2_7d_model_controller(
            planning_env, num_expansions=args.num_expansions)
        pr2_trainer = pr2_7d_approximate_knn_agent(args, env, controller)
    elif args.agent == 'qlearning':
        controller = pr2_7d_qlearning_controller(planning_env)
        pr2_trainer = pr2_7d_approximate_qlearning_agent(args, env, controller)
    n_steps = pr2_trainer.learn_online_in_real_world()
    print('Reached goal in', n_steps, 'steps')

    env.close()
    planning_env.close()

    return n_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--grid_size', type=int, default=10)
    parser.add_argument('--true_mass', type=float, default=4)
    parser.add_argument(
        '--agent', choices=['cmax', 'cmaxpp', 'adaptive_cmaxpp',
                            'model', 'knn', 'qlearning'], default='cmax')
    parser.add_argument('--num_expansions', type=int, default=10)
    parser.add_argument('--scenario', type=int,
                        default=4, choices=[1, 2, 3, 4])
    parser.add_argument('--goal_threshold', type=int, default=0)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--max_attempts', type=int, default=20)
    parser.add_argument('--num_updates', type=int, default=3)
    parser.add_argument('--num_updates_q', type=int, default=5)
    parser.add_argument('--max_timesteps', type=int, default=2000)

    parser.add_argument('--lr_value_residual', type=float, default=0.001)
    parser.add_argument('--l2_reg_value_residual', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_size_q', type=int, default=128)
    parser.add_argument('--polyak', type=float, default=0.5)

    parser.add_argument('--delta', type=int, default=1)

    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--rollout_length', type=int, default=10)

    parser.add_argument('--her_ratio', type=float, default=0.7)

    parser.add_argument('--alpha', type=float, default=1.0)

    # Model
    parser.add_argument('--lr_dynamics_residual', type=float, default=0.001)
    parser.add_argument('--l2_reg_dynamics_residual',
                        type=float, default=0.001)
    parser.add_argument('--batch_size_dyn', type=int, default=128)

    # KNN
    parser.add_argument('--knn_radius', type=float, default=3)

    # QLearning
    parser.add_argument('--qlearning_update_freq', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.1)

    args = parser.parse_args()
    ray.init(logging_level=logging.ERROR)
    ray.get(launch.remote(args))
    ray.shutdown()
