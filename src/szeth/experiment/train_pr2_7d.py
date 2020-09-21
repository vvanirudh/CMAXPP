import numpy as np
import argparse

from szeth.envs.pr2.pr2_7d_xyzrpy_env import pr2_7d_xyzrpy_env

from szeth.controllers.pr2.pr2_7d_controller import pr2_7d_controller
from szeth.controllers.pr2.pr2_7d_q_controller import pr2_7d_q_controller

from szeth.agents.pr2.pr2_7d.pr2_7d_cmax_agent import pr2_7d_cmax_agent
from szeth.agents.pr2.pr2_7d.pr2_7d_cmaxpp_agent import pr2_7d_cmaxpp_agent
from szeth.agents.pr2.pr2_7d.pr2_7d_adaptive_cmaxpp_agent import pr2_7d_adaptive_cmaxpp_agent

import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


def launch(args):
    env = pr2_7d_xyzrpy_env(args, mass=args.true_mass,
                            use_gui=args.gui, no_dynamics=False)
    planning_env = pr2_7d_xyzrpy_env(
        args, mass=0.01, use_gui=False, no_dynamics=True)
    if args.agent == 'cmax':
        controller = pr2_7d_controller(planning_env,
                                       num_expansions=args.num_expansions)
        pr2_trainer = pr2_7d_cmax_agent(args, env, controller)
    elif args.agent == 'cmaxpp':
        controller = pr2_7d_q_controller(
            planning_env, num_expansions=args.num_expansions)
        pr2_trainer = pr2_7d_cmaxpp_agent(args, env, controller)
    elif args.agent == 'adaptive_cmaxpp':
        controller = pr2_7d_q_controller(
            planning_env, num_expansions=args.num_expansions)
        controller_inflated = pr2_7d_controller(planning_env,
                                                num_expansions=args.num_expansions)
        pr2_trainer = pr2_7d_adaptive_cmaxpp_agent(
            args, env, controller, controller_inflated)

    n_steps = pr2_trainer.learn_online_in_real_world()
    print('Reached goal in', n_steps, 'steps')

    env.close()
    planning_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--grid_size', type=int, default=10)
    parser.add_argument('--true_mass', type=float, default=1)
    parser.add_argument(
        '--agent', choices=['cmax', 'cmaxpp', 'adaptive_cmaxpp'], default='cmax')
    parser.add_argument('--num_expansions', type=int, default=5)
    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--goal_threshold', type=int, default=0)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--max_attempts', type=int, default=5)
    parser.add_argument('--num_updates', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=500)

    args = parser.parse_args()
    launch(args)
