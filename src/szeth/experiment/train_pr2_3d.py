import argparse
import ray
import logging

from szeth.envs.pr2.pr2_3d_env import pr2_3d_env

from szeth.controllers.pr2.pr2_3d_controller import pr2_3d_controller
from szeth.controllers.pr2.pr2_3d_q_controller import pr2_3d_q_controller

from szeth.agents.pr2.pr2_3d.pr2_3d_cmax_agent import pr2_3d_cmax_agent
from szeth.agents.pr2.pr2_3d.pr2_3d_cmaxpp_agent import pr2_3d_cmaxpp_agent
from szeth.agents.pr2.pr2_3d.pr2_3d_adaptive_cmaxpp_agent import pr2_3d_adaptive_cmaxpp_agent

import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    ray.shutdown()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


@ray.remote
def launch(args):
    env = pr2_3d_env(args, mass=args.true_mass,
                     use_gui=args.gui, no_dynamics=False)
    planning_env = pr2_3d_env(
        args, mass=0.01, use_gui=False, no_kinematics=True)

    if args.agent == 'cmax':
        controller = pr2_3d_controller(planning_env,
                                       num_expansions=args.num_expansions)
        pr2_trainer = pr2_3d_cmax_agent(args, env, controller)
    elif args.agent == 'cmaxpp':
        controller = pr2_3d_q_controller(
            planning_env, num_expansions=args.num_expansions)
        pr2_trainer = pr2_3d_cmaxpp_agent(args, env, controller)
    elif args.agent == 'adaptive_cmaxpp':
        controller = pr2_3d_q_controller(
            planning_env, num_expansions=args.num_expansions)
        controller_inflated = pr2_3d_controller(
            planning_env, num_expansions=args.num_expansions)
        pr2_trainer = pr2_3d_adaptive_cmaxpp_agent(
            args, env, controller, controller_inflated)

    n_steps = pr2_trainer.learn_online_in_real_world()
    print('Reached goal in', n_steps, 'steps')

    env.close()
    planning_env.close()
    return n_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--grid_size', type=int, default=10)
    parser.add_argument('--true_mass', type=float, default=1)
    parser.add_argument('--num_expansions', type=int, default=5)
    parser.add_argument('--precomputed_heuristic', action='store_true')
    parser.add_argument('--scenario', type=int,
                        default=1, choices=[1, 2, 3, 4])
    parser.add_argument(
        '--agent', choices=['cmax', 'cmaxpp', 'adaptive_cmaxpp'], default='cmax')
    parser.add_argument('--goal_threshold', type=int, default=0)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--max_attempts', type=int, default=5)
    parser.add_argument('--num_updates', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=500)
    parser.add_argument('--alpha', type=int, default=4)

    args = parser.parse_args()
    ray.init(logging_level=logging.ERROR)
    ray.get(launch.remote(args))
    ray.shutdown()
