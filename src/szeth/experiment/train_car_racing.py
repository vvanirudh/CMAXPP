import numpy as np
import random
import argparse
import ray
import logging

from szeth.envs.car_racing.car_racing_env_revised import make_car_racing_env

from szeth.controllers.car_racing.car_controller import get_car_racing_controller
from szeth.controllers.car_racing.car_q_controller import get_car_racing_q_controller
from szeth.controllers.car_racing.car_model_controller import get_car_racing_model_controller

from szeth.agents.car_racing.car_racing_cmax_agent import car_racing_cmax_agent
from szeth.agents.car_racing.car_racing_cmaxpp_agent import car_racing_cmaxpp_agent
from szeth.agents.car_racing.car_racing_adaptive_cmaxpp_agent import car_racing_adaptive_cmaxpp_agent
from szeth.agents.car_racing.car_racing_qlearning_agent import car_racing_qlearning_agent
from szeth.agents.car_racing.car_racing_model_agent import car_racing_model_agent
import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(
        signal))
    ray.shutdown()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


def construct_friction_params(args):
    if args.ice_friction is None:
        return None
    else:
        friction_params = {}
        friction_params['ice'] = args.ice_friction

        return friction_params


@ray.remote
def launch(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = make_car_racing_env(
        seed=args.seed,
        friction_params=args.friction_params,
        variable_speed=args.variable_speed)
    if args.agent in ['adaptive_cmaxpp', 'cmaxpp']:
        controller = get_car_racing_q_controller(
            seed=args.seed,
            n_expansions=args.n_expansions,
            friction_params=None)
    elif args.agent == 'model':
        controller = get_car_racing_model_controller(
            seed=args.seed,
            n_expansions=args.n_expansions,
            friction_params=None
        )
    else:
        controller = get_car_racing_controller(
            seed=args.seed,
            n_expansions=args.n_expansions,
            friction_params=None)
    if args.agent == 'cmax':
        trainer = car_racing_cmax_agent(args, env, controller)
    elif args.agent == 'cmaxpp':
        trainer = car_racing_cmaxpp_agent(args, env, controller)
    elif args.agent == 'adaptive_cmaxpp':
        controller_inflated = get_car_racing_controller(
            seed=args.seed,
            n_expansions=args.n_expansions,
            friction_params=None
        )
        trainer = car_racing_adaptive_cmaxpp_agent(args, env, controller,
                                                   controller_inflated)
    elif args.agent == 'qlearning':
        trainer = car_racing_qlearning_agent(args, env, controller)
    elif args.agent == 'model':
        trainer = car_racing_model_agent(args, env, controller)

    n_steps, lap_t_steps, lap_returns = trainer.learn_online_in_real_world(
        max_timesteps=args.max_timesteps,
        max_laps=args.max_laps,
        render=(not args.headless),
        save_value_fn=args.save_value_fn)
    print('REACHED GOAL IN', lap_t_steps)
    return n_steps, lap_t_steps, lap_returns


if __name__ == '__main__':
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
    parser.add_argument('--alpha', type=float, default=10.0)

    # Scheduling alpha
    parser.add_argument('--schedule', type=str,
                        choices=['exp', 'linear', 'time', 'step'])
    parser.add_argument('--exp_step', type=float, default=0.9)
    parser.add_argument('--step_freq', type=float, default=5)

    args = parser.parse_args()

    args.friction_params = construct_friction_params(args)
    ray.init(logging_level=logging.ERROR)
    ray.get(launch.remote(args))
    ray.shutdown()
