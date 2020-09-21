import numpy as np
import random
# import ray
# import rospy

from szeth.envs.gridworld_env import make_gridworld_env
from szeth.controllers.gridworld_controller import get_gridworld_controller
from szeth.agents.gridworld_rts_agent import gridworld_rts_agent
from szeth.controllers.gridworld_qrts_controller import get_gridworld_qrts_controller
from szeth.controllers.gridworld_qlearning_controller import get_gridworld_qlearning_controller
from szeth.agents.gridworld_qrts_agent import gridworld_qrts_agent
from szeth.agents.gridworld_qlearning_agent import gridworld_qlearning_agent
from szeth.agents.arguments import get_args

import signal


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


# @ray.remote
def launch(args):
    # rospy.init_node('rts_trainer', anonymous=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = make_gridworld_env(args.env, args.grid_size,
                             args.render, args.incorrectness,
                             args.hard)
    planning_env = make_gridworld_env(
        args.planning_env, args.grid_size, False,
        args.incorrectness, args.hard)

    if args.agent == 'rts':
        controller = get_gridworld_controller(
            args.planning_env, args.grid_size, args.n_expansions)
        gridworld_trainer = gridworld_rts_agent(
            args, env, planning_env, controller)
    elif args.agent == 'qrts':
        controller = get_gridworld_qrts_controller(
            args.planning_env, args.grid_size, args.n_expansions)
        gridworld_trainer = gridworld_qrts_agent(
            args, env, planning_env, controller)
    elif args.agent == 'qlearning':
        controller = get_gridworld_qlearning_controller(args.grid_size)
        gridworld_trainer = gridworld_qlearning_agent(args, env, controller)
    else:
        raise NotImplementedError

    n_steps = gridworld_trainer.learn_online_in_real_world(args.max_timesteps)
    print('REACHED GOAL in', n_steps, 'by agent', args.agent)
    return n_steps


if __name__ == '__main__':
    args = get_args()
    # ray.init()
    # ray.get(launch.remote(args))
    launch(args)
