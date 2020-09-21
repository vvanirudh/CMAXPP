#!/usr/bin/env python

from __future__ import print_function
import pybullet as p
from pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    Pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time


def plan(robot, block, fixed, teleport):
    grasp_gen = get_grasp_gen(robot, 'top')
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport)
    free_motion_fn = get_free_motion_gen(
        robot, fixed=([block] + fixed), teleport=teleport)
    holding_motion_fn = get_holding_motion_gen(
        robot, fixed=fixed, teleport=teleport)

    pose0 = BodyPose(block)
    conf0 = BodyConf(robot)
    saved_world = WorldSaver()
    for grasp, in grasp_gen(block):
        saved_world.restore()
        result1 = ik_fn(block, pose0, grasp)
        if result1 is None:
            continue
        conf1, path2 = result1
        pose0.assign()
        result2 = free_motion_fn(conf0, conf1)
        if result2 is None:
            continue
        path1, = result2
        result3 = holding_motion_fn(conf1, conf0, block, grasp)
        if result3 is None:
            continue
        path3, = result3
        return Command(path1.body_paths +
                       path2.body_paths +
                       path3.body_paths)
    return None


def main(display='control'):  # control | execute | step
    connect(use_gui=True)
    disable_real_time()
    # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
    robot = load_model(DRAKE_IIWA_URDF, fixed_base=True)
    # floor = load_model('models/short_floor.urdf')
    floor = p.loadURDF("plane.urdf")
    block = load_model(
        "models/drake/objects/block_for_pick_and_place_heavy.urdf", fixed_base=False)
    set_pose(block, Pose(Point(y=0.5, z=stable_z(block, floor))))
    set_default_camera()
    dump_world()

    saved_world = WorldSaver()
    command = plan(robot, block, fixed=[floor], teleport=False)
    if (command is None) or (display is None):
        print('Unable to find a plan!')
        return

    saved_world.restore()
    update_state()
    user_input('{}?'.format(display))
    if display == 'control':
        enable_gravity()
        command.control(real_time=False, dt=0)
    elif display == 'execute':
        command.refine(num_steps=10).execute(time_step=0.005)
    elif display == 'step':
        command.step()
    else:
        raise ValueError(display)

    print('Quit?')
    wait_for_user()
    disconnect()


if __name__ == '__main__':
    main()
