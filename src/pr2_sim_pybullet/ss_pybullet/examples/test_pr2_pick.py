#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import pybullet as p
from pybullet_tools.utils import (connect, disable_real_time,
                                  set_default_camera, wait_for_user,
                                  disconnect, load_pybullet, TABLE_URDF, add_data_path,
                                  create_box, FLOOR_URDF, set_point, load_model,
                                  HideOutput, set_base_values, get_unit_vector, WorldSaver,
                                  get_pose, get_joint_positions)
from pybullet_tools.pr2_utils import (DRAKE_PR2_URDF, set_arm_conf, REST_LEFT_ARM, open_arm,
                                      close_arm, get_carry_conf, arm_conf, get_side_grasps,
                                      get_group_joints, get_other_arm)
from pybullet_tools.pr2_primitives import get_grasp_gen, get_ik_fn, get_ik_ir_gen, Pose, control_commands, Conf
from pybullet_tools.pr2_problems import create_pr2, create_floor, TABLE_MAX_Z, Problem


def pick_problem(arm='left', grasp_type='side'):
    other_arm = get_other_arm(arm)
    initial_conf = get_carry_conf(arm, grasp_type)

    pr2 = create_pr2()
    set_base_values(pr2, (0, -1.2, np.pi/2))
    set_arm_conf(pr2, arm, initial_conf)
    open_arm(pr2, arm)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)

    plane = create_floor()
    table = load_pybullet(TABLE_URDF)
    # table = create_table(height=0.8)
    # table = load_pybullet("table_square/table_square.urdf")
    box = create_box(.07, .05, .25)
    set_point(box, (-0.25, -0.3, TABLE_MAX_Z + .25/2))
    # set_point(box, (0.2, -0.2, 0.8 + .25/2 + 0.1))

    return Problem(robot=pr2, movable=[box], arms=[arm], grasp_types=[grasp_type], surfaces=[table],
                   goal_conf=get_pose(pr2), goal_holding=[(arm, box)])


def main():
    connect(use_gui=True)
    disable_real_time()
    set_default_camera()

    problem = pick_problem(arm='left', grasp_type='side')
    grasp_gen = get_grasp_gen(problem, collisions=False)
    ik_ir_fn = get_ik_ir_gen(problem, max_attempts=100, teleport=False)
    pose = Pose(problem.movable[0], support=problem.surfaces[0])
    base_conf = Conf(problem.robot, get_group_joints(problem.robot, 'base'))
    ik_fn = get_ik_fn(problem)
    found = False
    saved_world = WorldSaver()
    for grasp, in grasp_gen(problem.movable[0]):
        print(grasp.value)
        # confs_cmds = ik_ir_fn(problem.arms[0], problem.movable[0], pose, grasp)
        # for conf, cmds in confs_cmds:
        #     found = True
        cmds = ik_fn(problem.arms[0],
                     problem.movable[0], pose, grasp, base_conf)
        if cmds is not None:
            cmds = cmds[0]
            found = True
        if found:
            break
    if not found:
        raise Exception('No grasp found')
    saved_world.restore()
    for cmd in cmds.commands:
        print(cmd)
    control_commands(cmds.commands)
    print('Quit?')

    wait_for_user()
    disconnect()


if __name__ == '__main__':
    main()
