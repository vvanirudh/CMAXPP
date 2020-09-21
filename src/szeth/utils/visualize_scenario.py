import ipdb
import os
import os.path as osp
import numpy as np
from pr2_sim_pybullet.sim import PR2SimBullet
from pr2_sim_pybullet.ss_pybullet.pybullet_tools.utils import (
    wait_for_user, disconnect, BLUE, set_camera, Point,
    take_picture, save_image, BROWN, create_obj, BLACK,
    set_point, set_quat, quat_from_euler, get_pose,
    get_point
)

sim = PR2SimBullet(use_gui=True, scenario=4)
set_camera(180, -20, 1.5, Point(0.1, -0.4, 0.6))

# add object
min_object_position_x = 0.3
max_object_position_x = 0.45

object_position_x = np.random.uniform(
    min_object_position_x, max_object_position_x)
object_position = (object_position_x,
                   -0.4,
                   sim.table_max_z + 0.15/2 - 0.05/2)
# object_1 = sim.add_object(shape='box',
#                           dimensions=(0.05, 0.05, 0.15),
#                           position=object_position,
#                           orientation=(0, 0, 0),
#                           mass=0.001,
#                           color=BROWN)

object_1 = create_obj(osp.join(
    os.environ['HOME'],
    'workspaces/szeth_ws/src/szeth/src/pr2_sim_pybullet/ss_pybullet/models/dumbbell.obj'),
    mass=0.001,
    color=BROWN,
    scale=0.008)
object_position = (0.35,
                   -0.4,
                   sim.table_max_z + 0.3/2 - 0.05/2)
set_point(object_1, object_position)
set_quat(object_1, quat_from_euler((0, np.pi/2, 0)))
sim.objs[object_1] = {}

# add obstacle
min_obstacle_position_x = 0.05
max_obstacle_position_x = 0.15
min_obstacle_height = 0.15
max_obstacle_height = 0.25

obstacle_position_x = np.random.uniform(min_obstacle_position_x,
                                        max_obstacle_position_x)
obstacle_height = np.random.uniform(min_obstacle_height,
                                    max_obstacle_height)
obstacle_position_x = 0.1
obstacle_height = 0.25
obstacle_position = (obstacle_position_x,
                     -0.4,
                     sim.table_max_z + obstacle_height/2 - 0.05/2)
obstacle_dimensions = (0.1, 0.15, obstacle_height)
obstacle = sim.add_obstacle(shape='box',
                            dimensions=obstacle_dimensions,
                            position=obstacle_position,
                            orientation=(0, 0, 0),
                            color=sim.obstacle_color)

grasp_position = np.array(object_position)
grasp_position[0] = 0.4
sim.set_gripper_pose((grasp_position, sim.ee_orientation_quat))
sim.close_gripper()

img = take_picture()
save_image(osp.join(os.environ['HOME'],
                    'workspaces/szeth_ws/src/szeth/data/grasp.png'),
           img)

# # Grasp object
# (path, gripper_close_path, post_grasp_path) = sim.compute_grasp_object_path(
#     object_1, table_id=sim.table_id)
# sim.execute_grasp(object_1, sim.get_arm_joints(), sim.get_gripper_joints(),
#                   path, gripper_close_path, post_grasp_path)

# img = take_picture()
# save_image(osp.join(os.environ['HOME'],
#                     'workspaces/szeth_ws/src/szeth/data/grasp.png'),
#            img)

# # Define goal
# min_goal_position_x = -0.2
# max_goal_position_x = -0.1
# goal_position = (-0.1, -0.4, sim.table_max_z_other + 0.15/2 - 0.05/2)
# pre_place_position = np.array(goal_position)
# pre_place_position[2] += 0.05/2

# # Move object
# path = sim.compute_move_end_effector_path(
#     pre_place_position, obstacles=[sim.table_id_other, obstacle, sim.table_id],
#     ik_hack=True)
# imgs = sim.execute_path(sim.get_arm_joints(), path, visualize=False)

# idx = 0
# for img in imgs:
#     filename = osp.join(os.environ['HOME'],
#                         'workspaces/szeth_ws/src/szeth/data/')
#     filename = osp.join(filename, 'path'+str(idx)+'.png')
#     save_image(filename, img)
#     idx += 1

# img = take_picture()
# save_image(osp.join(os.environ['HOME'],
#                     'workspaces/szeth_ws/src/szeth/data/goal.png'),
#            img)

wait_for_user()
disconnect()
