import argparse
import os
import pickle
import time
import warnings
import numpy as np
import pybullet as p
import pybullet_data

from pr2_sim_pybullet.ss_pybullet.pybullet_tools.utils import (
    joint_from_name, joints_from_names, get_subtree_aabb, get_joints,
    enable_gravity, disable_real_time, enable_real_time, TABLE_URDF, set_quat,
    Euler, PI, quat_from_euler, load_model, set_base_values, set_default_camera,
    connect, disconnect, wait_for_user, HideOutput, reset_simulation,
    create_box, create_cylinder, load_pybullet, set_point, set_quat,
    STATIC_MASS, RED, GREEN, BLUE, BLACK, WHITE, BROWN, get_point,
    step_simulation, control_joints, get_joint_positions, get_joints,
    get_link_pose, plan_joint_motion, plan_direct_joint_motion,
    joint_controller_hold, get_min_limits, get_max_limits, WorldSaver,
    create_attachment, set_dynamics, control_joint, get_min_limit,
    joint_controller, add_fixed_constraint, remove_fixed_constraint,
    get_pose, get_aabb, get_aabb_center, get_aabb_extent, get_center_extent,
    set_pose, set_client, get_client, inverse_kinematics, get_max_limit,
    TABLE_URDF_2, remove_body, get_client, get_link_subtree, clone_body,
    get_joint_limits, pairwise_collision, plan_joint_motion_2, set_joint_positions,
    set_joint_position, get_joint, violates_limit, get_joint_position,
    euler_from_quat, GREY, TAN, any_link_pair_collision, take_picture)
from pr2_sim_pybullet.ss_pybullet.pybullet_tools.pr2_utils import (
    PR2_GROUPS, SIDE_HOLDING_LEFT_ARM, REST_LEFT_ARM,
    CENTER_LEFT_ARM, DRAKE_PR2_URDF,
    set_group_conf, get_carry_conf, set_arm_conf, arm_conf, open_arm,
    close_arm, get_gripper_link, get_arm_conf, get_arm_joints,
    get_gripper_joints, close_until_collision, get_gripper_conf,
    set_gripper_conf, open_arm, close_arm)
from pr2_sim_pybullet.ss_pybullet.pybullet_tools.ikfast.pr2.ik import pr2_inverse_kinematics
from pr2_sim_pybullet.ss_pybullet.pybullet_tools.pr2_problems import create_table


CONTROL_ITERATIONS = 20


class PR2SimBullet(object):
    ik_xyz_rpy_cache = {}
    collision_cache = {}
    collision_3d_cache = {}
    fk_cache = {}
    ik_cache = {}

    def __init__(self, use_gui, scenario=1, datapath=None,
                 shadows=False, real_time=False, seed=None):
        self.use_gui, self.datapath, self.shadows = use_gui, datapath, shadows
        self.real_time = real_time
        self.scenario = scenario
        self.seed = seed
        # Initialize GUI if needed
        self.sim_id = connect(use_gui, shadows=shadows)
        set_client(self.sim_id)
        # Add the pybullet_data to the search path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Add any additional datapath to the search path
        if datapath:
            p.setAdditionalSearchPath(datapath)

        # Enable gravity
        enable_gravity()
        # Add floor
        self._create_floor()
        # Enable/disable real time
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        # Set camera
        set_default_camera()

        # Initialize ss-pybullet urdf path
        urdf_path = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(urdf_path, 'ss_pybullet')

        # Create scenario
        self._create_scenario()

        # Dictionary of movable objects
        # object IDs to their properties
        self.objs = {}
        # Dictionary of goals
        # object IDs to their goal locations
        self.goals = {}
        # Dictionary of obstacles (static objects)
        # again, object IDs to their properties
        self.obstacles = {}
        # List of attachments
        self.attachments = []
        # Dictionary of constraints
        self.constraints = {}
        # FK cache
        # self.fk_cache = {}
        # IK cache
        # self.ik_cache = {}
        # XYZ_RPY IK cache
        # self.ik_xyz_rpy_cache = {}
        # self.load_ik_cache()
        # Collision cache
        # self.collision_cache = {}

        # saved_world instance for IK computation
        self.saved_world_ik = WorldSaver()

        # Create constants
        self._create_constants()

        # Redundant joint
        self.redundant_joint = get_joint(
            self.robot_id, 'l_upper_arm_roll_joint')
        # self.redundant_joint = get_joint(
        #     self.robot_id, 'l_shoulder_lift_joint')
        # self.redundant_joint = get_joint(self.robot_id,
        #                                  'l_forearm_roll_joint')
        self.redundant_joint_limits = get_joint_limits(
            self.robot_id, self.redundant_joint)

    def assign(self):
        set_client(self.sim_id)
        return

    def _create_constants(self):
        self.object_color = BLACK
        self.obstacle_color = RED
        self.carry_conf = get_carry_conf('left', 'side')
        return

    def _create_floor(self):
        self.ground_plane_id = p.loadURDF("plane.urdf")
        return

    def _create_robot(self, position=None):
        with HideOutput():
            robot_id = load_model(os.path.join(self.urdf_path, DRAKE_PR2_URDF),
                                  fixed_base=True)
        set_group_conf(robot_id, 'torso', [0.2])
        if position is None:
            position = (0.4, -1.15, PI/2)
        set_base_values(robot_id, position)
        set_arm_conf(robot_id, 'left', get_carry_conf('left', 'side'))
        open_arm(robot_id, 'left')
        set_arm_conf(robot_id, 'right', arm_conf('right', REST_LEFT_ARM))
        close_arm(robot_id, 'right')
        # HACK: Storing end-effector initial orientation, so we can ensure it
        # stays in that orientation throughout
        _, self.ee_orientation_quat = get_link_pose(
            robot_id, get_gripper_link(robot_id, 'left'))
        self.ee_orientation = euler_from_quat(self.ee_orientation_quat)
        return robot_id

    def _create_table(self):
        table_id = load_pybullet(os.path.join(self.urdf_path, TABLE_URDF),
                                 fixed_base=True)
        table_max_z = 0.625
        # set_quat(self.table_id, quat_from_euler(Euler(yaw=PI/2)))
        return table_id, table_max_z

    def _create_other_table(self):
        table_id = load_pybullet(os.path.join(self.urdf_path, TABLE_URDF_2),
                                 fixed_base=True)
        table_max_z = 0.725
        return table_id, table_max_z

    def _create_custom_table(self, width=0.6, length=1.2, height=0.73,
                             thickness=0.03, radius=0.015, color=(0.7, 0.7, 0.7, 1),
                             cylinder=True):
        table_id = create_table(width, length, height,
                                thickness, radius, color)
        table_max_z = height + thickness/2
        return table_id, table_max_z

    def _create_scenario(self):
        table_color = GREY
        if self.scenario == 1:
            # Single table
            # Add robot
            self.robot_id = self._create_robot()

            # Add table
            self.table_id, self.table_max_z = self._create_table()

        elif self.scenario == 2:
            # Two tables with different heights
            # Tall table on the left, short table on the right
            # Add robot
            self.robot_id = self._create_robot((0.35, -1.15, PI/2))
            min_height_table_1 = 0.55
            max_height_table_1 = 0.65

            min_height_table_2 = 0.75
            max_height_table_2 = 0.85
            if self.seed is None:
                table1_height = 0.55
                table2_height = 0.8
            else:
                rng = np.random.RandomState(self.seed)
                table1_height = rng.rand() * (max_height_table_1 - min_height_table_1) + \
                    min_height_table_1
                table2_height = rng.rand() * (max_height_table_2 - min_height_table_2) + \
                    min_height_table_2

            # Add table 1
            # self.table_id, self.table_max_z = self._create_table()
            self.table_id, self.table_max_z = self._create_custom_table(
                height=table1_height, thickness=0.05, width=2.1, length=1.0)
            set_point(self.table_id, (-0.9, 0, 0))

            # Add table 2
            # self.table_id_other, self.table_max_z_other = self._create_other_table()
            self.table_id_other, self.table_max_z_other = self._create_custom_table(
                height=table2_height, thickness=0.05, width=1.5, length=1.0)
            set_point(self.table_id_other, (0.9, 0, 0))
        elif self.scenario == 3:
            # Two tables with different heights
            # Short table on the left, tall table on the left
            # Add robot
            self.robot_id = self._create_robot((0.5, -1.15, PI/2))
            min_height_table_1 = 0.55
            max_height_table_1 = 0.65

            min_height_table_2 = 0.75
            max_height_table_2 = 0.85

            if self.seed is None:
                table1_height = 0.55
                table2_height = 0.8
            else:
                rng = np.random.RandomState(self.seed)
                table1_height = rng.rand() * (max_height_table_1 - min_height_table_1) + \
                    min_height_table_1
                table2_height = rng.rand() * (max_height_table_2 - min_height_table_2) + \
                    min_height_table_2
            # Add table1
            self.table_id, self.table_max_z = self._create_custom_table(
                height=table1_height, thickness=0.05, width=2.0, length=1.0, color=table_color)
            set_point(self.table_id, (0.9, 0, 0))

            # Add table 2
            self.table_id_other, self.table_max_z_other = self._create_custom_table(
                height=table2_height, thickness=0.05, width=1.5, length=1.0, color=table_color)
            set_point(self.table_id_other, (-0.85, 0, 0))

        elif self.scenario == 4:
            # Trying to create a clean scenario
            self.robot_id = self._create_robot((0.5, -1.15, PI/2))

            # Add table1
            table1_height = 0.6
            self.table_id, self.table_max_z = self._create_custom_table(
                height=table1_height, thickness=0.05, width=1.0, length=1.0,
                color=table_color)
            set_point(self.table_id, (0.5, 0, 0))

            # Add table2
            table2_height = 0.8
            self.table_id_other, self.table_max_z_other = self._create_custom_table(
                height=table2_height, thickness=0.05, width=1.0, length=1.0,
                color=table_color
            )
            set_point(self.table_id_other, (-0.5, 0, 0))

    def add_object(self, shape, dimensions, position, orientation, mass, color):
        self.assign()
        assert shape == 'box', "Only box supported for now"
        width, length, height = dimensions
        box_id = create_box(width, length, height, mass, color)
        # Move the box to its position
        set_point(box_id, position)
        # Move the box to its orientation
        quat = quat_from_euler(orientation)
        set_quat(box_id, quat)
        # Set friction of the box
        set_dynamics(box_id, lateralFriction=1.5)

        # Add to dictionary
        self.objs[box_id] = {'shape': shape,
                             'dimensions': dimensions,
                             'position': position,
                             'orientation': orientation,
                             'mass': mass,
                             'color': color}

        self.saved_world_ik = WorldSaver()

        return box_id

    def add_goal(self, object_id, goal_position):
        self.assign()
        assert object_id in self.objs, "Object not created yet"
        self.goals[object_id] = goal_position
        return goal_position

    def add_obstacle(self, shape, dimensions, position, orientation, color):
        self.assign()
        assert shape == 'box', "Only box supported for now"
        width, length, height = dimensions
        box_id = create_box(width, length, height,
                            mass=STATIC_MASS, color=color)
        # Move the box to its position
        set_point(box_id, position)
        # Move the box to its orientation
        quat = quat_from_euler(orientation)
        set_quat(box_id, quat)

        # Add to dictionary
        self.obstacles[box_id] = {'shape': shape,
                                  'dimensions': dimensions,
                                  'position': position,
                                  'orientation': orientation,
                                  'color': color}

        self.saved_world_ik = WorldSaver()

        return box_id

    def hold_position(self, group=None):
        self.assign()
        if group is None:
            # Hold all joints in position
            joints = get_joints(self.robot_id)
        else:
            joints = joints_from_names(self.robot_id, PR2_GROUPS[group])

        joints_state = get_joint_positions(self.robot_id, joints)
        return control_joints(self.robot_id, joints, joints_state)

    def compute_grasp_object_path(self, object_id, table_id=None, obstacles=[],
                                  ik_hack=False):
        self.assign()
        assert object_id in self.objs, "Object not created yet"
        if table_id is None:
            table_id = self.table_id
        if obstacles is None:
            obstacles = list(self.obstacles.keys())
        # Get object position
        position = get_point(object_id)
        # dimensions = self.objs[object_id]['dimensions']
        # Obtain a pre-grasp position
        pre_grasp_position = np.array(position)
        # pre_grasp_position[0] += dimensions[0]
        pre_grasp_position[1] -= 0.1
        # Get gripper link and set its target pose
        gripper_link = get_gripper_link(self.robot_id, 'left')
        gripper_position, gripper_orientation = get_link_pose(
            self.robot_id, gripper_link)
        gripper_pose = (pre_grasp_position, gripper_orientation)
        # Save the world so we can restore it later
        saved_world = WorldSaver()
        # if ik_hack:
        #     self.saved_world_ik.restore()
        # Do inverse kinematics to figure out the required pre-grasp joint conf
        # pre_grasp_joint_conf = pr2_inverse_kinematics(
        #     self.robot_id, 'left', gripper_pose,
        #     obstacles=[object_id, table_id]+obstacles)
        pre_grasp_joint_conf = self.arm_ik_2(
            gripper_pose, obstacles=[object_id, table_id]+obstacles, ik_hack=ik_hack)
        if pre_grasp_joint_conf is None:
            warnings.warn('Pre-grasp position IK failed')
            saved_world.restore()
            return None
        # print('Pre-grasp IK done')
        # Compute grasp position
        grasp_position = np.array(position)
        gripper_pose = (grasp_position, gripper_orientation)
        # Do inverse kinematics to figure out the required grasp joint conf
        # grasp_joint_conf = pr2_inverse_kinematics(
        #     self.robot_id, 'left', gripper_pose,
        #     obstacles=[table_id]+obstacles)
        grasp_joint_conf = self.arm_ik_2(
            gripper_pose, obstacles=[table_id]+obstacles)
        if grasp_joint_conf is None:
            warnings.warn('Grasp position IK failed')
            saved_world.restore()
            return None
        # print('Grasp IK done')
        # Compute post-grasp position
        post_grasp_position = np.array(position)
        post_grasp_position[2] += 0.1  # 0.05
        gripper_pose = (post_grasp_position, gripper_orientation)
        # Do inverse kinematics to figure out the required post-grasp joint conf
        # post_grasp_joint_conf = pr2_inverse_kinematics(
        #     self.robot_id, 'left', gripper_pose,
        #     obstacles=obstacles)
        post_grasp_joint_conf = self.arm_ik_2(
            gripper_pose, obstacles=obstacles)
        if post_grasp_joint_conf is None:
            warnings.warn('Post-grasp position IK failed')
            saved_world.restore()
            return None
        # print('Post-grasp IK done')
        # Compute paths
        # First, restore the world
        saved_world.restore()
        arm_joints = get_arm_joints(self.robot_id, 'left')
        gripper_joints = self.get_gripper_joints()
        # Second, plan joint motion from initial conf to pre-grasp conf
        resolutions = 0.05 ** np.ones(len(arm_joints))
        pre_grasp_path = plan_joint_motion(
            self.robot_id,
            arm_joints,
            pre_grasp_joint_conf,
            attachments=self.attachments,
            obstacles=[table_id, object_id] + obstacles,
            self_collisions=False,
            resolutions=resolutions,
            restarts=2, iterations=25, smooth=25)
        if pre_grasp_path is None:
            warnings.warn('Pre-grasp path computation failed')
            saved_world.restore()
            return None
        # print('Pre-grasp path computation done')
        # Third, plan direct joint motion from pre-grasp conf to grasp conf
        # grasp_path = plan_direct_joint_motion(
        #     self.robot_id,
        #     arm_joints,
        #     grasp_joint_conf,
        #     attachments=self.attachments,
        #     # obstacles=[table_id]+obstacles,
        #     self_collisions=False,
        #     resolutions=resolutions/2.0)
        grasp_path = plan_joint_motion(
            self.robot_id, arm_joints, grasp_joint_conf, attachments=self.attachments,
            self_collisions=False, resolutions=resolutions/2.0, restarts=2,
            iterations=25, smooth=25
        )
        if grasp_path is None:
            warnings.warn('Grasp path computation failed')
            saved_world.restore()
            return None
        # print('Grasp path computation done')
        # plan the gripper close path
        gripper_close_path = close_until_collision(
            self.robot_id, gripper_joints, bodies=[object_id], return_path=True)
        if gripper_close_path is None:
            warnings.warn('Gripper closing path computation failed')
            saved_world.restore()
            return None
        # print('Gripper closing path computation done')
        # Finally, plan joint motion from grasp conf to post-grasp conf
        post_grasp_path = plan_joint_motion(
            self.robot_id,
            arm_joints,
            post_grasp_joint_conf,
            attachments=[create_attachment(
                self.robot_id, gripper_link, object_id)],
            # obstacles=obstacles,
            self_collisions=False,
            resolutions=resolutions/2.0,
            restarts=2, iterations=25, smooth=25)
        if post_grasp_path is None:
            warnings.warn('Post-grasp path computation failed')
            saved_world.restore()
            return None
        # print('Post-grasp path computation done')
        # Concatenate paths until grasp
        path = pre_grasp_path + grasp_path
        # Restore the world before control
        saved_world.restore()

        return path, gripper_close_path, post_grasp_path

    def compute_place_object_path(self, object_id, goal_position, table_id=None, obstacles=[]):
        self.assign()
        saved_world = WorldSaver()
        assert object_id in self.objs, "Object not created yet"
        if table_id is None:
            table_id = self.table_id
        if obstacles is None:
            obstacles = list(self.obstacles.keys())
        # HACK: Assuming that the object is already at pre_place position
        # when this function is called
        gripper_position, gripper_orientation = self.get_end_effector_pose()
        place_pose = (goal_position, gripper_orientation)
        # Do inverse kinematics to figure out the required place joint conf
        # place_joint_conf = pr2_inverse_kinematics(self.robot_id, 'left', place_pose,
        #                                           obstacles=[
        #                                               table_id
        #                                           ]+obstacles)
        place_joint_conf = self.arm_ik_2(place_pose)
        if place_joint_conf is None:
            warnings.warn('Place position IK failed')
            saved_world.restore()
            return None
        # Compute post_place position for the gripper
        self.saved_world_ik.restore()
        place_position = goal_position.copy()
        place_position[1] -= 0.1
        post_place_pose = (place_position, gripper_orientation)
        # Do inverse kinematics to figure out the required post_place joint conf
        # post_place_joint_conf = pr2_inverse_kinematics(self.robot_id, 'left',
        #                                                post_place_pose,
        #                                                obstacles=[
        #                                                    table_id
        #                                                ]+obstacles)
        post_place_joint_conf = self.arm_ik_2(post_place_pose)
        if post_place_joint_conf is None:
            warnings.warn('Post place position IK failed')
            saved_world.restore()
            return None
        # Compute paths
        # First, restore the world
        saved_world.restore()
        arm_joints = get_arm_joints(self.robot_id, 'left')
        # Compute path to the place conf ignoring collision with table
        resolutions = 0.05 ** np.ones(len(arm_joints))
        # TODO: Should I be using plan_direct_joint_motion instead?
        # TODO: Increase resolution maybe?
        place_path = plan_joint_motion(
            self.robot_id, arm_joints, place_joint_conf,
            attachments=self.attachments,
            obstacles=obstacles,
            self_collisions=False,
            resolutions=resolutions,
            restarts=2, iterations=25, smooth=25
        )
        if place_path is None:
            warnings.warn('Place path computation failed')
            saved_world.restore()
            return None
        # Compute path from place conf to post_place conf
        post_place_path = plan_joint_motion(
            self.robot_id, arm_joints, post_place_joint_conf,
            attachments=[],
            self_collisions=False,
            resolutions=resolutions,
            restarts=2, iterations=25, smooth=25
        )
        if post_place_path is None:
            warnings.warn('Post place path computation failed')
            saved_world.restore()
            return None
        # Finally, plan the gripper open path
        gripper_open_path = [self.get_gripper_open_conf() for _ in range(10)]
        # Restore the world before control
        saved_world.restore()

        return place_path, gripper_open_path, post_place_path

    def get_end_effector_pose(self):
        self.assign()
        gripper_link = get_gripper_link(self.robot_id, 'left')
        gripper_position, gripper_orientation = get_link_pose(
            self.robot_id, gripper_link)
        return np.array(gripper_position), np.array(gripper_orientation)

    def get_gripper_open_conf(self):
        gripper_open_conf = [get_max_limit(
            self.robot_id, joint) for joint in self.get_gripper_joints()]
        return gripper_open_conf

    def add_attachment(self, object_id):
        self.assign()
        gripper_link = get_gripper_link(self.robot_id, 'left')
        self.attachments.append(create_attachment(self.robot_id,
                                                  gripper_link,
                                                  object_id))
        return

    def remove_attachment(self, object_id):
        self.assign()
        gripper_link = get_gripper_link(self.robot_id, 'left')
        for attachment in self.attachments:
            if (attachment.parent == self.robot_id) and (attachment.parent_link == gripper_link) and (attachment.child == object_id):
                self.attachments.remove(attachment)
        return

    def add_constraint(self, object_id):
        self.assign()
        gripper_link = get_gripper_link(self.robot_id, 'left')
        constraint = add_fixed_constraint(
            object_id, self.robot_id, gripper_link)
        self.constraints[object_id] = constraint
        return

    def remove_constraint(self, object_id):
        self.assign()
        gripper_link = get_gripper_link(self.robot_id, 'left')
        remove_fixed_constraint(object_id, self.robot_id, gripper_link)
        del self.constraints[object_id]
        return

    def compute_move_end_effector_path(self, ee_position,
                                       ee_orientation=None,
                                       obstacles=[],
                                       ik_hack=False,
                                       check_attachments=True):
        self.assign()
        saved_world = WorldSaver()
        if ik_hack:
            # HACK: For some reason, IK fails if there is an obstacle in between
            self.saved_world_ik.restore()
        if obstacles is None:
            obstacles = list(self.obstacles.keys()) + [self.table_id]
        # set_arm_conf(self.robot_id, 'left', get_carry_conf('left', 'side'))
        # Get the desired end effector pose
        # gripper_link = get_gripper_link(self.robot_id, 'left')
        if ee_orientation is None:
            # _, ee_orientation = get_link_pose(self.robot_id, gripper_link)
            ee_orientation = self.ee_orientation_quat
        gripper_pose = (ee_position, ee_orientation)
        # Do IK to get the required joint configuration
        # joint_conf = pr2_inverse_kinematics(
        #     self.robot_id, 'left', gripper_pose,
        #     obstacles=obstacles)
        joint_conf = self.arm_ik_2(
            gripper_pose, obstacles=obstacles)
        # joint_conf = self.arm_ik_xyz_rpy(gripper_pose, obstacles=obstacles)
        # import ipdb
        # ipdb.set_trace()
        if joint_conf is None:
            warnings.warn('IK failed')
            saved_world.restore()
            return None
        # Restore world before planning
        saved_world.restore()
        # Compute path
        attachments = self.attachments if check_attachments else []
        arm_joints = get_arm_joints(self.robot_id, 'left')
        resolutions = 0.05 ** np.ones(len(arm_joints)) / 2
        path = plan_joint_motion_2(
            self.robot_id,
            arm_joints,
            self.get_gripper_link(),
            joint_conf,
            attachments=attachments,
            obstacles=obstacles,
            self_collisions=False,
            resolutions=resolutions,
            restarts=2, iterations=25, smooth=25)
        if path is None:
            warnings.warn('Planning failed')
            saved_world.restore()
            return None
        # Restore the world before control
        saved_world.restore()
        return path

    def compute_move_arm_path(self, joint_conf, obstacles=[], check_attachments=True):
        self.assign()
        if obstacles is None:
            obstacles = list(self.obstacles.keys()) + [self.table_id]

        saved_world = WorldSaver()
        arm_joints = get_arm_joints(self.robot_id, 'left')
        resolutions = 0.05 ** np.ones(len(arm_joints))
        attachments = self.attachments if check_attachments else []
        path = plan_joint_motion_2(self.robot_id, arm_joints, self.get_gripper_link(),
                                   joint_conf,
                                   attachments=attachments,
                                   obstacles=obstacles,
                                   self_collisions=False, resolutions=resolutions,
                                   restarts=2, iterations=25, smooth=25)
        saved_world.restore()

        return path

    def execute_grasp(self, object_id, arm_joints, gripper_joints,
                      pre_grasp_and_grasp_path, gripper_close_path,
                      post_grasp_path):
        self.assign()
        # Pre-grasp and grasp
        self.execute_path(arm_joints, pre_grasp_and_grasp_path)
        self.execute_path(gripper_joints, gripper_close_path)
        # Create attachments and constraints
        self.add_attachment(object_id)
        self.add_constraint(object_id)
        if post_grasp_path is not None:
            # Post-grasp
            self.execute_path(arm_joints, post_grasp_path)
        return

    def execute_place(self, object_id, arm_joints, gripper_joints,
                      path, gripper_open_path, post_place_path):
        self.assign()
        # Place
        self.execute_path(arm_joints, path)
        self.execute_path(gripper_joints, gripper_open_path)
        # Remove attachments and constraints
        self.remove_attachment(object_id)
        self.remove_constraint(object_id)
        if post_place_path is not None:
            # post_place
            self.execute_path(arm_joints, post_place_path)
        return

    def execute_path(self, joints, path, visualize=False):
        self.assign()
        imgs = []
        for conf in path:
            if visualize:
                imgs.append(take_picture())
            self.execute_move_to_joint_conf_action(joints, conf)
        return imgs

    def execute_move_to_joint_conf_action(self, joints, conf):
        self.assign()
        n_iterations = 0
        for _ in joint_controller_hold(self.robot_id, joints, conf):
            step_simulation()
            n_iterations += 1
            if n_iterations >= CONTROL_ITERATIONS:
                break
        return

    def get_arm_joints(self):
        self.assign()
        return get_arm_joints(self.robot_id, 'left')

    def get_gripper_joints(self):
        self.assign()
        return get_gripper_joints(self.robot_id, 'left')

    def get_arm_conf(self):
        self.assign()
        return np.array(get_arm_conf(self.robot_id, 'left'), dtype=np.float32)

    def set_arm_conf(self, conf):
        self.assign()
        return set_arm_conf(self.robot_id, 'left', conf)

    def get_gripper_conf(self):
        self.assign()
        return get_gripper_conf(self.robot_id, 'left')

    def set_gripper_conf(self, conf):
        self.assign()
        return set_gripper_conf(self.robot_id, 'left', conf)

    def get_gripper_link(self):
        self.assign()
        return get_gripper_link(self.robot_id, 'left')

    def get_arm_joint_limits(self):
        self.assign()
        arm_joints = self.get_arm_joints()
        joint_limits = []
        for joint in arm_joints:
            joint_limits.append(get_joint_limits(self.robot_id, joint))
        return np.array(joint_limits, dtype=np.float32)

    def get_gripper_pose(self):
        self.assign()
        gripper_link = self.get_gripper_link()
        return get_link_pose(self.robot_id, gripper_link)

    def set_gripper_pose(self, gripper_pose, ik_hack=False):
        self.assign()
        saved_world = WorldSaver()
        # Restore default arm conf
        self.saved_world_ik.restore()
        # Do IK to figure out the required joint conf
        #start = time.time()
        # joint_conf = pr2_inverse_kinematics(
        #     self.robot_id, 'left', gripper_pose,
        #     obstacles=list(self.obstacles.keys())+[self.table_id],
        #     pos_tolerance=1e-3, ori_tolerance=1e-3*np.pi)
        joint_conf = self.arm_ik_2(gripper_pose, ik_hack=ik_hack)
        #end = time.time()
        #print('Gripper IK takes', end-start, 'seconds')
        # import ipdb
        # ipdb.set_trace()
        if joint_conf is None:
            # raise Exception('IK failed while setting gripper pose')
            warnings.warn('IK failed while setting gripper pose')
            saved_world.restore()
            return None
        # Restore world
        saved_world.restore()
        # Set arm joint conf
        set_arm_conf(self.robot_id, 'left', joint_conf)
        # HACK: Check if there are attachments, if so move them as well
        if len(self.attachments) != 0:
            for attachment in self.attachments:
                body = attachment.child
                set_pose(body, gripper_pose)

        return self.get_gripper_pose()

    def arm_ik(self, gripper_pose):
        self.assign()
        arm, arm_gripper_link = self._create_fake_arm()
        joint_conf = p.calculateInverseKinematics(arm, arm_gripper_link,
                                                  targetPosition=gripper_pose[0],
                                                  targetOrientation=gripper_pose[1],
                                                  physicsClientId=get_client())
        self._destroy_fake_arm(arm)
        return joint_conf[:7]

    def arm_ik_2(self, gripper_pose, obstacles=[], custom_limits={},
                 use_ikfast=False, ik_hack=False):
        self.assign()
        cache_key = tuple(gripper_pose[0]) + tuple(gripper_pose[1])
        if cache_key in PR2SimBullet.ik_cache:
            return PR2SimBullet.ik_cache[cache_key]
        saved_world = WorldSaver()
        if ik_hack:
            self.saved_world_ik.restore()
        # start = time.time()
        joint_conf = pr2_inverse_kinematics(
            self.robot_id, 'left', gripper_pose,
            obstacles=obstacles, custom_limits=custom_limits,
            use_ikfast=use_ikfast)
        # end = time.time()
        # print('Time for IK', end-start)
        if ik_hack:
            saved_world.restore()
        PR2SimBullet.ik_cache[cache_key] = joint_conf
        return joint_conf

    def arm_ik_xyz_rpy(self, gripper_pose, redundant_angle=None, obstacles=[], custom_limits={},
                       ik_hack=False):
        self.assign()
        # TODO: Cache IK for speed
        cache_key = tuple(gripper_pose[0]) + tuple(gripper_pose[1])
        cache_key += (redundant_angle,)
        if cache_key in PR2SimBullet.ik_xyz_rpy_cache:
            return PR2SimBullet.ik_xyz_rpy_cache[cache_key]
        # Save world
        saved_world = WorldSaver()
        if redundant_angle is not None:
            # Check joint limit for redundant angle
            if violates_limit(self.robot_id, self.redundant_joint, redundant_angle):
                warnings.warn('Violates redundant joint limits')
                PR2SimBullet.ik_xyz_rpy_cache[cache_key] = None
                return None
            # Set the redundant angle for the robot
            set_joint_position(
                self.robot_id, self.redundant_joint, redundant_angle)
        else:
            # Use the current redundant angle
            redundant_angle = get_joint_position(
                self.robot_id, self.redundant_joint)
        # Do IK restricting the redundant joint
        # setting upper_limits to None ensures that the
        # redundant joint is not moved
        joint_conf = pr2_inverse_kinematics(
            self.robot_id, 'left', gripper_pose, obstacles=obstacles,
            use_ikfast=True, upper_limits=None)
        # Restore world
        saved_world.restore()
        PR2SimBullet.ik_xyz_rpy_cache[cache_key] = joint_conf
        return joint_conf

    def get_joint_conf_from_xyz_rpy(self, conf):
        gripper_position = conf[:3]
        gripper_orientation = conf[3:6]
        gripper_orientation_quat = quat_from_euler(gripper_orientation)
        gripper_pose = (gripper_position, gripper_orientation_quat)
        redundant_angle = conf[6]

        return self.arm_ik_xyz_rpy(gripper_pose, redundant_angle)

    def get_xyz_rpy(self):
        gripper_position, gripper_orientation_quat = self.get_gripper_pose()
        gripper_orientation = euler_from_quat(gripper_orientation_quat)
        redundant_angle = get_joint_position(
            self.robot_id, self.redundant_joint)
        conf = np.zeros(7, dtype=np.float32)
        conf[:3] = gripper_position
        conf[3:6] = gripper_orientation
        conf[6] = redundant_angle
        return conf

    def arm_fk(self, joint_conf):
        self.assign()
        # HACK: Storing FK
        if tuple(joint_conf) in PR2SimBullet.fk_cache:
            return PR2SimBullet.fk_cache[tuple(joint_conf)]
        saved_world = WorldSaver()
        self.set_arm_conf(joint_conf)
        gripper_pose = self.get_gripper_pose()
        saved_world.restore()
        PR2SimBullet.fk_cache[tuple(joint_conf)] = gripper_pose
        return gripper_pose

    def check_arm_collision(self, joint_conf, obstacles, attachment=None):
        self.assign()
        # HACK: Storing collision checks
        if tuple(joint_conf) in PR2SimBullet.collision_cache:
            return PR2SimBullet.collision_cache[tuple(joint_conf)]
        saved_world = WorldSaver()
        self.set_arm_conf(joint_conf)
        collision = False
        if any(pairwise_collision(self.robot_id, b) for b in obstacles):
            collision = True
        if not collision:
            if attachment is not None:
                # Manually place the attachment to the end effector
                set_pose(attachment, self.get_gripper_pose())
                if any(pairwise_collision(attachment, b) for b in obstacles):
                    collision = True
        saved_world.restore()
        PR2SimBullet.collision_cache[tuple(joint_conf)] = collision
        return collision

    def check_xyzrpy_collision(self, conf, obstacles=[], attachment=None):
        self.assign()
        if tuple(conf) in PR2SimBullet.collision_3d_cache:
            return PR2SimBullet.collision_3d_cache[tuple(conf)]
        saved_world = WorldSaver()
        gripper_pose = (conf[:3], quat_from_euler(conf[3:6]))
        self.set_gripper_pose(gripper_pose)
        if attachment is not None:
            set_pose(attachment, gripper_pose)
        collision = False
        collision = any(any_link_pair_collision(
            self.robot_id, [self.get_gripper_link()], b) for b in obstacles)
        if not collision:
            collision = any(
                pairwise_collision(attachment, b) for b in obstacles)
        saved_world.restore()
        PR2SimBullet.collision_3d_cache[tuple(conf)] = collision
        return collision

    def check_edge_collision(self, start_conf, goal_conf, obstacles):
        self.assign()
        # Do cartesian path planning from start_conf to goal_conf
        # and check for collision
        raise NotImplementedError

    def _create_fake_arm(self):
        # HACK: Create a clone of the arm that can be used for IK
        arm_joints = get_arm_joints(self.robot_id, 'left')
        arm_links = get_link_subtree(self.robot_id, arm_joints[0])
        arm = clone_body(self.robot_id, links=arm_links,
                         visual=False, collision=False)
        arm_gripper_link = arm_links.index(
            get_gripper_link(self.robot_id, 'left'))
        return arm, arm_gripper_link

    def _destroy_fake_arm(self, arm):
        remove_body(arm)
        return

    def get_pose(self, object_id):
        self.assign()
        return get_pose(object_id)

    def set_pose(self, object_id, pose):
        self.assign()
        position, orientation = pose
        if orientation is None:
            orientation = quat_from_euler((0, 0, 0))
        pose = (position, orientation)
        return set_pose(object_id, pose)

    def get_position(self, object_id):
        self.assign()
        return get_point(object_id)

    def set_position(self, object_id, position):
        self.assign()
        return set_point(object_id, position)

    def open_gripper(self):
        self.assign()
        return open_arm(self.robot_id, 'left')

    def close_gripper(self):
        self.assign()
        return close_arm(self.robot_id, 'left')

    def get_aabb(self, object_id):
        self.assign()
        return get_aabb(object_id)

    def get_aabb_center(self, object_id):
        self.assign()
        return get_aabb_center(object_id)

    def get_aabb_extent(self, object_id):
        self.assign()
        return get_aabb_extent(object_id)

    def get_center_extent(self, object_id):
        self.assign()
        return get_center_extent(object_id)

    def get_simulation_state(self):
        self.assign()
        saved_world = WorldSaver()
        return saved_world

    def set_simulation_state(self, saved_world):
        self.assign()
        saved_world.restore()
        return

    def remove_body(self, body_id):
        self.assign()
        return remove_body(body_id)

    def reset(self):
        self.assign()
        # Remove any attachments
        for attachment in self.attachments:
            self.remove_attachment(attachment.child)
        # Remove any constraints
        list_of_objects_with_constraints = list(self.constraints.keys())
        for object_id in list_of_objects_with_constraints:
            self.remove_constraint(object_id)
        reset_simulation()
        enable_gravity()
        self._create_floor()
        if self.real_time:
            enable_real_time()
        else:
            disable_real_time()
        set_default_camera()
        self._create_scenario()
        return

    def close(self):
        self.assign()
        disconnect()
        # self.save_ik_cache()
        return

    def wait_for_user(self):
        self.assign()
        wait_for_user()
        return

    def load_ik_cache(self):
        path = os.path.join(
            os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/data/ik_cache.pkl')
        if os.path.exists(path):
            # IK cache file exists
            PR2SimBullet.ik_xyz_rpy_cache = pickle.load(open(path, 'rb'))
        return

    def save_ik_cache(self):
        path = os.path.join(
            os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/data/ik_cache.pkl')
        pickle.dump(PR2SimBullet.ik_xyz_rpy_cache, open(path, 'wb'))
        return


if __name__ == '__main__':
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    # Create simulation
    sim = PR2SimBullet(use_gui=True, scenario=args.scenario)
    arm_joints = get_arm_joints(sim.robot_id, 'left')
    gripper_joints = get_gripper_joints(sim.robot_id, 'left')
    if args.scenario == 1:
        # Create an object
        object_1 = sim.add_object(shape='box',
                                  dimensions=(0.05, 0.05, 0.15),
                                  position=(-0.1, -0.4,
                                            sim.table_max_z + 0.15/2),
                                  orientation=(0, 0, 0),
                                  mass=2,
                                  color=BLUE)
        # Create an obstacle
        obstacle_1 = sim.add_obstacle(shape='box',
                                      dimensions=(0.15, 0.15, 0.25),
                                      position=(
                                          0.1, -0.4, sim.table_max_z + 0.25/2),
                                      orientation=(0, 0, 0),
                                      color=RED)
        # Create a goal
        goal_1 = np.array((
            0.4, -0.4, sim.table_max_z + 0.15/2))
        pre_place_1 = goal_1.copy()
        pre_place_1[2] += 0.05
        # Plan a path to grasp the object
        (path,
         gripper_close_path,
         post_grasp_path) = sim.compute_grasp_object_path(object_1)
        # Execute grasp
        sim.execute_grasp(object_1, arm_joints, gripper_joints, path,
                          gripper_close_path, post_grasp_path)

        # Move end effector to goal
        path = sim.compute_move_end_effector_path(
            pre_place_1, ik_hack=True)
        # Execute path
        sim.execute_path(arm_joints, path)

        # Place the object at the place location
        (path, gripper_open_path, post_place_path) = sim.compute_place_object_path(
            object_1, goal_1)
        # Execute place
        sim.execute_place(object_1, arm_joints, gripper_joints, path,
                          gripper_open_path, post_place_path)

        # Print position of the object
        print(get_point(object_1))
    elif args.scenario == 2:
        # Create an object on the first table
        object_1 = sim.add_object(shape='box',
                                  dimensions=(0.05, 0.05, 0.15),
                                  position=(-0.2, -0.4,
                                            sim.table_max_z + 0.15/2),
                                  orientation=(0, 0, 0),
                                  mass=0.1,
                                  color=BLUE)

        # Create a goal
        # goal_1 = np.array((0.35, -0.25, sim.table_max_z_other + 0.15/2))
        goal_1 = np.array((0.35, -0.4, sim.table_max_z_other + 0.15/2))
        pre_place_1 = goal_1.copy()
        pre_place_1[2] += 0.05
        # Plan a path to grasp the object
        (path, gripper_close_path,
         post_grasp_path) = sim.compute_grasp_object_path(object_1, table_id=sim.table_id)
        # Execute grasp
        sim.execute_grasp(object_1, arm_joints, gripper_joints,
                          path, gripper_close_path, post_grasp_path)

        # Move end effector to goal
        path = sim.compute_move_end_effector_path(
            pre_place_1, obstacles=[sim.table_id_other, sim.table_id], ik_hack=True)
        # Execute path
        sim.execute_path(arm_joints, path)

        # Place the object at the place location
        (path, gripper_open_path, post_place_path) = sim.compute_place_object_path(
            object_1, goal_1, table_id=sim.table_id_other)
        # Execute path
        sim.execute_place(object_1, arm_joints, gripper_joints,
                          path, gripper_open_path, post_place_path)

        # Teleport object back to start
        set_point(object_1, (-0.2, -0.4, sim.table_max_z + 0.15/2))
        # Teleport arm to carry conf
        sim.set_arm_conf(sim.carry_conf)
        # Grasp it again
        # Plan a path to grasp the object
        (path, gripper_close_path,
         post_grasp_path) = sim.compute_grasp_object_path(object_1, table_id=sim.table_id)
        # Execute grasp
        sim.execute_grasp(object_1, arm_joints, gripper_joints,
                          path, gripper_close_path, post_grasp_path)

    elif args.scenario == 3:
        # Create an object on the short table
        object_1 = sim.add_object(shape='box', dimensions=(0.05, 0.05, 0.15),
                                  position=(
                                      0.35, -0.4, sim.table_max_z + 0.15/2),
                                  orientation=(0, 0, 0),
                                  mass=1,
                                  color=BLUE)

        # Create a goal
        goal_1 = np.array((-0.2, -0.4, sim.table_max_z_other + 0.15/2))
        pre_place_1 = goal_1.copy()
        pre_place_1[2] += 0.05
        # Plan a path to grasp the object
        (path, gripper_close_path, post_grasp_path) = sim.compute_grasp_object_path(
            object_1, table_id=sim.table_id)
        # Execute grasp
        sim.execute_grasp(object_1, arm_joints, gripper_joints,
                          path, gripper_close_path, post_grasp_path)

        # Move end effector to goal
        path = sim.compute_move_end_effector_path(
            pre_place_1, obstacles=[sim.table_id_other, sim.table_id], ik_hack=True)
        # Execute path
        sim.execute_path(arm_joints, path)

        # # Place the object at the place location
        # (path, gripper_open_path, post_place_path) = sim.compute_place_object_path(
        #     object_1, goal_1, table_id=sim.table_id_other)
        # # Execute path
        # sim.execute_place(object_1, arm_joints, gripper_joints,
        #                   path, gripper_open_path, post_place_path)

    wait_for_user()
    disconnect()
