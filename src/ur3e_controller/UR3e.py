#! /usr/bin/env python3

import sys
from copy import deepcopy

import tf2_ros
import rospy
import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory, Constraints, JointConstraint, OrientationConstraint

from sensor_msgs.msg import JointState

from roboticstoolbox.models import UR3
from ur3e_controller.utility import *
from ur3e_controller.gripper import Gripper
from ur3e_controller.collision_manager import CollisionManager
from scipy.spatial.transform import Rotation as R

import spatialmath.base as smb
from spatialmath import SE3
from scipy.optimize import minimize
import numpy as np


# Constants variables
POS_TOL = 0.001  # m
ORI_TOL = 0.001  # m
MAX_VEL_SCALE_FACTOR = 0.05
MAX_ACC_SCALE_FACTOR = 0.05
GRIPPER_OPEN = 500  # 0.1mm
GRIPPER_CLOSE = 200  # 0.1mm

# predefined configurations as reference guess for IK solver
PREDEFINED_CONFIGS = {
    'BACK':  [pi/2, -pi/2, -pi/2,  -pi/2, pi/2, 0.38],
    "FRONT": [-pi/2, -pi/2, 0, -1.19, -pi/2, 0.38],
    "LEFT":  [pi, -pi/2, 0,  -1.19, -pi/2, 0.38],
    "RIGHT": [0, -pi/2, 0, -1.19, -pi/2, 0.38],
}


class UR3e:

    r"""
    A class to control the UR3e robot arm using moveit_commander.

    """
    DETECT_CONFIG = 'detect_config'
    HOME_BACK_SIDE = "home_back_side"
    HOME = "home_back_side"

    def __init__(self) -> None:

        moveit_commander.roscpp_initialize(sys.argv)
        self._robot = RobotCommander()
        self._scene = PlanningSceneInterface()
        self._group = MoveGroupCommander("ur3e")
        self._gripper = Gripper()
        self.constraints = Constraints()
        self.virtual_UR = UR3()
        self.virtual_UR_tool = SE3(0, 0, 0.2)

        self._planning_frame = self._group.get_planning_frame()
        self._group_names = self._robot.get_group_names()
        self._eef_link = self._group.get_end_effector_link()
        self._cur_js = self._group.get_current_joint_values()

        self._display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path", DisplayTrajectory, queue_size=20)

        self._marker_pub = rospy.Publisher(
            "visualization_marker", Marker, queue_size=10)

        self._joint_states_sub = rospy.Subscriber(
            "/joint_states", JointState, self._joint_states_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        setup_completed = self.movegroup_setup()

        rospy.logdebug(f"============ Planning frame: {self._planning_frame}")
        rospy.logdebug(f"============ End effector link: {self._eef_link}")

    def movegroup_setup(self):
        r"""
        Setup the move group.
        """

        self._group.set_start_state_to_current_state()
        self._group.set_planner_id("RRTConnect")
        self._group.set_planning_time(10)
        self._group.set_num_planning_attempts(5)
        self._group.set_goal_position_tolerance(POS_TOL)
        self._group.set_goal_joint_tolerance(POS_TOL)
        self._group.set_max_velocity_scaling_factor(MAX_VEL_SCALE_FACTOR)
        self._group.set_max_acceleration_scaling_factor(MAX_ACC_SCALE_FACTOR)
        self.init_path_constraints()

        # Add a small fragment piece as gripper cable node
        cable_cap_pose = PoseStamped()
        cable_cap_pose.pose = list_to_pose([-0.045, -0.01, 0.01, 1.57, 0, 1.57])
        cable_cap_pose.header.frame_id = "tool0"
        self._scene.add_cylinder("cable_cap", cable_cap_pose, 0.02, 0.01)
        self._scene.attach_mesh("tool0", "cable_cap", touch_links=[
            "onrobot_rg2_base_link", "wrists_3_link"])

        # Add box to wrap around the camera mounter
        camera_mount_pose = PoseStamped()
        camera_mount_pose.pose = list_to_pose( [0.0, -0.0455, 0.0732, 0.0, 0.0, 0.0])
        camera_mount_pose.header.frame_id = "tool0"
        self._scene.add_box("camera_mount", camera_mount_pose, size=(0.08, 0.1, 0.035))
        self._scene.attach_mesh("tool0", "camera_mount", touch_links=[
            "onrobot_rg2_base_link"])

        # Add a thin box above the crate to restraint the robot motion not to go downward too much
        wall_pose = PoseStamped()
        wall_pose.pose = list_to_pose([0.0, -0.61, -0.07, 0.0, 0.0, 0.0])
        wall_pose.header.frame_id = "base_link"
        bound_id = "wall"
        self._scene.add_box(bound_id, wall_pose, size=(2, 0.01, 2))

        # Add a ceiling to avoid robot with bottle to go overhead
        ceilling_pose = PoseStamped()
        ceilling_pose.pose = list_to_pose([0.0, 0.0, 0.7, 0.0, 0.0, 0.0])
        ceilling_pose.header.frame_id = "base_link"
        ceilling_id = "ceilling"
        self._scene.add_box(ceilling_id, ceilling_pose, size=(2, 2, 0.01))

        return True
    
    def add_crate_bound(self):
        r"""
        add a thin box above the crate to restraint the robot motion not to go downward too much
        """

        bound_pose = PoseStamped()
        bound_pose.pose = list_to_pose([0.0, -0.36, -0.07, 0.0, 0.0, 0.0])
        bound_pose.header.frame_id = "base_link"
        bound_id = "bound"
        self._scene.add_box(bound_id, bound_pose, size=(0.45, 0.35, 0.01))

        return bound_id
    
    def remove_crate_bound(self, bound_id):
        r"""
        remove the thin box above the crate
        """
        if self._scene.get_objects(bound_id) is not None:
            self._scene.remove_world_object(bound_id[0])


    def init_path_constraints(self):

        constraints = Constraints()
        constraints.name = "elbow_up"

        joint_constraint_04 = JointConstraint()
        joint_constraint_04.joint_name = "shoulder_pan_joint"
        joint_constraint_04.position = 2.0944
        joint_constraint_04.tolerance_above = 2.1944
        joint_constraint_04.tolerance_below = 2.1944
        joint_constraint_04.weight = 1
        constraints.joint_constraints.append(joint_constraint_04)

        joint_constraint_02 = JointConstraint()
        joint_constraint_02.joint_name = "shoulder_lift_joint"
        joint_constraint_02.position = -2.443
        joint_constraint_02.tolerance_above = np.pi/4
        joint_constraint_02.tolerance_below = np.pi/4
        joint_constraint_02.weight = 1
        constraints.joint_constraints.append(joint_constraint_02)

        joint_constraint = JointConstraint()
        joint_constraint.joint_name = "elbow_joint"
        joint_constraint.position = -2.3562/2
        joint_constraint.tolerance_above = 2.3562/2
        joint_constraint.tolerance_below = 2.3562/2
        joint_constraint.weight = 1
        constraints.joint_constraints.append(joint_constraint)

        joint_constraint_03 = JointConstraint()
        joint_constraint_03.joint_name = "wrist_2_joint"
        joint_constraint_03.position = pi/2
        joint_constraint_03.tolerance_above = 0.5
        joint_constraint_03.tolerance_below = 0.5
        joint_constraint_03.weight = 1
        constraints.joint_constraints.append(joint_constraint_03)

        joint_constraint_05 = JointConstraint()
        joint_constraint_05.joint_name = "wrist_3_joint"
        joint_constraint_05.position = self._group.get_current_joint_values()[5]
        joint_constraint_05.tolerance_above = 3.14
        joint_constraint_05.tolerance_below = 3.14
        joint_constraint_05.weight = 1
        constraints.joint_constraints.append(joint_constraint_05)

        self.constraints = constraints
        self._group.set_path_constraints(constraints)

    def shutdown(self):
        r"""
        Shutdown the moveit_commander.
        """
        self._group.stop()
        moveit_commander.roscpp_shutdown()

    def display_traj(self, plan):
        r"""
        Display the trajectory.
        """
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self._robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self._display_trajectory_publisher.publish(display_trajectory)

    # Robot control basic actions

    def go_to_goal_joint(self, joint_goal, wait=True):
        r"""
        Move the robot to the specified joint angles.
        @param: joint_angle A list of floats
        @returns: bool True if successful by comparing the goal and actual joint angles
        """
        if not isinstance(joint_goal, list):
            rospy.logerr("Invalid joint angle")
            return False

        self._group.go(joint_goal, wait=wait)
        self._group.stop()

        cur_joint = self._group.get_current_joint_values()
        return all_close(joint_goal, cur_joint, 0.001)

    def go_to_pose_goal(self, pose: Pose, child_frame_id, parent_frame_id, wait=True):
        r"""
        #### Move the robot to the specified pose.

        Current expected behavior of this planner

        @param: pose A Pose instance
        @param: wait A bool to wait for the robot to reach the goal
        @returns: bool True if successful by comparing the goal and actual poses
        """

        if parent_frame_id != self._planning_frame:
            pose = self.get_transform_in_planning_frame(
                pose, child_frame_id, parent_frame_id)

        self.visualize_target_pose(pose)

        self._group.set_pose_target(pose)
        self._group.go(wait=True)
        self._group.stop()
        self._group.clear_pose_targets()

        cur_pose = self._group.get_current_pose().pose
        return all_close(pose, cur_pose, 0.01)

    def gen_carternian_path(self, target_pose: Pose, max_step=0.001, jump_thresh=0.0):
        r"""
        Generate a cartesian path as a straight line to a desired pose .
        @param: waypoints A list of Pose instances
        @returns: RobotTrajectory instance for the cartesian path
        """
        current_pose = self._group.get_current_pose().pose
        plan = None
        fraction = 0.0
        fix_itterations = 0

        # generate a straight line path using a scaling 0 to 1 applied to the target pose differ to current pose
        waypoints = []
        waypoints.append(current_pose)
        waypoints.append(target_pose)

        # Blocking loop to ensure the cartesian path is fully planned
        while fraction < 1.0:

            (plan, fraction) = self._group.compute_cartesian_path(
                waypoints, max_step, jump_thresh, avoid_collisions=True, path_constraints=self.constraints)

            fix_itterations += 1
            if fix_itterations > 50:  # Maxium fix itterations
                rospy.logerr("Failed to find a plan")
                return None, 0.0

            # check if the plan is valid with timestamp duplication
        path = []
        for i in range(len(plan.joint_trajectory.points)):
            if i == 0:
                continue

            time_diff = plan.joint_trajectory.points[i].time_from_start.secs - \
                plan.joint_trajectory.points[i-1].time_from_start.secs
            print(time_diff)
            if not plan.joint_trajectory.points[i].time_from_start.secs == plan.joint_trajectory.points[i].time_from_start.secs:
                path.append(plan.joint_trajectory.points[i])
                
        rospy.loginfo(
            f"Fraction planned: {fraction}; Fix itteration {fix_itterations}")
        return plan, fraction

    def execute_plan(self, plan: RobotTrajectory):
        r"""
        Execute the plan.
        @param: plan A RobotTrajectory instance
        @returns: bool True if successful by comparing the goal and actual poses
        """
        try:
            result = self._group.execute(plan, wait=True)
            print(result)
        except Exception as e:
            rospy.logerr(e)
            return False
        
        # self._group.execute(plan, wait=True)
        return all_close(plan.joint_trajectory.points[-1], self._group.get_current_pose().pose, 0.01)

    def stop(self):

        self._group.stop()

    # Delicted actions
    def go_to_target_pose_name(self, name):

        self._group.set_named_target(name)
        self._group.go(wait=True)

        self._group.plan()
        joint_goal = self._group.get_named_target_values(name)
        cur_joint = self._group.get_current_joint_values()
        return all_close(joint_goal, cur_joint, 0.001)

    def move_ee_along_axis(self, axis: str, delta: float) -> bool:
        r"""
        Move the end effector along the specified axis of the planning frame.
        @param: axis The axis to move along
        @param: delta The distance to move
        @returns: bool True if successful by comparing the goal and actual poses
        """

        goal = deepcopy(self._group.get_current_pose().pose)
        if axis == "x":
            goal.position.x += delta
        elif axis == "y":
            goal.position.y += delta
        elif axis == "z":
            goal.position.z += delta
        else:
            rospy.logerr("Invalid axis")
            return False

        plan, frac = self.gen_carternian_path(target_pose=goal)
        if plan is None:
            return False
        
        done = self.execute_plan(plan=plan)
        return done
    
    def rotate_ee(self, angle: float) -> bool:
        r"""
        Rotate the end effector around the z-axis of the planning frame."""
        goal = deepcopy(self._group.get_current_joint_values())
        goal[-1] += angle
        done = self.go_to_goal_joint(goal, wait=False)
        return done
        

    # Gripper control

    def open_gripper(self, force=None):
        self._gripper.open(force)
        rospy.sleep(1)

    def close_gripper(self, force=None):
        self._gripper.close(force)
        rospy.sleep(1)

    def open_gripper_to(self, width, force=None):
        self._gripper.open_to(width, force)
        rospy.sleep(1)

    # Getters

    def get_scene(self):
        return self._scene

    def get_current_pose(self):
        return self._group.get_current_pose().pose

    def get_current_rpy(self):
        return self._group.get_current_rpy()

    def get_current_joint_values(self):
        return self._group.get_current_joint_values()

    def get_joint_names(self):
        return self._group.get_joints()

    def get_end_effector_link(self):
        return self._group.get_end_effector_link()

    def get_transform_in_planning_frame(self, pose, child_frame_id: str, parent_frame_id: str, lookup_frame_id=None, to_SE3=False):
        r"""
        Get the transform between two frames in the tf tree.
        @param: pose A Pose instance
        @param: child_frame_id The child frame id
        @param: parent_frame_id The parent frame id
        @returns: Pose The pose in the planning frame"""

        transform_target = pose_to_transformstamped(
            pose=pose, child_frame_id=child_frame_id, parent_frame_id=parent_frame_id)

        self.tf_broadcaster.sendTransform(transform_target)

        rospy.sleep(0.01)

        if lookup_frame_id is None:
            lookup_frame_id = self._planning_frame

        tf_is_received = False
        while not tf_is_received:
            try:
                tf_received = self.tf_buffer.lookup_transform(
                    lookup_frame_id, child_frame_id, rospy.Time(0), rospy.Duration(1.0))
                tf_is_received = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue

        pose = transformstamped_to_pose(
            tf_received)

        return pose if not to_SE3 else pose_to_SE3(pose)

    # Visualization
    def visualize_target_pose(self, pose: Pose, type: int = 2, frame_id: str = "trolley"):
        r"""
        Visualize the target pose in rviz
        @param: pose The pose to be visualized
        @param: frame_id The frame id of the pose, default is the planning frame
        """

        target_marker = create_marker(frame_id, type, pose)
        self._marker_pub.publish(target_marker)

    # Callbacks
    def _joint_states_callback(self, data: JointState):
        r"""
        Callback function for the joint states subscriber.
        This data feeds directly to the virtual UR3 model as kinematic solver."""

        self.virtual_UR.q = data.position
