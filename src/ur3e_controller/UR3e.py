#! /usr/bin/env python3

import sys

import tf
import rospy
import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
import moveit_msgs.msg
import geometry_msgs.msg

from ur3e_controller.utility import *
from ur3e_controller.gripper import Gripper
from ur3e_controller.collision_manager import CollisionManager


class UR3e:

    r"""
    A class to control the UR3e robot arm using moveit_commander.

    """

    def __init__(self) -> None:

        moveit_commander.roscpp_initialize(sys.argv)
        self._robot = RobotCommander()
        self._scene = PlanningSceneInterface()
        self._group = MoveGroupCommander("manipulator")
        self._gripper = Gripper()
        self._object_manager = CollisionManager(self._scene)

        self._planning_frame = self._group.get_planning_frame()
        self._eef_link = self._group.get_end_effector_link()
        self._group_names = self._robot.get_group_names()
        self._cur_js = self._group.get_current_joint_values()

        setup_completed = self.movegroup_setup()

        rospy.logdebug(f"============ Planning frame: {self._planning_frame}")
        rospy.logdebug(f"============ End effector link: {self._eef_link}")
        rospy.logdebug(f"============ Robot Groups: {self._group_names}")
        rospy.logdebug(f"============ Initialized UR3e state: {self._cur_js}")


    def movegroup_setup(self):
        r"""
        Setup the move group.
        """
        self._group.set_start_state_to_current_state()
        self._group.set_planner_id("RRTConnectkConfigDefault")
        self._group.set_planning_time(10)
        self._group.set_num_planning_attempts(5)
        self._group.set_goal_position_tolerance(POS_TOL)
        self._group.set_goal_orientation_tolerance(ORI_TOL)
        self._group.set_max_velocity_scaling_factor(MAX_VEL_SCALE_FACTOR)
        self._group.set_max_acceleration_scaling_factor(MAX_ACC_SCALE_FACTOR)
        
        return True


    def shutdown(self):
        r"""
        Shutdown the moveit_commander.
        """
        moveit_commander.roscpp_shutdown()


    # Robot control basic actions

    def go_to_goal_joint(self, joint_angle):
        r"""
        Move the robot to the specified joint angles.
        @param: joint_angle A list of floats
        @returns: bool True if successful by comparing the goal and actual joint angles
        """

        if not isinstance(joint_angle, list):
            rospy.logerr("Invalid joint angle")
            return False

        joint_goal = self._group.get_current_joint_values()
        joint_goal = joint_angle
        self._group.go(joint_angle, wait=True)
        self._group.stop()

        cur_joint = self._group.get_current_joint_values()
        return all_close(joint_goal, cur_joint, 0.01)
    

    def go_to_pose_goal(self, pose_goal):
        r"""
        Move the robot to the specified pose.
        @param: pose_goal A PoseStamped instance
        @returns: bool True if successful by comparing the goal and actual poses
        """

        if not isinstance(pose_goal, geometry_msgs.msg.PoseStamped):
            rospy.logerr("Invalid pose goal")
            return False

        self._group.set_pose_target(pose_goal)
        self._group.go(wait=True)
        self._group.stop()
        self._group.clear_pose_targets()

        cur_pose = self._group.get_current_pose().pose
        return all_close(pose_goal, cur_pose, 0.01)
    

    def smoothing_path(self, cart_traj: list, resolution=0.01, jump_thresh=0.0):
        r"""
        Smooth the path using the path constraints.
        @param: plan A RobotTrajectory instance
        @returns: RobotTrajectory instance for smoothed path with addtional interpolation points
        """
        (interp_traj, fraction) = self._group.compute_cartesian_path(
            cart_traj, resolution, jump_thresh,)
        return interp_traj, fraction


    def execute_plan(self, plan):
        r"""
        Execute the plan.
        @param: plan A RobotTrajectory instance
        @returns: bool True if successful by comparing the goal and actual poses
        """
        self._group.execute(plan, wait=True)
        self._group.stop()
        return True


    # Delicted actions

    def home(self):
        r"""
        Move the robot to the home position.
        """
        self.go_to_goal_joint([0, -pi/2, pi/2, 0, pi/2, 0])


    # Gripper control

    def open_gripper(self, force=None):
        self._gripper.open(force)

    def close_gripper(self, force=None):
        self._gripper.close(force)

    def open_gripper_to(self, width, force=None):
        self._gripper.open_to(width, force)

    # Getters

    def get_current_pose(self):
        return self._group.get_current_pose().pose

    def get_current_rpy(self):
        return self._group.get_current_rpy()

    def get_current_joint_values(self):
        return self._group.get_current_joint_values()

    def get_joint_names(self):
        return self._group.get_joints()
