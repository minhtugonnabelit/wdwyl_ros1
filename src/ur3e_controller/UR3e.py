#! /usr/bin/env python3

import sys

import tf
import rospy
import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
import moveit_msgs.msg
from moveit_msgs.msg import RobotTrajectory
import geometry_msgs.msg

from ur3e_controller.utility import *
from ur3e_controller.gripper import Gripper
from ur3e_controller.collision_manager import CollisionManager

from copy import deepcopy


class UR3e:

    r"""
    A class to control the UR3e robot arm using moveit_commander.

    """

    def __init__(self) -> None:

        moveit_commander.roscpp_initialize(sys.argv)
        self._robot = RobotCommander()
        self._scene = PlanningSceneInterface()
        self._group = MoveGroupCommander("ur3e")
        self._gripper = Gripper()

        self._planning_frame = self._group.get_planning_frame()
        self._group_names = self._robot.get_group_names()
        self._eef_link = self._group.get_end_effector_link()
        self._cur_js = self._group.get_current_joint_values()

        self._display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)

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
        self._group.set_goal_orientation_tolerance(ORI_TOL)
        self._group.set_max_velocity_scaling_factor(MAX_VEL_SCALE_FACTOR)
        self._group.set_max_acceleration_scaling_factor(MAX_ACC_SCALE_FACTOR)
        self._group.set_trajectory_constraints
        self._group.set_pose_targets

        # Add a small fragment piece as gripper cable node
        cable_cap_pose = PoseStamped()
        cable_cap_pose.pose = list_to_pose([-0.045, -0.01, 0.01, 1.57, 0, 1.57])
        cable_cap_pose.header.frame_id = "tool0"
        self._scene.add_cylinder("cable_cap", cable_cap_pose, 0.02, 0.01)
        self._scene.attach_mesh("tool0", "cable_cap", touch_links=[
                               "onrobot_rg2_base_link", "wrists_3_link"])

        return True


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
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
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
        return all_close(pose_goal.pose, cur_pose, 0.01)

    def gen_carternian_path(self, target_pose: Pose, max_step=0.001, jump_thresh=0.0):
        r"""
        Generate a cartesian path as a straight line to a desired pose .
        @param: waypoints A list of Pose instances
        @returns: RobotTrajectory instance for the cartesian path
        """
        current_pose = self._group.get_current_pose().pose

        # generate a straight line path using a scaling 0 to 1 applied to the target pose differ to current pose
        waypoints = []

        for i in range(1, 11):
            scale = i / 10.0

            pt = Pose()
            pt.position.x = current_pose.position.x + scale * \
                (target_pose.position.x - current_pose.position.x)
            pt.position.y = current_pose.position.y + scale * \
                (target_pose.position.y - current_pose.position.y)
            pt.position.z = current_pose.position.z + scale * \
                (target_pose.position.z - current_pose.position.z)
            pt.orientation = target_pose.orientation

            waypoints.append(pt)


        (plan, fraction) = self._group.compute_cartesian_path(
            waypoints, max_step, jump_thresh, avoid_collisions=True)
        return plan, fraction   

    def execute_plan(self, plan : RobotTrajectory):
        r"""
        Execute the plan.
        @param: plan A RobotTrajectory instance
        @returns: bool True if successful by comparing the goal and actual poses
        """
        self._group.execute(plan, wait=True)
        self._group.stop()
        return all_close(plan.joint_trajectory.points[-1], self._group.get_current_pose().pose, 0.01)
    
    def stop(self):

        self._group.stop()

    # Delicted actions

    def home(self):
        r"""
        Move the robot to the home position.
        """

        self._group.set_named_target("home")
        self._group.go(wait=True)
        joint_goal = self._group.get_named_target_values("home")

        cur_joint = self._group.get_current_joint_values()
        return all_close(joint_goal, cur_joint, 0.01)

    def move_to_hang(self):
        r"""
        Move the robot to the hang position.
        """

        self._group.set_named_target("hanger_right_sided")
        self._group.go(wait=True)
        joint_goal = self._group.get_named_target_values("hanger_right_sided")

        cur_joint = self._group.get_current_joint_values()
        return all_close(joint_goal, cur_joint, 0.01)
    
    def elevate_ee(self, delta_z : float) -> bool:
        r"""
        """

        goal = deepcopy(self._group.get_current_pose().pose)
        goal.position.z += delta_z

        plan, _ = self.gen_carternian_path(target_pose=goal)
        done = self.execute_plan(plan=plan)
        
        return done
        

    # Gripper control

    def open_gripper(self, force=None):
        self._gripper.open(force)

    def close_gripper(self, force=None):
        self._gripper.close(force)

    def open_gripper_to(self, width, force=None):
        self._gripper.open_to(width, force)


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
