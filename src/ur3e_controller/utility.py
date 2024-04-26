#! /usr/bin/env python3
from dataclasses import dataclass

import rospy, tf
from geometry_msgs.msg import Pose, PoseStamped
from moveit_commander.conversions import pose_to_list
from moveit_commander import PlannerInterfaceDescription, MoveGroupCommander, PlanningSceneInterface

from math import pi, tau, dist, fabs, cos

import tf.transformations

# Constants variables
CONTROL_RATE = 10 #Hz
POS_TOL = 0.01  #m
ORI_TOL = 0.01  #m
MAX_VEL_SCALE_FACTOR = 0.4
MAX_ACC_SCALE_FACTOR = 0.4
INITIAL_CONFIG = [0, -pi/2, pi/2, 0, 0, 0]  #rad
LOCALIZE_POSE = Pose(position=(0.5, 0.5, 0.5), orientation=(0, 0, 0, 1))    #m
CLASSIFY_POSE = Pose(position=(0.5, 0.5, 0.5), orientation=(0, 0, 0, 1))    #m
GRIPPER_OPEN = 1000     #0.1mm
GRIPPER_CLOSE = 600     #0.1mm


@dataclass
class Bottle:
    r"""
    A class to represent a bottle object.
    """
    id : str
    type : str
    pose : Pose


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True

def get_pose(x, y, z, roll, pitch, yaw):

    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    ori_in_quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw, axes='sxyz')
    pose.orientation.x = ori_in_quat[0]
    pose.orientation.y = ori_in_quat[1]
    pose.orientation.z = ori_in_quat[2]
    pose.orientation.w = ori_in_quat[3]

    return pose

