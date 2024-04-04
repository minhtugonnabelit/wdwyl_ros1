#! /usr/bin/env python3

from geometry_msgs.msg import Pose, PoseStamped
from moveit_commander.conversions import pose_to_list

from math import pi, tau, dist, fabs, cos

# Constants variables
CONTROL_RATE = 10
POS_TOL = 0.01
ORI_TOL = 0.01
MAX_VEL_SCALE_FACTOR = 0.1
MAX_ACC_SCALE_FACTOR = 0.1
INITIAL_CONFIG = [0, -pi/2, pi/2, 0, 0, 0]
CLASSIFY_POSE = Pose(position=(0.5, 0.5, 0.5), orientation=(0, 0, 0, 1))


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
