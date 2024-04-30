#! /usr/bin/env python3
from dataclasses import dataclass

import rospy, tf
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
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
GRIPPER_OPEN = 500     #0.1mm
GRIPPER_CLOSE = 200     #0.1mm

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

def list_to_pose(pose: list):

    p = Pose()
    p.position.x = pose[0]
    p.position.y = pose[1]
    p.position.z = pose[2]

    ori_in_quat = tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5], axes='sxyz')
    p.orientation.x = ori_in_quat[0]
    p.orientation.y = ori_in_quat[1]
    p.orientation.z = ori_in_quat[2]
    p.orientation.w = ori_in_quat[3]

    return p
    
def transformstamped_to_posestamped(ts: TransformStamped) -> PoseStamped:
    r"""
    Convert a TransformStamped to a PoseStamped instance

    @param: ts The TransformStamped to be converted
    @returns: PoseStamped A PoseStamped instance

    """
    ps = PoseStamped()
    ps.header.stamp = ts.header.stamp
    ps.header.frame_id = ts.header.frame_id
    ps.pose.position.x = ts.transform.translation.x
    ps.pose.position.y = ts.transform.translation.y
    ps.pose.position.z = ts.transform.translation.z
    ps.pose.orientation = ts.transform.rotation

    return ps

def transformstamped_to_pose(ts: TransformStamped) -> Pose:
    r"""
    Convert a TransformStamped to a Pose instance

    @param: ts The TransformStamped to be converted
    @returns: Pose A Pose instance

    """
    p = Pose()
    p.position.x = ts.transform.translation.x
    p.position.y = ts.transform.translation.y
    p.position.z = ts.transform.translation.z
    p.orientation = ts.transform.rotation

    return p

def pose_to_transformstamped(pose: Pose, frame_id: str, child_frame_id: str) -> TransformStamped:
    r"""
    Convert a pose to a TransformStamped instance

    @param: pose The pose to be converted
    @param: frame_id The frame id of the pose
    @param: child_frame_id The child frame id of the pose
    @returns: TransformStamped A TransformStamped instance

    """
    ts = TransformStamped()
    ts.header.stamp = rospy.Time.now()
    ts.header.frame_id = frame_id
    ts.child_frame_id = child_frame_id
    ts.transform.translation.x = pose.position.x
    ts.transform.translation.y = pose.position.y
    ts.transform.translation.z = pose.position.z
    ts.transform.rotation = pose.orientation

    return ts

def create_marker(frame: str, type: int, pose: Pose, scale=[0.01, 0.01, 0.01], color=[0, 1, 0, 1]):
    r"""
    Create a marker for visualization

    @param: frame The frame id of the marker
    @param: type The type of the marker, 0: Arrow, 1: Cube, 2: Sphere, 3: Cylinder, 4: Line Strip, 5: Line List, 6: Cube List, 7: Sphere List, 8: Points, 9: Text
    @param: pose The pose of the marker
    @param: scale The scale of the marker
    @param: color The color of the marker
    @returns: Marker A marker instance

    """
    marker = Marker()

    marker.header.frame_id = frame
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker.type = type
    marker.id = 0

    # Set the pose of the marker
    marker.pose = pose

    # Set the scale of the marker
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]

    # Set the color
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    return marker

def list_to_PoseStamped(pose : list, frame_id : str = "base_link_inertia"):

    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.pose.position.x = pose[0]
    ps.pose.position.y = pose[1]
    ps.pose.position.z = pose[2]

    ori_in_quat = tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5], axes='sxyz')
    ps.pose.orientation.x = ori_in_quat[0]
    ps.pose.orientation.y = ori_in_quat[1]
    ps.pose.orientation.z = ori_in_quat[2]
    ps.pose.orientation.w = ori_in_quat[3]

    return ps