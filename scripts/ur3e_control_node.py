#! /usr/bin/env python3

import numpy as np

import rospy
import tf2_ros

# Importing planner module
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from ur3e_controller.UR3e import UR3e
from ur3e_controller.collision_manager import CollisionManager
from ur3e_controller.utility import *

# Importing perception module
from perception.utility import *
from perception.localizer_model import RealSense
from perception.classifier_model import Classifier

from copy import deepcopy

# @TODO: Implementing loading scene with object detected
# @TODO: initializing a static tf from camera to toolpose via static broadcaster


TARGET_CRATE_POSE = list_to_pose([0.033 + TX,
                                 -0.090 + TY*1.3,
                                 0.5052 - 0.28,
                                 0, 0, np.deg2rad(1.31)])

TARGET_BOTTLE_POSE = list_to_pose([-0.025,
                                   -0.116,
                                   0.525 - 0.28,
                                   0, 0, np.deg2rad(0)])


AVAILABLE_MESHES = ["coke", "pepsi", "sprite", "fanta"]
BOTTLE_PLACEMENT = {
    "coke": list_to_pose([0, 0, 0, 0, 0, 0]),
    "pale_ales": list_to_pose([0, 0, 0, 0, 0, 0]),
    "heniken": list_to_pose([0, 0, 0, 0, 0, 0]),
}


class MissionPlanner:

    FINISHED = False

    def __init__(self) -> None:

        rospy.init_node("What_drink_would_you_like",
                        log_level=1, anonymous=True)
        rospy.loginfo("Initializing MissionPlanner")
        self.rate = rospy.Rate(CONTROL_RATE)

        # Initialize the UR3e controller
        self.ur3e = UR3e()
        self.scene = self.ur3e.get_scene()
        self.collisions = CollisionManager(self.scene)

        self.marker_pub = rospy.Publisher(
            "visualization_marker", Marker, queue_size=10)

        # Initialize the perception module
        self.rs = RealSense(on_UR=True)
        self.classifier = Classifier()

        self.setup_scene()
        self.ur3e.move_to_hang()
        self.ur3e.open_gripper_to(width=1100, force=400)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.bottle_pose = Pose()
        self.crate_pose = Pose()

        rospy.on_shutdown(self.cleanup)

    def setup_scene(self):

        rospy.sleep(1)

    def box_pick(self):

        rospy.loginfo("box pick action")

        done = False
        on_going = False

        while not rospy.is_shutdown():

            if not done:

                self.ur3e.go_to_pose(pose = TARGET_CRATE_POSE,
                                     child_frame_id="crate_center",
                                     frame_id="tool0")

                # # Move to crate
                # self.ur3e.set_transform_target(pose=TARGET_CRATE_POSE,
                #                                child_frame_id="crate_center",
                #                                frame_id="tool0")

                # self.set_transform_target(pose=TARGET_CRATE_POSE,
                #                           child_frame_id="crate_center",
                #                           frame_id="tool0")

                # try:
                #     tf_received = self.tf_buffer.lookup_transform(
                #         "base_link_inertia", "crate_center", rospy.Time(0), rospy.Duration(1.0))
                # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                #     self.rate.sleep()
                #     continue

                # pose = transformstamped_to_pose(
                #     tf_received)

                # self.visualize_target_pose(pose)

                # self.ur3e.go_to_pose_goal(pose)
                # rospy.sleep(2)

                self.ur3e.go_to_pose(pose = TARGET_BOTTLE_POSE,
                                     child_frame_id="bottle",
                                     frame_id="tool0")
                                

                # # Move to above bottle location and pick
                # self.set_transform_target(pose=TARGET_BOTTLE_POSE,
                #                           child_frame_id="bottle",
                #                           frame_id="tool0")

                # try:
                #     tf_received = self.tf_buffer.lookup_transform(
                #         "base_link_inertia", "bottle", rospy.Time(0), rospy.Duration(1.0))
                # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                #     self.rate.sleep()
                #     continue

                # pose = transformstamped_to_pose(
                #     tf_received)

                # self.visualize_target_pose(pose)
                # self.ur3e.go_to_pose_goal(pose)
                # rospy.sleep(2)

                # target_pose = self.ur3e.get_current_pose()
                # target_pose.position.z -= 0.3
                # self.visualize_target_pose(pose.pose)
                # self.ur3e.go_to_pose_goal(target_pose)
                # rospy.sleep(2)

                done = True

    def system_loop(self):
        r"""
        The main system loop for the mission planner
        """

        rospy.loginfo("Starting system loop")

        while not rospy.is_shutdown():

            ## =========================================================================== ##
            # =================STATE PRE-0: MOVE TO THE CRATE CENTER======================= #
            # ----------------------------------------------------------------------------- #
            # Move to the crate center and wait for the crate pose, then save the crate     #
            # pose for further use as reset state after each bottle sorted                  #
            ## =========================================================================== ##

            self.ur3e.move_to_hang()
            self.rs.set_Crate_Flag(True, wait=True)
            pose = self.rs.get_crate_pose()
            self.rs.set_Crate_Flag(False)

            # Re-align end effector to the crate center in plane xy only.
            pose = list_to_pose(
                [self.crate_pose[0],     # x
                    self.crate_pose[1],     # y
                    0,                      # z
                    0,                      # roll
                    0,                      # pitch
                    self.crate_pose[3]]      # yaw
            )

            # Start broadcasting the crate center
            self.set_transform_target(pose=pose,
                                      child_frame_id="crate_center",
                                      frame_id="tool0")

            # Move to crate center in plane xy
            try:
                tf_received = self.tf_buffer.lookup_transform(
                    "base_link_inertia", "crate_center", rospy.Time(0), rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.rate.sleep()
                continue            # If the transform is not received, wait for the next loop
# TARGET_BOTTLE_PICK_POSE = list_to_pose([0.0246,
#                                         -0.0532,
#                                         0.525 - 0.28,
#                                         0, 0, np.deg2rad(-8.2)])
            # Add mesh of the crate into planning scene for collision avoidance
            self.collisions.add_collision_object(
                obj_id="crate", pose=self.crate_pose, object_type="crate", frame_id="base_link_inertia")

            # Visualize the target pose
            self.visualize_target_pose(self.crate_pose.pose)

            # Command the robot to move to the target pose.  State reset at pose right on top of the crate center. The system have to ensure that the crate is not moved during the process
            self.ur3e.go_to_pose_goal(self.crate_pose)
            rospy.sleep(1)

            ## =========================================================================== ##
            # =================STATE 0: LOCALIZE THE BOTTLE TO BE PICKED=================== #
            # ----------------------------------------------------------------------------- #
            # Localize the bottle to be picked and move to pick the bottle                  #
            ## =========================================================================== ##

            self.rs.set_Bottle_Flag(True, wait=True)

            if self.rs.bottle_num == 0:

                rospy.loginfo("All bottles have been sorted!")
                rospy.signal_shutdown("Finish sorting mission!")

            bottle_pose_raw = self.rs.get_bottle_pose()
            self.rs.set_Bottle_Flag(False)

            # Re-align end effector to the bottle in plane xy only.
            pose = list_to_pose(
                [bottle_pose_raw[0],
                 bottle_pose_raw[1],
                 0,
                 0,
                 0,
                 bottle_pose_raw[3]]
            )

            # Start broadcasting the bottle center
            self.set_transform_target(pose=pose,
                                      child_frame_id="bottle_center",
                                      frame_id="tool0")

            # Move to bottle center in plane xy
            try:

                tf_received = self.tf_buffer.lookup_transform(
                    "base_link_inertia", "bottle_center", rospy.Time(0), rospy.Duration(1.0))

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.rate.sleep()
                continue

            bottle_pose_stamped = get_PoseStamped_from_TransformStamped(
                tf_received)

            self.visualize_target_pose(bottle_pose_stamped.pose)
            self.ur3e.go_to_pose_goal(bottle_pose_stamped)

            # Command the robot to move to the lower the gripper to the bottle
            goal = deepcopy(bottle_pose_stamped)
            goal.pose.position.z -= bottle_pose_raw[2]
            self.visualize_target_pose(goal.pose)
            self.ur3e.go_to_pose_goal(goal)

            self.ur3e.close_gripper(force=400)
            rospy.sleep(1)

            # Attach the bottle to the end effector of the robot in planning scene
            _, bottle_id = self.collisions.add_bottle(
                initial_pose=bottle_pose_stamped.pose)
            self.collisions.attach_object(
                eef_link=self.ur3e.get_end_effector_link(), obj_id=bottle_id)

            # Move the robot to the classify pose
            self.ur3e.go_to_pose_goal(CLASSIFY_POSE)

            ## =========================================================================== ##
            # =================STATE 1: CLASSIFY THE OBJECT AND DELIVER==================== #
            # ----------------------------------------------------------------------------- #
            #           Classify the object and deliver to the specified location           #
            ## =========================================================================== ##

            # # Turn on classification tag
            # self.classifier.set_classify_flag(True, wait=True)

            # # Turn wrist 3 joint until bottle type classified
            # cur_js = self.ur3e.get_current_joint_values()
            # target_js = deepcopy(cur_js)
            # target_js[-1] = cur_js[-1] + 1.57
            # while self.classifier.bottle_class is None:
            #     self.ur3e.go_to_goal_joint(target_js, wait=False)

            # self.ur3e.stop()

            # bottle_class = deepcopy(self.classifier.bottle_class)
            # self.classifier.set_classify_flag(False)      # stop classification task performing in RGB callback

            # # Set type of bottle by changing the object name in planning scene and its color
            # self.collisions.update_bottle_type(bottle_id, bottle_class)

            # # Plan a collision free path to the deliver pose with dedicated bottle type and hang
            # deliver_pose = BOTTLE_PLACEMENT[bottle_class]
            # deliver_pose.position.z = self.ur3e.get_current_pose().position.z
            # self.ur3e.go_to_pose_goal(deliver_pose)
            # rospy.sleep(1)

            # # Lower the robot to the deliver pose and drop the bottle
            # self.ur3e.go_to_pose_goal(BOTTLE_PLACEMENT[bottle_class])
            # self.ur3e.open_gripper(force=400)
            # rospy.sleep(1)

            # self.collisions.detach_object(
            #     eef_link=self.ur3e.get_end_effector_link(),
            #     obj_id=bottle_id
            # )

            # ## =========================================================================== ##
            # # =================STATE 2: HANG THE ROBOT BACK TO LOCALIZING POSE============= #
            # # ----------------------------------------------------------------------------- #
            # #                                                                               #
            # ## =========================================================================== ##

            # self.ur3e.elevate_ee(delta_z=0)

            # Move the robot to the home position

            # Reset the planning scene

            # rospy.loginfo("System loop running")

            # if done:
            #     rospy.signal_shutdown("Mission complete")

            rospy.sleep(1)

    def set_transform_target(self, pose: Pose, child_frame_id: str, frame_id: str = "base_link_inertia",):

        transform_target = get_TransformStamped_from_pose(pose=pose,
                                                          frame_id=frame_id,
                                                          child_frame_id=child_frame_id)

        self.tf_broadcaster.sendTransform(transform_target)

        rospy.sleep(0.01)

    def visualize_target_pose(self, pose: Pose, type: int = 2, frame_id: str = "base_link_inertia"):
        r"""
        Visualize the target pose in rviz
        @param: pose The pose to be visualized
        @param: frame_id The frame id of the pose, default is base_link_inertia"
        """

        target_marker = create_marker(
            frame_id, type, pose)
        self.marker_pub.publish(target_marker)

    def cleanup(self):

        rospy.loginfo("Cleaning up")

        self.FINISHED = True

        self.ur3e.move_to_hang()
        rospy.sleep(1)

        self.ur3e.shutdown()

        # remove all collision objects
        self.collisions.remove_collision_object()

        rospy.loginfo("Mission complete")


if __name__ == "__main__":
    mp = MissionPlanner()
    # mp.system_loop()
    mp.box_pick()
