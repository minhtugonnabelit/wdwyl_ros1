#! /usr/bin/env python3

import rospy
import tf2_ros

# Importing planner module
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from ur3e_controller.UR3e import UR3e
from ur3e_controller.collision_manager import CollisionManager
from ur3e_controller.utility import *

# Importing perception module
from perception.localizer_model import RealSense
from perception.classifier_model import Classifier

import threading
from threading import Thread
from copy import deepcopy

# @TODO: Implementing loading scene with object detected
# @TODO: initializing a static tf from camera to toolpose via static broadcaster


BOX_01_POSE = list_to_pose(
    [-172.44 * 0.001, -324 * 0.001, 54.7 * 0.001, 0, 0, 0])
BOX_02_POSE = list_to_pose([-193.42 * 0.001, -400 *
                           0.001, 54.44 * 0.001, 0, 0, 0])
TEST_POSE_TARGET = list_to_pose([0.117, -0.104, 0.36, 0, 0, 10.65])

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
        self._ur3e = UR3e()
        self.scene = self._ur3e.get_scene()
        self.collisions = CollisionManager(self.scene)

        self.marker_pub = rospy.Publisher(
            "visualization_marker", Marker, queue_size=10)

        # Initialize the perception module
        self.rs = RealSense(on_UR=True)
        self.classifier = Classifier()

        # self.setup_scene()
        self._ur3e.move_to_hang()
        self._ur3e.open_gripper(force=400)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.bottle_pose = Pose()
        self.crate_pose = Pose()

        rospy.on_shutdown(self.cleanup)

    # def setup_scene(self):

    #     rospy.sleep(1)

    def box_pick(self):

        rospy.loginfo("box pick action")

        # # Move the robot to the localize pose
        # cone_pose = TEST_POSE_TARGET
        # _, cone_ID = self.collisions.add_cone_collision_object(pose=cone_pose,
        #                                                        object_type="cone",
        #                                                        frame_id="tool0",
        #                                                        size=(0.06, 0.06, 0.06))

        # target_marker = MissionPlanner.__create_marker(
        #     "tool0", 2, TEST_POSE_TARGET)
        # self.marker_pub.publish(target_marker)

        # goal_pose = PoseStamped()
        # goal_pose.pose = TEST_POSE_TARGET
        # goal_pose.header.frame_id = "tool0"
        # self._ur3e.go_to_pose_goal(goal_pose)
        # rospy.sleep(1)

        done = False
        on_going = False

        while not rospy.is_shutdown():

            if not done:

                self.set_transform_target(pose=TEST_POSE_TARGET,
                                          child_frame_id="crate_center",
                                          frame_id="tool0")

                try:
                    tf_received = self.tf_buffer.lookup_transform(
                        "base_link_inertia", "crate_center", rospy.Time(0), rospy.Duration(1.0))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    self.rate.sleep()
                    continue

                pose = get_PoseStamped_from_TransformStamped(
                    tf_received)

                self.visualize_target_pose(pose.pose)

                self._ur3e.go_to_pose_goal(pose)
                rospy.sleep(1)

                done = True

        # current_pose = self._ur3e.get_current_pose()
        # goal_pose = PoseStamped()
        # goal_pose.pose = current_pose
        # goal_pose.header.frame_id = "trolley"
        # goal_pose.pose.position.x -= 0.1
        # self._ur3e.go_to_pose_goal(goal_pose)
        # rospy.sleep(1)

        # plan, fraction = self._ur3e.gen_carternian_path(goal_pose)
        # self._ur3e.display_traj(plan)
        # rospy.sleep(1)

        # self._ur3e.execute_plan(plan)

        # self._ur3e.close_gripper(force=400)
        # rospy.sleep(3)

        # # self.collisions.attach_object(end_effector_link, box_ID)

        # target_pose_stamped = PoseStamped()
        # target_pose_stamped.pose = list_to_pose(309 * 0.001, -320 * 0.001,
        #                                         54.78 * 0.001, -3.14, 0, -3.140)
        # target_pose_stamped.header.frame_id = "base"
        # self._ur3e.go_to_pose_goal(target_pose_stamped)
        # rospy.sleep(1)

        # self._ur3e.open_gripper(force=400)
        # rospy.sleep(3)

        # # self.collisions.detach_object(link=end_effector_link, name=box_ID)
        # rospy.sleep(1)

        # self._ur3e.home()

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

            self._ur3e.move_to_hang()
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
            
            ### @TODO: OFFSET FOR THE CAMERA TO BE ON TOP OF CRATE CENTER TO BE MEASURED 
            self.crate_pose = get_PoseStamped_from_TransformStamped(
                tf_received)

            # Add mesh of the crate into planning scene for collision avoidance
            self.collisions.add_collision_object(
                obj_id="crate", pose=self.crate_pose, object_type="crate", frame_id="base_link_inertia")

            # Visualize the target pose
            self.visualize_target_pose(self.crate_pose.pose)

            # Command the robot to move to the target pose.  State reset at pose right on top of the crate center. The system have to ensure that the crate is not moved during the process
            self._ur3e.go_to_pose_goal(self.crate_pose)
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
            self._ur3e.go_to_pose_goal(bottle_pose_stamped)

            # Command the robot to move to the lower the gripper to the bottle
            goal = deepcopy(bottle_pose_stamped)
            goal.pose.position.z -= bottle_pose_raw[2]
            self.visualize_target_pose(goal.pose)
            self._ur3e.go_to_pose_goal(goal)

            self._ur3e.close_gripper(force=400)
            rospy.sleep(1)

            # Attach the bottle to the end effector of the robot in planning scene
            _, bottle_id = self.collisions.add_bottle(
                initial_pose=bottle_pose_stamped.pose)
            self.collisions.attach_object(
                eef_link=self._ur3e.get_end_effector_link(), obj_id=bottle_id)

            # Move the robot to the classify pose
            self._ur3e.go_to_pose_goal(CLASSIFY_POSE)

            ## =========================================================================== ##
            # =================STATE 1: CLASSIFY THE OBJECT AND DELIVER==================== #
            # ----------------------------------------------------------------------------- #
            #           Classify the object and deliver to the specified location           #
            ## =========================================================================== ##

            # Turn on classification tag
            self.classifier.set_classify_flag(True, wait=True)

            # Turn wrist 3 joint until bottle type classified
            cur_js = self._ur3e.get_current_joint_values()
            target_js = deepcopy(cur_js)
            target_js[-1] = cur_js[-1] + 1.57
            while self.classifier.bottle_class is None:    
                self._ur3e.go_to_goal_joint(target_js, wait=False)
            
            self._ur3e.stop() 

            bottle_class = deepcopy(self.classifier.bottle_class)
            self.classifier.set_classify_flag(False)      # stop classification task performing in RGB callback

            # Set type of bottle by changing the object name in planning scene and its color
            self.collisions.update_bottle_type(bottle_id, bottle_class)

            # Plan a collision free path to the deliver pose with dedicated bottle type and hang
            deliver_pose = BOTTLE_PLACEMENT[bottle_class]
            deliver_pose.position.z = self._ur3e.get_current_pose().position.z
            self._ur3e.go_to_pose_goal(deliver_pose)
            rospy.sleep(1)

            # Lower the robot to the deliver pose and drop the bottle
            self._ur3e.go_to_pose_goal(BOTTLE_PLACEMENT[bottle_class])
            self._ur3e.open_gripper(force=400)
            rospy.sleep(1)

            self.collisions.detach_object(
                eef_link=self._ur3e.get_end_effector_link(),
                obj_id=bottle_id
            )

            ## =========================================================================== ##
            # =================STATE 2: HANG THE ROBOT BACK TO LOCALIZING POSE============= #
            # ----------------------------------------------------------------------------- #
            #                                                                               #
            ## =========================================================================== ##

            self._ur3e.elevate_ee(delta_z=0)


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

        self._ur3e.move_to_hang()
        self._ur3e.shutdown()

        # remove all collision objects
        self.collisions.remove_collision_object()

        rospy.loginfo("Mission complete")


if __name__ == "__main__":
    mp = MissionPlanner()
    # mp.system_loop()
    mp.box_pick()
