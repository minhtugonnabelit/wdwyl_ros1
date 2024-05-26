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
from perception.YOLO.utility import *
from perception.YOLO.localizer_model import RealSense
from perception.YOLO.classifier_model import Classifier

from copy import deepcopy

# @TODO: Implementing loading scene with object detected
# @TODO: initializing a static tf from camera to toolpose via static broadcaster


HEIGHT_TO_DETECT_BOTTLE = 0.28
ENDPOINT_OFFSET = 0.2
HANG_OFFSET = 0.05
CAM_OFFSET = list_to_pose([TX,
                           TY,
                           ENDPOINT_OFFSET,
                           0, 0, 0])

AVAILABLE_MESHES = ["coke", "pepsi", "sprite", "fanta"]
BOTTLE_PLACEMENT = {
    "coke": list_to_pose([0, 0, 0, 0, 0, 0]),
    "pale_ales": list_to_pose([0, 0, 0, 0, 0, 0]),
    "heniken": list_to_pose([0, 0, 0, 0, 0, 0]),
}

CONTROL_RATE = 10  # Hz


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
        self.rs = RealSense()
        self.classifier = Classifier()

        # Setup the scene with ur3e controller and homing
        self.setup_scene()
        self.ur3e.go_to_target_pose_name(UR3e.DETECT_CONFIG)
        # Fully open the gripper to get a better view
        self.ur3e.open_gripper_to(width=1100, force=400)

        # TF2 listener and broadcaster to deal with the transformation
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

                # Fully open the gripper to 1100 0.1mm wide to maximise camera view
                self.ur3e.open_gripper_to(width=1100, force=400)
                done = True

    def system_loop(self):
        r"""
        The main system loop for the mission planner
        """

        rospy.loginfo("Starting system loop")
        done = False

        while not rospy.is_shutdown():

            ## =========================================================================== ##
            # =================STATE PRE-0: MOVE TO THE CRATE CENTER======================= #
            # ----------------------------------------------------------------------------- #
            # Move to the crate center and wait for the crate pose, then save the crate     #
            # pose for further use as reset state after each bottle sorted                  #
            ## =========================================================================== ##

            ## =========================================================================== ##
            # =================STATE 0: LOCALIZE THE BOTTLE TO BE PICKED=================== #
            # ----------------------------------------------------------------------------- #
            # Localize the bottle to be picked and move to pick the bottle                  #
            ## =========================================================================== ##

            if not done:

                self.rs.set_Bottle_Flag(True)
                self.bottle_pose = self.rs.get_bottle_pos()

                while self.bottle_pose is None:   # Wait until the bottle is detected

                    self.bottle_pose = self.rs.get_bottle_pos()
                    if not self.rs.get_circle_flag():
                        self.bottle_pose = None
                        rospy.loginfo("No circle detected") 

                # if self.rs.get_num_of_bottle() == 0:

                #     rospy.loginfo("All bottles have been sorted!")
                #     rospy.signal_shutdown("Finish sorting mission!")

                if self.bottle_pose[-1] is None:
                    self.bottle_pose[-1] = 0.0

                # rospy.loginfo(f"Bottle pose: {np.round(self.bottle_pose, 4)}")
                rospy.loginfo(f"Bottle pose: {self.bottle_pose}")


                # Turn off bottle flag to stop the bottle detection
                self.rs.set_Bottle_Flag(False)

                # Close the gripper to 500 0.1mm wide to fit in the tiny gripping state
                self.ur3e.open_gripper_to(width=500, force=200)
                rospy.sleep(1)

                # Re-align end effector to the crate center in plane xy only.
                pose = list_to_pose(
                    [self.bottle_pose[0],
                     self.bottle_pose[1] + 0.005,
                     self.bottle_pose[2] - HANG_OFFSET,
                     0, 0, 0]
                )
                self.ur3e.go_to_pose_goal(pose=pose,
                                          child_frame_id="bottle_center",
                                          parent_frame_id="tool0")

                # Lower the robot to the bottle neck
                rospy.sleep(1)
                self.ur3e.move_ee_along_axis(axis="z", delta=-0.07)

                # Close the gripper to 180 0.1mm wide to grip the bottle
                rospy.sleep(1)
                self.ur3e.open_gripper_to(width=180, force=200)

                # Add mesh of the bottle into planning scene for collision avoidance
                # self.collisions.add_collision_object(
                #     obj_id="bottle", pose=bottle_pose_raw, object_type="bottle", frame_id="base_link_inertia")

                # # Attach the bottle to the end effector of the robot in planning scene
                # _, bottle_id = self.collisions.add_bottle(
                #     initial_pose=bottle_pose_stamped.pose)
                # self.collisions.attach_object(
                #     eef_link=self.ur3e.get_end_effector_link(), obj_id=bottle_id)

                # Move to the classify pose
                rospy.sleep(1)
                self.ur3e.move_ee_along_axis(axis="z", delta=0.14)

                rospy.sleep(1)
                self.ur3e.go_to_target_pose_name(UR3e.DETECT_CONFIG)

                done = True

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

            # self.collisions.detach_object(
            #     eef_link=self.ur3e.get_end_effector_link(),
            #     obj_id=bottle_id
            # )

            # ## =========================================================================== ##
            # # =================STATE 2: HANG THE ROBOT BACK TO LOCALIZING POSE============= #
            # # ----------------------------------------------------------------------------- #
            # #                                                                               #
            # ## =========================================================================== ##

            # Move the robot to the home position

            # Reset the planning scene

            rospy.loginfo("System loop running")    
            self.rate.sleep()


    def cleanup(self):

        rospy.loginfo("Cleaning up")

        self.FINISHED = True

        # self.ur3e.go_to_target_pose_name(UR3e.DETECT_CONFIG)
        # self.ur3e.open_gripper_to(width=1100, force=400)
        self.ur3e.shutdown()

        # remove all collision objects
        self.collisions.remove_collision_object()

        rospy.loginfo("Clean-up completed")


if __name__ == "__main__":
    mp = MissionPlanner()
    mp.system_loop()
