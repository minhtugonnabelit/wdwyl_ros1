#! /usr/bin/env python3

import numpy as np
from threading import Thread

import rospy
import tf2_ros
from stream_data_on_web.srv import CurrentState, CurrentStateRequest

# Importing planner module
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from ur3e_controller.UR3e import UR3e
from ur3e_controller.collision_manager import CollisionManager
from ur3e_controller.utility import *

# Importing perception module
from perception.detection.localizer_model import RealSense
from perception.detection.utility import *
from perception.classification.classification_real_time import Classification
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

BOTTLE_PLACEMENT = {
    "heineken": np.deg2rad([28, -124, -55, -91, 90, 28]).tolist(),
    'crown': np.deg2rad([19, -121, -59, -90, 90, 19]).tolist(),
    'greatnorthern': np.deg2rad([8, -123, -56, -91, 90, 8]).tolist(),
    'paleale': np.deg2rad([-2, -128, -47, -94, 90, -2]).tolist()

}

CONTROL_RATE = 10  # Hz


class MissionPlanner:

    FINISHED = False

    def __init__(self) -> None:

        rospy.init_node("What_drink_would_you_like",
                        log_level=2, anonymous=True)
        rospy.loginfo("Initializing MissionPlanner")
        self.rate = rospy.Rate(CONTROL_RATE)

        # Initialize the UR3e controller
        self.ur3e = UR3e()
        self.collisions = CollisionManager(self.ur3e.get_scene())

        self.marker_pub = rospy.Publisher(
            "visualization_marker", Marker, queue_size=10)

        # Initialize the perception module
        self.rs = RealSense()
        self.classifier = Classification()
        self.classifier.set_Classify_Flag(False)

        # Setup the scene with ur3e controller and homing
        self.setup_scene()
        self.ur3e.go_to_target_pose_name(UR3e.DETECT_CONFIG)

        # TF2 listener and broadcaster to deal with the transformation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.bottle_pose = Pose()
        self.crate_pose = Pose()
        self._success = True
        self._bound_id = 'bound'

        self._safety_thread = Thread(target=self.safety_check)
        self._safety_thread.start()

        rospy.on_shutdown(self.cleanup)

    def setup_scene(self):

        rospy.sleep(1)

    def system_loop(self):
        r"""
        The main system loop for the mission planner
        """

        rospy.loginfo("Starting system loop")
        done = False

        while not rospy.is_shutdown():

            # check if the crate bound box is still in the scene
            self.collisions.remove_crate_bound([self._bound_id])

            # check if there still bottle in the crate
            self.rs.set_Bottle_Flag(True)
            rospy.sleep(3)
            if self.rs.get_num_of_bottle() == 0:
                rospy.loginfo("No more bottle in the crate")
                self.rs.set_Bottle_Flag(False)
                rospy.signal_shutdown("No more bottle in the crate")
            else:
                rospy.loginfo(f'{self.rs.get_num_of_bottle()} detected!')

            # call service to display number of bottle

            # this condition is used to limit the number of mission to one bottle only
            if not done:

                # looking for crate pose
                # self.rs.set_Crate_Flag(True)
                # rospy.sleep(1)
                # self.crate_pose = self.rs.get_crate_pos()
                # rospy.loginfo(f"Crate pose: {self.crate_pose}")
                # self.rs.set_Crate_Flag(False)
                crate_pose = self.look_for_crate()

                pose = list_to_pose(
                    [crate_pose[0] + TX,
                     crate_pose[1] + 0.005 + TY,
                     crate_pose[2] - HANG_OFFSET - 0.3,
                     0, 0, np.deg2rad(crate_pose[-1])]
                )

                # Move to the crate position
                self._success = self.ur3e.go_to_pose_goal(pose=pose,
                                                          child_frame_id="crate_center",
                                                          parent_frame_id="tool0")

                if not self._success:
                    break
                else:
                    rospy.loginfo("Arrived at the crate center")

                ## =========================================================================== ##
                # =================STATE 0: LOCALIZE THE BOTTLE TO BE PICKED=================== #
                # ----------------------------------------------------------------------------- #
                # Localize the bottle to be picked and move to pick the bottle                  #
                ## =========================================================================== ##

                # self.rs.set_Bottle_Flag(True)
                # rospy.sleep(3)
                # self.bottle_pose = self.rs.get_bottle_pos()

                # while self.bottle_pose is None:   # Wait until the bottle is detected
                #     rospy.loginfo("Bottle not detected!")
                #     self.bottle_pose = self.rs.get_bottle_pos()
                #     if not self.rs.get_circle_flag():
                #         self.bottle_pose = None

                # if self.bottle_pose[-1] is None:
                #     self.bottle_pose[-1] = 0.0

                # rospy.loginfo(f"Bottle pose: {self.bottle_pose}")

                # # Turn off bottle flag to stop the bottle detection
                # self.rs.set_Bottle_Flag(False)
                bottle_pose = self.look_for_bottle()

                # Close the gripper to 500 0.1mm wide to fit in the tiny gripping state
                self.ur3e.open_gripper_to(width=500, force=200)

                self._success = self.ur3e.go_to_pose_goal(pose=bottle_pose,
                                                          child_frame_id="bottle_center",
                                                          parent_frame_id="tool0")
                if not self._success:
                    rospy.loginfo("Failed to align the bottle center")

                # Lower the robot to the bottle neck
                rospy.sleep(1)
                self._success = self.ur3e.move_ee_along_axis(
                    axis="z", delta=-0.07)

                # Close the gripper to 180 0.1mm wide to grip the bottle
                rospy.sleep(1)
                if self._success:
                    rospy.loginfo("Bottle picked successfully")
                    self.ur3e.open_gripper_to(width=180, force=400)

                # setup virtual cylinder for the bottle collision object
                bottle_pose = list_to_pose([0, 0, 0.23/2, 0, 0, 0])
                _, bot_id = self.collisions.add_cylinder(
                    bottle_pose, object_type='cylinder', frame_id='end_point_link', height=0.24, radius=0.04)
                self.collisions.attach_object(eef_link='tool0', obj_id=bot_id, touch_links=[
                                              'right_inner_finger', 'left_inner_finger'])

                # Move to the classify pose
                self.ur3e.move_ee_along_axis(axis="z", delta=0.16)
                rospy.sleep(1)

                self.ur3e.go_to_target_pose_name(UR3e.DETECT_CONFIG)
                rospy.sleep(2)

                ## =========================================================================== ##
                # =================STATE 1: CLASSIFY THE OBJECT AND DELIVER==================== #
                # ----------------------------------------------------------------------------- #
                #           Classify the object and deliver to the specified location           #
                ## =========================================================================== ##

                # Turn on classification tag
                self.classifier.set_Classify_Flag(True, wait=True)
                cur_js = self.ur3e.get_current_joint_values()
                target_js = deepcopy(cur_js)
                target_js[-1] += 1.57
                while self.classifier.get_brand() is None or self.classifier.get_brand() == "Unidefined":
                    self.ur3e.go_to_goal_joint(target_js, wait=False)
                    rospy.loginfo("Waiting for classification")
                    rospy.sleep(1)
                self.ur3e.stop()
                bottle_type = self.classifier.get_brand()
                self.classifier.set_Classify_Flag(False, wait=True)

                # add thin layer for crate top collision object
                self.bound_id = self.collisions.add_crate_bound()

                self._success = self.ur3e.move_ee_along_axis(
                    axis='x', delta=0.2)
                rospy.sleep(1)

                self._success = self.ur3e.go_to_goal_joint(
                    BOTTLE_PLACEMENT[bottle_type])
                rospy.sleep(1)

                self._success = self.ur3e.move_ee_along_axis(
                    axis='z', delta=-0.155)
                if self._success:
                    rospy.loginfo("Bottle placed successfully")
                    self.ur3e.open_gripper_to(width=580, force=200)

                rospy.sleep(1)

                self.ur3e.move_ee_along_axis(axis='z', delta=0.155)
                self.ur3e.go_to_target_pose_name(UR3e.DETECT_CONFIG)

                done = True

            rospy.loginfo("System loop running")
            self.rate.sleep()

    def look_for_crate(self):

        self.rs.set_Crate_Flag(True)
        rospy.sleep(1)
        crate_pose = self.rs.get_crate_pos()
        rospy.loginfo(f"Crate pose: {crate_pose}")
        self.rs.set_Crate_Flag(False)

        pose = list_to_pose(
            [crate_pose[0] + TX,
             crate_pose[1] + 0.005 + TY,
             crate_pose[2] - HANG_OFFSET - 0.3,
             0, 0, np.deg2rad(crate_pose[-1])]
        )

        return pose

    def look_for_bottle(self):

        self.rs.set_Bottle_Flag(True)
        rospy.sleep(3)
        bottle_pose = self.rs.get_bottle_pos()

        while bottle_pose is None:   # Wait until the bottle is detected
            rospy.loginfo("Bottle not detected!")
            bottle_pose = self.rs.get_bottle_pos()
            if not self.rs.get_circle_flag():
                bottle_pose = None

        if bottle_pose[-1] is None:
            bottle_pose[-1] = 0.0

        rospy.loginfo(f"Bottle pose: {bottle_pose}")

        # Turn off bottle flag to stop the bottle detection
        self.rs.set_Bottle_Flag(False)

        # Re-align end effector to the crate center in plane xy only.
        pose = list_to_pose(
            [bottle_pose[0],
             bottle_pose[1] + 0.005,
             bottle_pose[2] - HANG_OFFSET,
             0, 0, 0]
        )
        return pose

    @staticmethod
    def call_current_service(input):
        rospy.init_node('current_service_client')
        rospy.wait_for_service('current_state')
        try:
            current_state = rospy.ServiceProxy('current_state', CurrentState)
            resp = current_state(CurrentStateRequest(input="Your input here"))
            rospy.loginfo("Service response: %s" % resp.output)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def safety_check(self):

        while not rospy.is_shutdown():
            if not self._success:
                rospy.loginfo("Failed to pick the bottle")
                rospy.signal_shutdown("Failed to pick the bottle")
            rospy.sleep(1)

    def cleanup(self):

        rospy.loginfo("Cleaning up")

        self._safety_thread.join()

        self.FINISHED = True
        self.ur3e.shutdown()

        # remove all collision objects
        self.collisions.remove_collision_object()

        rospy.loginfo("Clean-up completed")


if __name__ == "__main__":
    mp = MissionPlanner()
    mp.system_loop()
