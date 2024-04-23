#! /usr/bin/env python3

import rospy
import tf

# Importing planner module
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker
from ur3e_controller.UR3e import UR3e
from ur3e_controller.collision_manager import CollisionManager
from ur3e_controller.utility import *

# Importing perception module
from perception.localizer_model import RealSense

# @TODO: Implementing loading scene with object detected
# @TODO: initializing a static tf from camera to toolpose via static broadcaster


class MissionPlanner:

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
        # self._rs = RealSense(on_UR=True, tool_link=ee_link)

        # Initial action to be performed

        # self.setup_scene()
        self._ur3e.home()
        self._ur3e.open_gripper(force=400)

        rospy.on_shutdown(self.cleanup)

    def setup_scene(self):

        # Add static collision object rather than the trolley and the plate
        wall = get_pose(155 * 0.001, -295 * 0.001, 105 * 0.001, 0, 0, 1.57)
        _, self.wall_ID = self.collisions.add_abitrary_collision_object(
            pose=wall, obj_ID="wall", frame_id="base", size=(0.34, 0.1, 0.18))

        ceilling = get_pose(0 * 0.001, 0 * 0.001, 605 * 0.001, 0, 0, 0)
        _, self.ceilling_ID = self.collisions.add_abitrary_collision_object(
            pose=ceilling, obj_ID="ceilling", frame_id="base", size=(1.5, 1.5, 0.01))

        rospy.sleep(1)

    def box_pick(self):

        rospy.loginfo("box pick action")
        end_effector_link = self._ur3e.get_end_effector_link()

        # Move the robot to the localize pose
        box_pose = get_pose(-172.44 * 0.001, -324 *
                            0.001, 54.7 * 0.001, 0, 0, 0)
        _, box_ID = self.collisions.add_box_collision_object(pose=box_pose,
                                                             object_type="cube",
                                                             frame_id="base",
                                                             size=(0.06, 0.06, 0.06))
        rospy.sleep(1)

        goal_pose = PoseStamped()
        goal_pose.pose = box_pose
        goal_pose.header.frame_id = "base"
        goal_pose.pose = get_pose(-172.44 * 0.001, -
                                  324 * 0.001, 54.7 * 0.001, 0, 3.14, 0)
        self._ur3e.go_to_pose_goal(goal_pose)
        rospy.sleep(1)

        self._ur3e.close_gripper(force=400)
        rospy.sleep(3)

        self.collisions.attach_object(end_effector_link, box_ID)

        target_pose_stamped = PoseStamped()
        target_pose_stamped.pose = get_pose(309 * 0.001, -320 * 0.001,
                                            54.78 * 0.001, -3.14, 0, -3.140)
        target_pose_stamped.header.frame_id = "base"
        self._ur3e.go_to_pose_goal(target_pose_stamped)
        rospy.sleep(1)

        self._ur3e.open_gripper(force=400)
        rospy.sleep(3)

        self.collisions.detach_object(link=end_effector_link, name=box_ID)
        rospy.sleep(1)

        self._ur3e.home()


    def system_loop(self):

        rospy.loginfo("Starting system loop")

        done = False
        on_going = False

        while not rospy.is_shutdown():

            # State 1: Localize the bottle to be picked and move to pick the bottle

            # Call service to localize the bottle, return the pose of bottle chosen from perception node

            # Feed coordinates to UR3e controller to plan a collision free path to the object

            # Move the robot to the object and hang the gripper to the object at pose with bottle z + hanging offset

            # Lower the gripper to the bottle and close the gripper

            # Attach bottle to the end effector of the robot in planning scene

            # Move the robot to the classify pose

            # State 2: Classify the object and deliver to the specified location

            # Call service to classify the object, return the class of the bottle

            # Set type of bottle by changing the object name in planning scene and its color

            # Specify the deliver pose for the bottle type

            # Plan a collision free path to the deliver pose

            # Move the robot to the deliver pose and drop the bottle

            # State 3: Back to home position ready for repeared action

            # Move the robot to the home position

            # Reset the planning scene

            # rospy.loginfo("System loop running")

            # if done:
            #     rospy.signal_shutdown("Mission complete")

            rospy.sleep(1)

    def cleanup(self):

        rospy.loginfo("Cleaning up")

        # self._ur3e.home()
        self._ur3e.shutdown()

        # remove all collision objects
        self.collisions.remove_collision_object()

        rospy.loginfo("Mission complete")


if __name__ == "__main__":
    mp = MissionPlanner()
    mp.box_pick()
    # mp.system_loop()
