#! /usr/bin/env python3

import rospy
import tf

# Importing planner module
from ur3e_controller.UR3e import UR3e
from ur3e_controller.utility import *

# Importing perception module
from perception.localizer_model import RealSense

# @TODO: Implementing loading scene with object detected
# @TODO: initializing a static tf from camera to toolpose via static broadcaster

class MissionPlanner:

    MISSION_STATE = {
    "LOCALIZE": 1,
    "CLASSIFY": 2,
    "GRASP": 3,
    "DELIVER": 4

}
    
    action = [
    "LOCALIZE",
    "CLASSIFY",
    "GRASP",
    "DELIVER"

    ]

    def __init__(self) -> None:
        
        rospy.init_node("What_drink_would_you_like", anonymous=True)
        rospy.loginfo("Initializing MissionPlanner")
        self.rate = rospy.Rate(CONTROL_RATE)

        # Initialize the UR3e controller
        self._ur3e = UR3e()
        ee_link = self._ur3e.get_end_effector_link()

        # Initialize the perception module
        self._rs = RealSense(on_UR=True, tool_link=ee_link)
        
        # Initial action to be performed
        self._ur3e.home()
        self._ur3e.open_gripper(force=0)

        rospy.on_shutdown(self.cleanup)


    def system_loop(self):
        
        rospy.loginfo("Starting system loop")
        
        done = False
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

            rospy.loginfo("System loop running")
            rospy.sleep(1)

    def cleanup(self):
        
        rospy.loginfo("Cleaning up")

        self._ur3e.home()
        self._ur3e.shutdown()
        
        rospy.loginfo("Mission complete")


if __name__ == "__main__":
    mp = MissionPlanner()
    mp.system_loop()