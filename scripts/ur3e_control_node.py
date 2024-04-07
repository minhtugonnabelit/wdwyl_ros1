#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

from ur3e_controller.UR3e import UR3e
from ur3e_controller.utility import *


class MissionPlanner:

    def __init__(self) -> None:
        
        rospy.init_node("What_drink_would_you_like?", anonymous=True)
        rospy.loginfo("Initializing MissionPlanner")
        self.rate = rospy.Rate(CONTROL_RATE)
        self._ur3e = UR3e()

        # Initialize data callback channels
        # self._img_msg = None
        # self._img_sub = rospy.Subscriber("camera/image", Image, self._img_callback)
        
        # Initial action to be performed
        self._ur3e.home()

        rospy.on_shutdown(self.cleanup)

    def _img_callback(self, img_msg : Image):
        
        self._img_msg = img_msg

    def system_loop(self):
        
        rospy.loginfo("Starting system loop")
        while not rospy.is_shutdown():
            rospy.loginfo("System loop running")
            rospy.sleep(1)

    def cleanup(self):
        
        rospy.loginfo("Cleaning up")
        # self._ur3e.close_gripper()
        self._ur3e.home()
        self._ur3e.shutdown()
        rospy.loginfo("Mission complete")


if __name__ == "__main__":
    mp = MissionPlanner()
    mp.system_loop()