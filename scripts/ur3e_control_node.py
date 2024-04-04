#! /usr/bin/env python3

import rospy

from ur3e_controller.UR3e import UR3e
from ur3e_controller.utility import *

class MissionPlanner:

    def __init__(self) -> None:

        rospy.loginfo("Initializing MissionPlanner")
        self._ur3e = UR3e()
        self._ur3e.home()
        self._ur3e.open_gripper()

        rospy.on_shutdown(self.cleanup)

    def system_loop(self):
        
        rospy.loginfo("Starting system loop")
        while not rospy.is_shutdown():
            rospy.loginfo("System loop running")
            rospy.sleep(1)

    def cleanup(self):
        rospy.loginfo("Cleaning up")
        self._ur3e.close_gripper()
        self._ur3e.home()
        rospy.loginfo("Mission complete")