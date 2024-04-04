import sys


import rospy
import moveit_commander
import geometry_msgs.msg


robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()