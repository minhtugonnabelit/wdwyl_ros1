import rospy, rospkg, tf2_ros
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs

# from perception.utility import *

class Classifier:

    def __init__(self) -> None:

        self._bottle_class = None
        pass

    @property
    def bottle_class(self) -> str:
        return self._bottle_class

    def rgb_callback(self, data: Image) -> None:
        pass

    def depth_callback(self, data: Image) -> None:
        pass

    def set_classify_flag(self, flag: bool) -> None:
        pass


