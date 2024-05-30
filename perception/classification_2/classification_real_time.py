import rospy, rospkg
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs
import copy
import math

class Classification:
    def __init__(self):
        self.classification_flag = False

        self.imageSub = rospy.Subscriber("/usb_cam/image_raw", Image, self.rgb_callback)

        self.model_path = rospkg.RosPack().get_path('wdwyl_ros1') + '/config/runs/classify/train/weights/best.pt'

        self.bridge = CvBridge()
        
        self.model = YOLO(self.model_path)

        self.rgb_image = None

    def rgb_callback(self, data):
        # Convert the ROS Image message to a cv2 image
        self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        if self.classification_flag:
            result = self.model(self.rgb_image)

            names_dict = result[0].names

            probs = result[0].probs
            
            # name = "Unidefined"
            # # print(names_dict[probs.top1])
            if probs.top1 > 0.5:
                name = names_dict[probs.top1]
            elif probs.top1 < 0.5:
                name = "Unidefined"
            # if probs.top1 < 0.5:
            #     name = "Unidefined"

            text_position = (10, self.rgb_image.shape[0] - 10)

            cv2.putText(self.rgb_image, name.upper(), text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Display the image
            cv2.imshow('Result', self.rgb_image)
            cv2.waitKey(1)

if __name__ == '__main__':
    # Initialize the ROS Node
    rospy.init_node('realsense_yolo', anonymous=True, log_level=1)
    
    # Create the RealSense object
    rs = Classification()

    rs.classification_flag = True

    # Spin to keep the script for exiting
    rospy.spin()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
