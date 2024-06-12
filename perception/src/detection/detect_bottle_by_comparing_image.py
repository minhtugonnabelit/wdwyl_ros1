#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class RealSense:

    # Constructor:
    def __init__(self):
        # Create a CvBridge object
        self.bridge = CvBridge()

        # Subscriber:
        self.imageSub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depthSub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)

        # Data Members of this class:
        self.depth_image = None
        self.rgb_image = None

        self.depth = None

        self.reference_image = cv2.imread('/home/anhtungnguyen/referenced_image.jpg', cv2.IMREAD_GRAYSCALE)
        self.reference_image = cv2.GaussianBlur(self.reference_image, (5, 5), 0)

    
    def crop(self, img,factor):
        # Get the dimensions of the image
        height, width = img.shape[:2]

        # Calculate the dimensions of the rectangle to crop
        w = int(width / (factor*1.6))
        h = int(height / (factor))


        x = (width - w) // 2
        y = (height - h) // 2

        # Crop the image
        cropped = img[y:y+h, x:x+w]

        # Create a black image with the same size as the original image
        black = np.zeros((height, width), dtype=np.uint8)

        # Calculate the dimensions to paste the cropped image onto the black image
        x_offset = (width - w) // 2
        y_offset = (height - h) // 2

        # Paste the cropped image onto the black image
        black[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

        return black

    def canny_edge(self, img):

        img = cv2.convertScaleAbs(img, alpha=1, beta=110)

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grey = cv2.GaussianBlur(grey, (5,5),1)

        grey = self.crop(grey,1.7)

        T, thresh = cv2.threshold(grey, 40, 255, 0)

        # Perform Canny edge detection
        edges = cv2.Canny(grey, 70, 200)

        return edges

    # Callback function for the subscriber:
    def depth_callback(self, msg):

        # Store the image in the data member:
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg):
        try:
            # Convert the ROS image message to OpenCV format
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            print(e)

        
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)

        # Blur the image to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Subtract the reference image from the current image
        diff_image = cv2.absdiff(self.reference_image, gray_image)

        # Threshold the difference image
        _, thresh_image = cv2.threshold(diff_image, 25, 255, cv2.THRESH_BINARY)

        # Find contours of the bottles
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the current image
        cv2.drawContours(self.rgb_image, contours, -1, (0, 255, 0), 2)

        
        # Show the result
        cv2.imshow("Detected Shapes", self.rgb_image)
        cv2.waitKey(1)

