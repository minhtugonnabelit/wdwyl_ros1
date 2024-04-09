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


    # Callback function for the subscriber:
    def depth_callback(self, msg):

        # Store the image in the data member:
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # print(self.depth_image.shape)

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # print(self.rgb_image.shape)
        self.img_processing()
        # self.process_image()

    def process_image(self):
        # Check if depth image is available
        if self.depth_image is None:
            rospy.logwarn("Depth image not available.")
            return

        # Convert depth image to meters
        depth_meters = self.depth_image * 0.001  # Convert mm to meters

        # Threshold the depth image to select pixels between 0.2 and 0.5 meters
        mask = cv2.inRange(depth_meters, 0.01, 0.45)

        # Apply the mask to the RGB image
        masked_rgb = cv2.bitwise_and(self.rgb_image, self.rgb_image, mask=mask)

        # Convert to grayscale
        gray_image = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2GRAY)

        # Blur the image to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 1.5)

        # Circle detection using Hough Transform
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, minDist=30,
                                   param1=50, param2=30, minRadius=10, maxRadius=30)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                if 10 < i[2] < 30:  # example radius range, adjust as needed
                    # Draw the outer circle
                    cv2.circle(masked_rgb, (i[0], i[1]), i[2], (255, 0, 0), 2)
                    # Draw the center of the circle
                    cv2.circle(masked_rgb, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Display the masked RGB image
        cv2.imshow("Objects between 0.2m and 0.5m", masked_rgb)
        cv2.waitKey(1)

    def img_processing(self):
        
        img = self.rgb_image
        # Resize the image if it's too large for display
        # Specify the desired width and height or a scaling factor
        # scale_percent = 50  # percentage of original size
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)
        
        # # Resize image
        # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur the image to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 1.5)

        # Use adaptive thresholding to enhance the crate edges
        thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection
        edges = cv2.Canny(blurred_image, 50, 150)

        # Convert image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color range for segmentation (adjust these values)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        
        # Create a mask for the blue color ("color" of the crate)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply morphological closing (dilation followed by erosion)
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        crate_contour = max(contours, key=cv2.contourArea)

        # Initialize a variable to hold the largest area found
        # largest_area = 0
        # largest_contour = None
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        if contours is not None:
            # print("contour")
            # Draw the contour itself (just for visualization)
            cv2.drawContours(img, [crate_contour], -1, (0, 255, 0), 3)

            # Calculate the bounding rectangle, which gives us the four points
            rect = cv2.minAreaRect(crate_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Draw a rectangle connecting those four points
            cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

            # Extract the angle of the rotated rectangle
            angle = rect[-1]

            if angle > 45:
                angle -= 90

            # print(angle)

            # The order of box points: bottom-left, top-left, top-right, bottom-right
            bottom_left, top_left, top_right, bottom_right = box

            # Draw circles on the corners
            cv2.circle(img, tuple(top_left), 5, (255, 0, 255), -1)
            cv2.circle(img, tuple(top_right), 5, (0, 0, 255), -1)
            cv2.circle(img, tuple(bottom_right), 5, (255, 255, 0), -1)
            cv2.circle(img, tuple(bottom_left), 5, (0, 255, 255), -1)
        
        # Show the result
        cv2.imshow("Detected Shapes", self.rgb_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('real_sense_node')
    real_sense = RealSense()
    rospy.spin()