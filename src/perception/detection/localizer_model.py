import rospy, rospkg
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs
import copy
import math

class RealSense:

    def __init__(self):
        self.get_crate = False
        self.get_bottle = False
        self.get_aruco = False

        self.bridge = CvBridge()
        self.imageSub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depthSub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        # self.depthSub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        self.rgb_image = None
        self.depth_image = None
        self.depth_scale = 0.001  # Example value, replace with actual scale

        self.closest_depth = None
        self.last_closest_depth = None

        self.model_path = rospkg.RosPack().get_path('wdwyl_ros1') + '/src/perception/detection/config/detect/detect/train/weights/best.pt'

        # Load the YOLO model
        self.model = YOLO(self.model_path)

        # Define detection threshold
        self.threshold = 0.5

        self.depth_intrinsics = rs.intrinsics()
        self.depth_intrinsics.width = 640  # Assuming width of the depth sensor
        self.depth_intrinsics.height = 480  # Assuming height of the depth sensor
        self.depth_intrinsics.ppx = 324.8121337890625  # Principal point x (cx)
        self.depth_intrinsics.ppy = 230.55308532714844  # Principal point y (cy)
        self.depth_intrinsics.fx = 612.493408203125  # Focal length x
        self.depth_intrinsics.fy = 612.2056274414062  # Focal length y
        self.depth_intrinsics.model = rs.distortion.none  # Assuming no lens distortion
        self.depth_intrinsics.coeffs = [0, 0, 0, 0, 0]  # Assuming no distortion coefficients

        self.crate_pos = None
        self.bottle_pos = None

        self.yaw = None

        self.num_of_bottle = 0
        
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

        self.circle_detected = False

        self.aruco_position = None

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()

    def set_Crate_Flag(self, flag: bool, wait=False):
        r"""
        Set the flag to enable the crate detection.
        @param: flag A boolean value"""

        self.get_crate = flag
        return self.get_crate
        
    def set_Bottle_Flag(self, flag: bool, wait=False):
        r"""
        Set the flag to enable the bottle detection.
        @param: flag A boolean value"""

        self.get_bottle = flag
        return self.get_bottle

    def get_crate_pos(self):
        return self.crate_pos
    
    def get_bottle_pos(self):
        return self.bottle_pos
    
    def get_yaw(self):
        return self.yaw
    
    def get_num_of_bottle(self):
        return self.num_of_bottle
    
    def get_circle_flag(self):
        return self.circle_detected

    def get_aruco_position(self):
        return self.aruco_position

    # Projecting a 2D point to a 3D image plane:
    def project_2D_to_3D(self, u, v, depth):

        # Extract intrinstic parameters:
        fx, fy = self.depth_intrinsics.fx, self.depth_intrinsics.fy
        cx, cy = self.depth_intrinsics.ppx, self.depth_intrinsics.ppy

        # translation of camera 
        tx = -0.0329  # unit: meters
        ty = 0.0755
        tz = -0.0632
            
        # Calculate the 3D point (with end of effector frame)
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        
        #for UR3
        x = (x + tx)*-1
        y = (y + ty)*-1
        depth = depth - tz

        P_camera = np.array([x, y, depth, self.yaw])  # 3D point in the camera frame

        
        return P_camera

    def rgb_callback(self, data):
        # Convert the ROS Image message to a cv2 image
        self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        brightness_increment = 50
        self.rgb_image = cv2.add(self.rgb_image, (brightness_increment, brightness_increment, brightness_increment, 0))

        if (self.get_crate == True and self.get_bottle == False):
            self.img_processing()
        
        elif (self.get_crate == False and self.get_bottle == True):
            
            width_in_pixels = None
            height_in_pixels = None

            lowest_corner = None

            # # Run YOLO model on the frame
            results = self.model(self.rgb_image)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > self.threshold:
                    self.num_of_bottle += 1
            # print(self.num_of_bottle)
            center_x_bottle = 0
            center_y_bottle = 0

            nearest_x_bottle = 0
            nearest_y_bottle = 0
            # Annotate the image
            self.num_of_bottle = 0
            min_dis = 1000
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > self.threshold:
                    cv2.rectangle(self.rgb_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    # cv2.putText(self.rgb_image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # ///////////////////////////////////////////////
                    # Calculate the center of the bounding box
                    center_x_bottle = int((x1 + x2) / 2)
                    center_y_bottle = int((y1 + y2) / 2)

                    # # Detect circle inside the box
                    # # Extract the region of interest (ROI) from the RGB image within the bounding box of the detected bottle
                    roi_rgb = self.rgb_image[int(y1-10):int(y2+10), int(x1-10):int(x2+10)]

                    # Convert the ROI to grayscale
                    gray_roi = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)

                    # Apply Gaussian blur to reduce noise
                    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

                    # Detect circles using Hough Circle Transform
                    circles = cv2.HoughCircles(blurred_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=20)
                    self.circle_detected = False
                    # Ensure circles were found
                    if circles is not None:
                        # Convert coordinates and radius to integers
                        circles = np.round(circles[0, :]).astype("int")

                        # Iterate over detected circles and draw them on the RGB image
                        for (x, y, r) in circles:
                            # Adjust coordinates to the original image
                            x += int(x1)
                            y += int(y1)
                            center_x_bottle = x
                            center_y_bottle = y

                            # Draw the circle
                            cv2.circle(self.rgb_image, (x, y), r, (0, 255, 0), 2)
                            cv2.circle(self.rgb_image, (x, y), 2, (0, 255, 0), 3)  # Center of the circle
                        self.circle_detected = True
                    # ////////////////////////////////////////////////

                    # spot_x = (int((center_x_bottle - min_x) / (width_in_pixels/5)))
                    # spot_y = (int((center_y_bottle - min_y) / (height_in_pixels/4)))

                    # cv2.circle(self.rgb_image, (center_x_bottle, center_y_bottle), 5, (0, 255, 0), -1)  # Change the color and size as needed
                    # Make sure the depth image is already available and synced with the current RGB frame
                    if self.depth_image is not None:
                        # Ensure center_x and center_y are within the bounds of the depth image
                        if 0 <= center_x_bottle < self.depth_image.shape[1] and 0 <= center_y_bottle < self.depth_image.shape[0]:
                            depth = self.depth_image[center_y_bottle, center_x_bottle].astype(float)
                            depth_meters = depth * self.depth_scale  # Convert depth to meters
                            
                            # Assuming self.depth_intrinsics is set correctly
                            # real_world_coords = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [center_x_bottle, center_y_bottle], depth_meters)
                            # real_world_coords = self.project_2D_to_3D(center_x_bottle, center_y_bottle, depth_meters)
                            bottle_depth = self.closest_depth + 0.048
                            real_world_coords = self.project_2D_to_3D(center_x_bottle, center_y_bottle, bottle_depth)
                            # print(f"Object real-world coordinates: x={real_world_coords[0]}, y={real_world_coords[1]}, z={real_world_coords[2]}")
                            # print(spot_x)
                            # print(spot_y)
                            
                            # print(bottle_depth)
                            # print(self.circle_detected)
                            to_cam_x = real_world_coords[0]*(-1) - (-0.0329)
                            to_cam_y = real_world_coords[1]*(-1) - 0.0755
                            distance = math.sqrt(to_cam_x*to_cam_x + to_cam_y*to_cam_y)
                            if (distance < min_dis):
                                min_dis = copy.deepcopy(distance)
                                self.bottle_pos = copy.deepcopy(real_world_coords)
                                nearest_x_bottle = copy.deepcopy(center_x_bottle)
                                nearest_y_bottle = copy.deepcopy(center_y_bottle)
                            # print(f"Object real-world coordinates: x={self.bottle_pos[0]}, y={self.bottle_pos[1]}, z={self.bottle_pos[2]}")
                    # /////////////////////////////////////////////
            print(f"Object real-world coordinates: x={self.bottle_pos[0]}, y={self.bottle_pos[1]}, z={self.bottle_pos[2]}")
            print(self.circle_detected)
            # print(self.closest_depth)
            cv2.circle(self.rgb_image, (nearest_x_bottle, nearest_y_bottle), 5, (0, 255, 0), -1)

        elif (self.get_crate == False and self.get_bottle == False and self.get_aruco == True):
            self.aruco_processing()
            
        else:
            pass

        
        width = int(self.rgb_image.shape[1] * 1.5)
        height = int(self.rgb_image.shape[0] * 1.5)
        self.rgb_image = cv2.resize(self.rgb_image, (width, height))
        # Display the image
        cv2.imshow('YOLO Detection', self.rgb_image)
        cv2.waitKey(1)  # Add this line to update the window; essential for imshow to work properly

    def depth_callback(self, data):
        # Convert the ROS Image message to a cv2 image
        self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

        # Mask out zero values (no data)
        masked_image = np.ma.masked_equal(self.depth_image, 0.0)

        # Find the closest depth (minimum value) excluding zeros
        closest_depth = masked_image.min() * self.depth_scale

        if (closest_depth > 0.3):
            self.closest_depth = closest_depth


    def img_processing(self):
        
        # img = self.rgb_image.copy()
        img = self.rgb_image


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

            # Calculate the bounding rectangle, which gives us the four points
            rect = cv2.minAreaRect(crate_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # ///////////////////////////////////
            cx_crate, cy_crate = int(rect[0][0]), int(rect[0][1])
            cv2.circle(img, (cx_crate, cy_crate), 5, (0, 255, 0), -1)
            # Make sure the depth image is already available and synced with the current RGB frame
            if self.depth_image is not None:
                # Ensure center_x and center_y are within the bounds of the depth image
                if 0 <= cx_crate < self.depth_image.shape[1] and 0 <= cy_crate < self.depth_image.shape[0]:
                    depth_crate = self.depth_image[cy_crate, cx_crate].astype(float)
                    depth_meters = depth_crate * self.depth_scale  # Convert depth to meters
                    
                    # Assuming self.depth_intrinsics is set correctly
                    # real_world_coords = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [center_x_bottle, center_y_bottle], depth_meters)
                    # real_world_coords = self.project_2D_to_3D(cx_crate, cy_crate, depth_meters)
                    real_world_coords = self.project_2D_to_3D(cx_crate, cy_crate, self.closest_depth)
                    print(f"Object real-world coordinates: x={real_world_coords[0]}, y={real_world_coords[1]}, z={real_world_coords[2]}")
                    print(self.closest_depth)
                    self.crate_pos = real_world_coords
            # /////////////////////////////////////////////

            # Draw a rectangle connecting those four points
            cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

            # Extract the angle of the rotated rectangle
            angle = rect[-1]

            if angle > 45:
                angle -= 90

            print("angle:", angle)
            self.yaw = angle

            # The order of box points: bottom-left, top-left, top-right, bottom-right
            bottom_left, top_left, top_right, bottom_right = box

            # Draw circles on the corners
            cv2.circle(img, tuple(top_left), 5, (255, 0, 255), -1)
            cv2.circle(img, tuple(top_right), 5, (0, 0, 255), -1)
            cv2.circle(img, tuple(bottom_right), 5, (255, 255, 0), -1)
            cv2.circle(img, tuple(bottom_left), 5, (0, 255, 255), -1)

    def aruco_processing(self):

        rospy.sleep(1.0)
        img = self.rgb_image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers:
        markerCorners, markerIds, _= cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        if markerIds is not None:
            ########################
            #This function is used to get the rotation matrix and translation matrix 
            #for reference: https://docs.opencv.org/4.8.0/d9/d6a/group__aruco.html#ga3bc50d61fe4db7bce8d26d56b5a6428a
            marker_size = 0.06          #replace with real marker size

            fx = self.depth_intrinsics.fx       #focal length in x axis
            fy = self.depth_intrinsics.fy      #focal length in y axis
            cx = self.depth_intrinsics.ppx           #principal point x
            cy = self.depth_intrinsics.ppy            #principal point y

            camera_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]], dtype=np.float64)

            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            k3 = 0
            
            dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, marker_size, camera_matrix, dist_coeffs)
            
            self.translation = tvecs
            self.rotation = rvecs
            

            #now use rvecs and tvecs for controller
            ########################

            cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)

            max_distance = 0
            max_position = None
            max_markerID = None

            for i in range(len(markerIds)):
                # Print the position of the marker in the camera coordinate system
                position = tvecs[i][0]
                # print(f"Marker ID: {markerIds[i][0]}, Position (x, y, z): {position}")

                distance = np.linalg.norm(position)

                if (distance > max_distance):
                    max_distance = copy.deepcopy(distance)
                    max_position = copy.deepcopy(position)
                    max_markerID = copy.deepcopy(markerIds[i][0])

                # print(f"Distance to Marker ID: {markerIds[i][0]}: {distance:.2f} meters")

                # # Draw the position on the image
                # cv2.putText(img, f"ID: {markerIds[i][0]} Pos: {position}",
                #             (int(markerCorners[i][0][0][0]), int(markerCorners[i][0][0][1])),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

                # Example: Taking the depth value of the first detected marker's center
                center_x = int((markerCorners[i][0][0][0] + markerCorners[i][0][2][0]) / 2)
                center_y = int((markerCorners[i][0][0][1] + markerCorners[i][0][2][1]) / 2)

                # Draw a circle at the center of the marker
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

            self.aruco_position = max_position
            print(f"Marker ID: {max_markerID}, Position (x, y, z): {position}")
        else:
            print("can not DETECT")
