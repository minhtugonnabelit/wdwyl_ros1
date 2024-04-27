import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs

class RealSense:

    def __init__(self):
        self.get_crate = False
        self.get_bottle = False

        self.bridge = CvBridge()
        self.imageSub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depthSub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        # self.depthSub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        self.rgb_image = None
        self.depth_image = None
        self.depth_scale = 0.001  # Example value, replace with actual scale

        self.closest_depth = None

        self.model_path = '/home/anh/workspace/test_image_contour/src/detect/detect/train/weights/best.pt'

        # Load the YOLO model
        self.model = YOLO(self.model_path)

        # Define detection threshold
        self.threshold = 0.5

        self.depth_intrinsics = rs.intrinsics()
        self.depth_intrinsics.width = 640  # Assuming width of the depth sensor
        self.depth_intrinsics.height = 640  # Assuming height of the depth sensor
        self.depth_intrinsics.ppx = 324.8121337890625  # Principal point x (cx)
        self.depth_intrinsics.ppy = 230.55308532714844  # Principal point y (cy)
        self.depth_intrinsics.fx = 612.493408203125  # Focal length x
        self.depth_intrinsics.fy = 612.2056274414062  # Focal length y
        self.depth_intrinsics.model = rs.distortion.none  # Assuming no lens distortion
        self.depth_intrinsics.coeffs = [0, 0, 0, 0, 0]  # Assuming no distortion coefficients

        self.crate_pos = None
        self.bottle_pos = None

        self.yaw = None

    def get_crate_pos(self):
        return self.crate_pos
    
    def get_bottle_pos(self):
        return self.bottle_pos
    
    def get_yaw(self):
        return self.yaw

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
        
        tung_x = None
        tung_y = None

        if (self.get_crate == True and self.get_bottle == False):
            self.img_processing()
        
        elif (self.get_crate == False and self.get_bottle == True):
            # # Run YOLO model on the frame
            results = self.model(self.rgb_image)[0]

            # Annotate the image
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > self.threshold:
                    cv2.rectangle(self.rgb_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(self.rgb_image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # ///////////////////////////////////////////////
                    # Calculate the center of the bounding box
                    center_x_bottle = int((x1 + x2) / 2)
                    center_y_bottle = int((y1 + y2) / 2)
                    tung_x = center_x_bottle
                    tung_y = center_y_bottle
                    cv2.circle(self.rgb_image, (center_x_bottle, center_y_bottle), 5, (0, 255, 0), -1)  # Change the color and size as needed
                    # Make sure the depth image is already available and synced with the current RGB frame
                    if self.depth_image is not None:
                        # Ensure center_x and center_y are within the bounds of the depth image
                        if 0 <= center_x_bottle < self.depth_image.shape[1] and 0 <= center_y_bottle < self.depth_image.shape[0]:
                            depth = self.depth_image[center_y_bottle, center_x_bottle].astype(float)
                            depth_meters = depth * self.depth_scale  # Convert depth to meters
                            
                            # Assuming self.depth_intrinsics is set correctly
                            # real_world_coords = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [center_x_bottle, center_y_bottle], depth_meters)
                            # real_world_coords = self.project_2D_to_3D(center_x_bottle, center_y_bottle, depth_meters)
                            bottle_depth = self.closest_depth + 0.02
                            real_world_coords = self.project_2D_to_3D(center_x_bottle, center_y_bottle, bottle_depth)
                            print(f"Object real-world coordinates: x={real_world_coords[0]}, y={real_world_coords[1]}, z={real_world_coords[2]}")
                            print(bottle_depth)
                            self.bottle_pos = real_world_coords
                    # /////////////////////////////////////////////
                
        
        else:
            pass
        
        # cv2.circle(self.rgb_image, (tung_x, tung_y), 5, (0, 255, 0), -1)
        
        width = int(self.rgb_image.shape[1] * 1.5)
        height = int(self.rgb_image.shape[0] * 1.5)
        self.rgb_image = cv2.resize(self.rgb_image, (width, height))
        # Display the image
        cv2.imshow('YOLO Detection', self.rgb_image)
        # cv2.imshow('Depth Detection', self.depth_image)
        cv2.waitKey(1)  # Add this line to update the window; essential for imshow to work properly

    def depth_callback(self, data):
        # Convert the ROS Image message to a cv2 image
        self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

        # Mask out zero values (no data)
        masked_image = np.ma.masked_equal(self.depth_image, 0.0)

        # Find the closest depth (minimum value) excluding zeros
        self.closest_depth = masked_image.min() * self.depth_scale

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
            # print("contour")
            # Draw the contour itself (just for visualization)
            # cv2.drawContours(img, [crate_contour], -1, (0, 255, 0), 3)

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
        

if __name__ == '__main__':
    # Initialize the ROS Node
    rospy.init_node('realsense_yolo', anonymous=True)
    
    # Create the RealSense object
    rs = RealSense()

    # rs.get_bottle = True
    rs.get_bottle = True

    # Spin to keep the script for exiting
    rospy.spin()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
