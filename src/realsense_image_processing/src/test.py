import cv2
# print(cv2.__version__)
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import imutils

class ShapeDetector:

    def __init__(self, image_path, ref_image_path):
        # Load the reference image
        self.image = cv2.imread(image_path)
        self.ref_image = cv2.imread(ref_image_path)

        self.process_image(self.image, self.ref_image)


    def process_image(self, img, ref_img):
        
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

        if crate_contour is not None:
            # Draw the contour itself (just for visualization)
            cv2.drawContours(img, [crate_contour], -1, (0, 255, 0), 3)

            # Calculate the bounding rectangle, which gives us the four points
            rect = cv2.minAreaRect(crate_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Draw a rectangle connecting those four points
            # cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

            # Extract the angle of the rotated rectangle
            angle = rect[-1]

            if angle > 45:
                angle -= 90

            print(angle)

            # The order of box points: bottom-left, top-left, top-right, bottom-right
            bottom_left, top_left, top_right, bottom_right = box

            # Draw circles on the corners
            cv2.circle(img, tuple(top_left), 5, (255, 0, 255), -1)
            cv2.circle(img, tuple(top_right), 5, (0, 0, 255), -1)
            cv2.circle(img, tuple(bottom_right), 5, (255, 255, 0), -1)
            cv2.circle(img, tuple(bottom_left), 5, (0, 255, 255), -1)

        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
            
        #     # Filter on the basis of the area
        #     if area > 500:  # example threshold, adjust as needed
                
        #         # Approximate the contour to a polygon
        #         epsilon = 0.02 * cv2.arcLength(cnt, True)
        #         approx = cv2.approxPolyDP(cnt, epsilon, True)
                
        #         # If the polygon has 4 vertices, consider it as a rectangle
        #         if len(approx) == 4:
        #             print("rec detected")
        #             (x, y, w, h) = cv2.boundingRect(approx)
        #             aspect_ratio = float(w) / h
        #             # Check if aspect ratio is close to that of a crate
        #             # if 0.8 < aspect_ratio < 1.2:  # example range, adjust as needed
        #             cv2.drawContours(img, [approx], 0, (0, 255, 0), 4)
        
        # Circle detection using Hough Transform
        # circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, minDist=30,
        #                            param1=50, param2=30, minRadius=10, maxRadius=30)
        
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         if 10 < i[2] < 30:  # example radius range, adjust as needed
        #             # Draw the outer circle
        #             cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
        #             # Draw the center of the circle
        #             cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        # Show the result
        cv2.imshow("Detected Shapes", img)
        cv2.waitKey(0)  # Change to 0 for static images to wait indefinitely until a key is pressed
        cv2.destroyAllWindows()


class RealSense:

    # Constructor:
    def __init__(self, image_path, ref_image_path):
        # Create a CvBridge object
        self.bridge = CvBridge()

        # Subscriber:
        self.image = cv2.imread(image_path)
        self.ref_image = cv2.imread(ref_image_path)

        self.process_img(self.image, self.ref_image)


    
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

    def process_img(self, img, ref_img):
        # try:
        #     # Convert the ROS image message to OpenCV format
        #     self.rgb_image = self.bridge.imgmsg_to_cv2(img, 'bgr8')
        # except CvBridgeError as e:
        #     print(e)

        # Resize the image if it's too large for display
        # Specify the desired width and height or a scaling factor
        scale_percent = 50  # percentage of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # Resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        ref_img = cv2.resize(ref_img, dim, interpolation = cv2.INTER_AREA)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur the image to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        ref_gray_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        # Subtract the reference image from the current image
        diff_image = cv2.absdiff(ref_gray_img, gray_image)

        # Threshold the difference image
        _, thresh_image = cv2.threshold(diff_image, 25, 255, cv2.THRESH_BINARY)

        # Find contours of the bottles
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0] if len(contours) == 2 else contours[1]
        # Draw contours on the current image
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        
        # Show the result
        cv2.imshow("Detected Shapes", img)
        cv2.waitKey(0)  # Change to 0 for static images to wait indefinitely until a key is pressed
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Replace 'path_to_image' with your image file path
    path_to_image = '/home/anh/workspace/test_image_contour/src/data_image/train/0b5c08d8-d1df-4683-a870-bdb94e9fbfe1.jpg'
    # path_to_image = '/home/anh/workspace/test_image_contour/src/data_image/Bottle5.jpeg'
    path_to_ref_image = '/home/anh/workspace/test_image_contour/src/data_image/NoBottle1.jpeg'
    detector = ShapeDetector(path_to_image, path_to_ref_image)
