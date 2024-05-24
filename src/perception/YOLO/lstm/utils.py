import os 
import numpy as np
import cv2
from ultralytics import YOLO

MODEL_PATH = '/root/aifr/wdwyl_ros1/config/detect/detect/train/weights/best.pt'



def match_detections(d1, d2):
    # Simplest matching based on the closest center point
    if not d1 or not d2:
        return [], []

    matched = []
    for det in d1: # det is (x1, y1, x2, y2)
        # Calculate center of each detection in d1
        center_x1 = (det[0] + det[2]) / 2
        center_y1 = (det[1] + det[3]) / 2

        # Find the closest detection in d2
        distances = [(np.hypot(center_x1 - ((d[0]+d[2])/2), center_y1 - ((d[1]+d[3])/2)), i) for i, d in enumerate(d2)]
        distances.sort()
        matched_index = distances[0][1]
        matched.append((det, d2[matched_index]))

    # Unpack matched pairs
    matched_d1, matched_d2 = zip(*matched) if matched else ([], [])
    return matched_d1, matched_d2


def load_video_frames(video_path):
    # This function loads video frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames



def get_bottle_position(video_path, model):
    THRESHOLD = 0.5

    def is_coordinate_matched(p1, p2):
        threshold=200
        return abs(p1-p2) < threshold
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file")
        return []
    if os.path.isfile(video_path):
        print("File exists")
    
    positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)[0]
        print("results: ", len(results.boxes.data.tolist()))

        # frame_height, frame_width, frame_channels = frame.shape
        # print(f"Frame size: Width={frame_width}, Height={frame_height}, Channels={frame_channels}")
    
        # if previous_detections is not None:
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = result
            # print(f"Initial detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score}")
            if score > THRESHOLD:
                matched = False
                for pos in positions:
                    if (is_coordinate_matched(pos[0], x1) and 
                        is_coordinate_matched(pos[1], y1) and
                        is_coordinate_matched(pos[2], x2) and 
                        is_coordinate_matched(pos[3], y2)):
                        matched = True
                        break
                if not matched:
                    positions.append((x1, y1, x2, y2))
                    # print(f"Added position: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    return positions

if __name__ == '__main__':

    video_path = '/root/aifr/wdwyl_ros1/videos/IMG_1746.mp4' # Width=1080, Height=1920
    model = YOLO(MODEL_PATH)

    positions = get_bottle_position(video_path, model)
    # print("Found Positions: ", len(positions))    
    # print(positions)




