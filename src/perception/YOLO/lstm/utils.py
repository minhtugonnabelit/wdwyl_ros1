import os 
import numpy as np
import cv2
from ultralytics import YOLO

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



