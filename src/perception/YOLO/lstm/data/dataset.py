import os 
import sys
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from ultralytics import YOLO
import matplotlib.pyplot as plt

MODEL_PATH = '/root/aifr/wdwyl_ros1/config/detect/detect/train/weights/best.pt'


class BoundingBoxDataset(Dataset):
    """
    Dataset for the LSTM model
    Use FrameChecker to preprocess the videos, save the sequence to a npy file

    The target data is a sequence of whether a bounding box appears in the frame or not (check FrameChecker for more details)

    The labels are the ground truth for the sequence (1 for true positive and 0 for false positive)
    
    Args:
        videos_path (str): The path to the videos directory
        sqeuence_length (int): The length of the sequence
    """

    DATA_DIR = '/root/aifr/wdwyl_ros1/src/perception/YOLO/lstm/data/sequence_npy'

    def __init__(self, videos_path, yolo_model, sequence_length):
        self.videos_path = videos_path
        self.videos = os.listdir(videos_path)
        self.yolo_model = yolo_model

        # self.npy_files = os.listdir(BoundingBoxDataset.DATA_DIR)
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.preprocess()

        
    def preprocess(self):
        """
        Preprocess the videos to get the bounding boxes
        """

        self.sequences = []
        self.labels = []

        for video in self.videos:
            video_path = os.path.join(self.videos_path, video)

            checker = FrameChecker(self.yolo_model, video_path)
            sequence_path, labels = checker.process()
            sequence = np.load(sequence_path) # Load from npy, sequence is a list of list of 0 and 1

            for bb_id, bbox in enumerate(sequence):
                for i in range(len(bbox)-self.sequence_length+1):
                    self.sequences.append(bbox[i:i+self.sequence_length])
                    self.labels.append(labels[bb_id])

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32).to(self.device)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).to(self.device)
        return sequence, label





def is_equal(p1, p2, threshold=200):
    return abs(p1-p2) < threshold


class FrameChecker:
    """
    Preprocessing to obtain the dataset for the LSTM model
    For workflow, check process() method

    Args:
        model (nn.Module): The YOLO model
        video_path (str): The path to the video file
        threshold (int): The threshold for the detection score
    """

    DATA_DIR = '/root/aifr/wdwyl_ros1/src/perception/YOLO/lstm/data/sequence_npy'


    def __init__(self, model, video_path):
        self.valid = True
        self.model = model
        self.threshold = 0.5
        
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Error: The video file {self.video_path} does not exist.")
        self.video_path = video_path

    def process(self):
        """
        Process the video and save the sequence to a npy file
        """
        
        print("------ Getting all of the detected bounding boxes ------")
        all_detected_bbox = self.get_detection()

        print("------ Labeling the valid bounding boxes ------")
        print("Press 'q' to accept, 'f' to reject")
        valid_bbox = self.label_valid_bbox()

        print("------ Getting the sequence and save to npy file ------")
        sequence_path = self.get_sequence(all_detected_bbox)

        print("------ Getting the labels ------")
        labels = self.get_label(all_detected_bbox, valid_bbox)

        return sequence_path, labels


    @staticmethod
    def is_bbox_matched(b1, b2, threshold=200):
        count = 0
        for i in range(4):
            if abs(b1[i] - b2[i]) < threshold:
                count += 1
        return count >= 3

    def on_key(self, event):
        if event.key == 'q':
            self.valid = True
            plt.close(event.canvas.figure)
        elif event.key == 'f':
            self.valid = False
            plt.close(event.canvas.figure)

    def init_cap(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Error: Unable to open video file {self.video_path}")
        return cap

    def label_valid_bbox(self):
        '''
        Label using the first frame of the video
        Can add setting to use the next frame if the first frame is not good enough
        Show the final selection at the end, try again if not approved
        '''

        while True:
            cap = self.init_cap()
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read the first frame")
                return

            # Check only the first frame
            with torch.no_grad():
                results = self.model(frame)[0]
            valid_bbox = []

            final_frame = frame.copy()

            for id, result in enumerate(results.boxes.data.tolist()):
                x1, y1, x2, y2, score, cls = result
                # print(f"Detection {id}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score}, class={cls}")
                
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Convert the frame copy to RGB format for Matplotlib
                frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                # Create a plot to display the frame
                fig, ax = plt.subplots()
                ax.imshow(frame_rgb)
                plt.axis('off')  # Turn off axis numbers and ticks
                # Connect the key press event to the handler
                fig.canvas.mpl_connect('key_press_event', self.on_key)
                plt.show()

                # Print the result of the detection validity
                if self.valid:
                    print("Detection is valid")
                    valid_bbox.append((x1, y1, x2, y2))
                    cv2.rectangle(final_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                else:
                    print("Detection is not valid")
                    cv2.rectangle(final_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


            final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots()
            ax.imshow(final_frame_rgb)
            plt.axis('off')  # Turn off axis numbers and ticks
            fig.canvas.mpl_connect('key_press_event', self.on_key)
            plt.show()

            if self.valid:
                print("Accept the set of correct bounding boxes")
                break
            else:
                print("Try again")

        return valid_bbox


    def get_detection(self):
        """
        Run through the videos and get all the detected bounding boxes
        Perform matching to avoid duplicates
        """
        detected_bbox = []

        cap = self.init_cap()   
        with torch.no_grad():  
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame)[0]        
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, cls = result
                    # print(f"Initial detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score}")
                    if score > self.threshold:
                        matched = False
                        for pos in detected_bbox:
                            if (is_equal(pos[0], x1) and 
                                is_equal(pos[1], y1) and
                                is_equal(pos[2], x2) and 
                                is_equal(pos[3], y2)):
                                matched = True
                                break
                        if not matched:
                            detected_bbox.append((x1, y1, x2, y2))
        return detected_bbox


    def get_sequence(self, all_detected_bbox):
        """
        From the detected bounding boxes and ground truth bounding boxes, get the sequence 

        The sequence is a list of list of 0 and 1 (1 means the bounding box is detected in the frame, 0 otherwise)
        
        Therefore, sequence is a list of list of 0 and 1. 
        The top level list represents a list of bounding boxes, each inner list represents a single bounding box. 
        Each element in the inner list represents a frame, can be 0 and 1. 
        0 means the bounding box is not detected in the frame, 1 otherwise.
        """

        sequence = [[] for _ in range(len(all_detected_bbox))]

        cap = self.init_cap()
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.model(frame)[0]    

                all_detected_bbox_in_frame = []
                for id, d_bbox in enumerate(all_detected_bbox): # For each detected bbox

                    # Does that bbox appear in this frame?
                    in_frame = False
                    for result in results.boxes.data.tolist():
                        x1, y1, x2, y2, score, cls = result
                        if self.is_bbox_matched(d_bbox, (x1, y1, x2, y2)):
                            in_frame = True
                            break
                    if in_frame:
                        all_detected_bbox_in_frame.append(True)
                    else:
                        all_detected_bbox_in_frame.append(False)
                assert len(all_detected_bbox) == len(all_detected_bbox_in_frame)

                for i in range(len(all_detected_bbox)):
                    if all_detected_bbox_in_frame[i]:
                        sequence[i].append(1)
                    else:
                        sequence[i].append(0)

        # print(sequence)

        # Save to npy file
        filename = os.path.basename(self.video_path)
        video_name, _ = os.path.splitext(filename)
        save_path = os.path.join(FrameChecker.DATA_DIR, f'sequence_{video_name}.npy')        
        np.save(save_path, np.array(sequence))

        return save_path
    

    def get_label(self, all_detected_bbox, valid_bbox):
        """
        From all the detected bounding boxes, which is correct?
        """

        labels = []        
        for d_bbox in all_detected_bbox:
            correct = False
            for valid_box in valid_bbox:
                if self.is_bbox_matched(d_bbox, valid_box):
                    correct = True
                    break
            if correct:
                labels.append(1)
            else:
                labels.append(0)
        assert len(labels) == len(all_detected_bbox)
        return labels


if __name__ == '__main__':

    model = YOLO(MODEL_PATH)
    video_path = '/root/aifr/wdwyl_ros1/videos/IMG_1746.mp4' # Width=1080, Height=1920


    # checker = FrameChecker(model, video_path)
    # checker.process()

    data = BoundingBoxDataset('/root/aifr/wdwyl_ros1/videos', 10)