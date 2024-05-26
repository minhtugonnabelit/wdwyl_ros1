# LSTM for improving object detection

LSTM model is used to improve the YOLO detection in real-time when the objects are moving. The model input is a sequence of 1/0, indicating if there is a detection from YOLO in a specific location. 

To train the LSTM model:
Use FrameChecker to preprocess the videos and save the sequence to a npy file. This class also handles the labelling using two methods: human labelling for true positive detection or synthesising plausible sequences of 1/0.

The target data is 1/0 of whether a specific bounding box appears in the frame or not.

We use YOLO to detect the object, then feed the results of YOLO into the LSTM model to improve the probability of YOLO’s detection