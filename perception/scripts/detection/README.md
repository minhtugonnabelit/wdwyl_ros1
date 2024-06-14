# Bottle Detection and Localisation

For efficient bottle sorting, our initial step is to detect the presence of bottles within a crate. We employ YOLO v8, a state-of-the-art deep learning object detection system, tailored for high-speed and accurate performance. Our dataset, created manually, comprises a variety of images capturing bottles in mixed and challenging scenarios to mimic real-world conditions. This dataset is utilized to train the YOLO model, enabling it to identify different bottle types and their orientations within a crate with high precision.

Once bottles are detected, the next crucial step is their precise localization. To achieve this, we combine the detection results from the YOLO model with inputs from an RGB-D (Red, Green, Blue - Depth) camera. This setup allows us to ascertain the exact 3D positions of the bottles. The RGB component of the camera captures detailed color imagery of the crate's contents, while the depth sensor provides the distance information necessary to map each bottle in three-dimensional space. This integrated approach ensures accurate placement and retrieval of each bottle, which is critical for the subsequent sorting and classification tasks.


# Main Files

`image_processing_before_training.py` image pre-processing module

`real_time_detect_using_model.py` module for YOLOv8 real-time detection with ROS

`trained_model_test.py` module for YOLOv8 raw image detection

`utility.py` config files/utility

`Compute_global_mean_and_std_dev.py`

# Previous Approaches

`detect_bottle_by_comparing_image.py` a failed attempt to compare empty slots with occupied slots to roughly perform detection

`detect_bottle_using_aimge_processing.py` a failed attempt to detect using traditional detection methods

`shape_detector.py` inferring from raw image, using traditional methods

