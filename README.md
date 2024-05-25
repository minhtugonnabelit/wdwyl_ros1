# What drink would you like?

Bottle sorting is a labourious task that involves:

- Detect bottles inside a crate

For efficient bottle sorting, our initial step is to detect the presence of bottles within a crate. We employ YOLO v8, a state-of-the-art deep learning object detection system, tailored for high-speed and accurate performance. Our dataset, created manually, comprises a variety of images capturing bottles in mixed and challenging scenarios to mimic real-world conditions. This dataset is utilized to train the YOLO model, enabling it to identify different bottle types and their orientations within a crate with high precision.
  
- Picking each bottle from from a mixed crate

Once bottles are detected, the next crucial step is their precise localization. To achieve this, we combine the detection results from the YOLO model with inputs from an RGB-D (Red, Green, Blue - Depth) camera. This setup allows us to ascertain the exact 3D positions of the bottles. The RGB component of the camera captures detailed color imagery of the crate's contents, while the depth sensor provides the distance information necessary to map each bottle in three-dimensional space. This integrated approach ensures accurate placement and retrieval of each bottle, which is critical for the subsequent sorting and classification tasks.

- Classify them by brand

- Place them into respective crates

Our team ideated an automated picking process 
