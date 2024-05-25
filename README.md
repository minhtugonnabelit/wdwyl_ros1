# What drink would you like?

This package is an integration of Moveit! planning framework and YOLO V8 model

### System setup requirement

#### Hardware
Beside the robot UR3e with OnRobot RG2 gripper, the system requires additional hardware modification and sensors setup perception system.
Here is an image of an example setup around the robot on a custom trolley.

![Example Image](image/imfine.jpg)

#### Moveit! and YOLO setup
For planning module, a custom Moveit! configuration package are created with the consideration for the trolley and gripper with camera mounted on it. Using the setup assistance from Moveit! allow us to seemlessly account for the collision object relatively on robot body for collision avoidance feature of Moveit!. A custom URDF also need to be defined based on URDF available from each component description package (if available) and manually measured one.

With this custom Moveit! configuration file we built and addtional workspace constraint we set, UR3e with RG2 gripper are able to consider the bottle picked along with the gripper for collision-free trajectory planning for bottle pick and place mission.

##### Other requirement dependencies
 - OpenCV
 - Ultralytics
 - pymodbus==2.5.3

##### ROS installation
For installation, please follow the instruction the snippet below 
```shell
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src 

# clone standard package for ur robot, onrobot gripper and realsense camera driver 
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git
git clone -b $ROS_DISTRO-devel https://github.com/ros-industrial/universal_robot.git

git clone https://github.com/takuya-ki/onrobot.git --depth 1
git clone https://github.com/roboticsgroup/roboticsgroup_upatras_gazebo_plugins.git --depth 1

sudo apt-get install ros-$ROS_DISTRO-realsense2-camera

# clone custom Moveit! config package
git clone https://github.com/minhtugonnabelit/ur3e_rg2_moveit_config.git

# clone this package to your workspace
git clone https://github.com/minhtugonnabelit/wdwyl_ros1.git

# install all required dependency for this package
sudo rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y --os=ubuntu:focal -y

# build
cd ../
catkin build
```

Run this line to start necessary driver for robot, gripper and camera on the master machine. 
```shell
roslaunch wdwyl_ros1 ur3e_rg2_bringup.launch robot_ip:=<your_robot_ip> gripper:=<you_gripper_type> ip:=<your_gripper_ip> 
```

To start the automated sorting example, run the line below
```shell
rosrun wdwyl_ros1 ur3e_control_node.py
```

### System operation guide
Bottle sorting is a labourious task that involves:

- Detect bottles inside a crate

For efficient bottle sorting, our initial step is to detect the presence of bottles within a crate. We employ YOLO v8, a state-of-the-art deep learning object detection system, tailored for high-speed and accurate performance. Our dataset, created manually, comprises a variety of images capturing bottles in mixed and challenging scenarios to mimic real-world conditions. This dataset is utilized to train the YOLO model, enabling it to identify different bottle types and their orientations within a crate with high precision.
  
- Picking each bottle from from a mixed crate

Once bottles are detected, the next crucial step is their precise localization. To achieve this, we combine the detection results from the YOLO model with inputs from an RGB-D (Red, Green, Blue - Depth) camera. This setup allows us to ascertain the exact 3D positions of the bottles. The RGB component of the camera captures detailed color imagery of the crate's contents, while the depth sensor provides the distance information necessary to map each bottle in three-dimensional space. This integrated approach ensures accurate placement and retrieval of each bottle, which is critical for the subsequent sorting and classification tasks. With this precise bottle location datea 


- Classify them by brand

- Place them into respective crates

Our team ideated an automated picking process 

