# GAP_SAC
## Demo Video
[![Deep Reinforcement Learning for Path Planning of Autonomous Mobile Robots in Complicated Environments](https://res.cloudinary.com/marcomontalbano/image/upload/v1730985565/video_to_markdown/images/youtube--PEMLhgnqBpE-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=PEMLhgnqBpE "Deep Reinforcement Learning for Path Planning of Autonomous Mobile Robots in Complicated Environments")


## Installation

### Requirements

1. **Ubuntu 20.04** 

2. **CUDA 11.8** 

3. **Torch 2.3** 

4. **Gazebo 11** 

4. **ROS Noetic** 
### Demo
Create your ROS workspace and conda environment

```bash
catkin_make
soure ./devel/setup.bash
```
You need to replace the path and package names in your folder.
The demo script can be run using the following format:

```bash
roslaunch Your ROS workspace main_slam.launch
```
