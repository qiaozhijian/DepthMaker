# Depth Maker

This is the ROS package of generating depth ground truth images from RGBD images and LiDAR point clouds. 

To use this package, you need to install the following packages:
+ [OpenCV](https:://opencv.org)
+ [PCL](https://pointclouds.org/)
+ [ROS](http://wiki.ros.org/ROS/Installation)
+ [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
+ [camodocal](https://github.com/hengli/camodocal)
+ [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
+ [VINS-RGBD](https://github.com/STAR-Center/VINS-RGBD)
+ [R3LIVE](https://github.com/hku-mars/r3live)

And prepare according mapping data using the above packages and save them in the corresponding folders. To generate depth ground truth images, you need to run the following commands:
```
./gen_depth_dataset ../../../src/DepthMaker/config/hku_campus_seq_00.yaml
./gen_depth_dataset ../../../src/DepthMaker/config/hkust_campus_seq_00.yaml
./gen_depth_dataset ../../../src/DepthMaker/config/hkust_campus_seq_01.yaml
./gen_depth_dataset ../../../src/DepthMaker/config/hkust_campus_seq_02.yaml
./gen_depth_dataset ../../../src/DepthMaker/config/project_2022-11-12-23-13-10.yaml
./gen_depth_dataset ../../../src/DepthMaker/config/project_2022-11-12-23-44-15.yaml
./gen_depth_dataset ../../../src/DepthMaker/config/project_2022-11-18-00-02-32.yaml
```