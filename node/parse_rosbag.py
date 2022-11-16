#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import os
import sys
import rospy
from time import perf_counter
import numpy as np
import cv2
from scipy.spatial import KDTree
import open3d as o3d
from tqdm import tqdm
from sensor_msgs.point_cloud2 import PointCloud2
from sensor_msgs.msg import CompressedImage, Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from py_src.ros_utils.cloud_publisher import CloudPublisher
from py_src.ros_utils.pose_publisher import PosePublisher
from py_src.misc.pcd_utils import make_open3d_point_cloud
from py_src.misc.plot_utils import visualize_pcds, visualize_points_list
from py_src.config import get_config
from py_src.ros_utils.image_utils import depth_msg_to_numpy, imgmsg_to_pil, depth2points, depth2xyzrgb, BackprojectDepth
from py_src.ros_utils.listener import Listener
from py_src.mapping import Mapping

def main():

    rospy.init_node('parse_rosbag', anonymous=True)
    rospy.loginfo(sys.executable)
    rospy.loginfo(sys.version)

    mapping = Mapping()

    mapping.mapping()

if __name__ == '__main__':

    main()