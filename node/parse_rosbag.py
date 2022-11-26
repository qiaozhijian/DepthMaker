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
import rosbag

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
from py_src.ros_utils.image_utils import gene_cicle_mask

def main():

    rospy.init_node('parse_rosbag', anonymous=True)
    rospy.loginfo(sys.executable)
    rospy.loginfo(sys.version)

    mapping = Mapping()

    # mapping.mapping()
    mapping.fetch_image(t = 1668267872.589593)

if __name__ == '__main__':

    bag = rosbag.Bag("/media/qzj/Extreme SSD/datasets/CV_Project/trainning_data/project_2022-11-12-23-44-15.bag", 'r')
    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw/compressed']):
        time_sec = float("%.6f" %  msg.header.stamp.to_sec())
        if time_sec > 1668267872.589593 - 0.01 and time_sec < 1668267872.589593 + 0.01:
            print(time_sec)
            rgb = np.array(imgmsg_to_pil(msg))
            cv2.imwrite("test.png", rgb)
            cv2.waitKey()
    # main()