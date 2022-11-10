#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import os
import sys
import rospy
from time import perf_counter
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from py_src.ros_utils.cloud_publisher import CloudPublisher
from py_src.ros_utils.pose_publisher import PosePublisher
from py_src.config import get_config

if __name__ == '__main__':

    rospy.init_node('parse_rosbag', anonymous=True)
    rospy.loginfo(sys.executable)
    rospy.loginfo('parse_rosbag node started')

    args = get_config()
    rosparams = rospy.get_param_names()
    pcl_topic = rospy.get_param('/lidar_topic', '/velodyne_points')
    rospy.loginfo('pcl_topic: {}'.format(pcl_topic))
    pub_cloud_pred = CloudPublisher(pcl_topic, frame='map')
    pub_pose = PosePublisher('/pose_world')

    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        Twc = np.eye(4)
        pub_pose.publish_pose_stamped(Twc, rospy.Time.now())

        pcl = np.random.rand(100, 3)
        pub_cloud_pred.publish_cloud_xyz(pcl[:, :3], rospy.Time.now())

        rate.sleep()