#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import rospy
import numpy as np
import cv2

from py_src.ros_utils.image_utils import depth_msg_to_numpy, imgmsg_to_pil, depth2points, depth2xyzrgb, BackprojectDepth
from py_src.ros_utils.pose_publisher import PosePublisher
from sensor_msgs.point_cloud2 import PointCloud2
from sensor_msgs.msg import CompressedImage, Image
from py_src.ros_utils.cloud_publisher import CloudPublisher

class Listener:
    def __init__(self):
        self.lidar_topic = rospy.get_param('/lidar_topic', '/velodyne_points')
        self.intrinsic = rospy.get_param('/l515/intrinsics')
        rospy.loginfo('pcl_topic: {}'.format(self.lidar_topic))
        self.depth_topic = '/ause/camera/aligned_depth_to_color/image_raw'
        self.pub_cloud = CloudPublisher(self.lidar_topic, frame='base_link')
        self.back_proj = BackprojectDepth(batch_size=1, height=480, width=640, intrinsic=self.intrinsic).cuda()
        self.depth_sub = rospy.Subscriber('/ause/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        print('init listener')

    def depth_callback(self, msg):
        depth_img = depth_msg_to_numpy(msg).astype(np.float32)
        points = self.back_proj(depth_img)
        # print(depth_img.shape)
        # points = depth2points(depth_img, self.intrinsic)
        # self.pub_cloud.publish_cloud_xyz(points, msg.header.stamp)
        print(points.shape)