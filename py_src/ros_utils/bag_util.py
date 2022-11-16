#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import cv2
import rospy
import rosbag
import os
from nav_msgs.msg import Odometry
import std_msgs
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import KDTree

from py_src.misc.utils import mkdirs, so3_to_quat
from py_src.ros_utils.cloud_publisher import create_cloud_xyzcvt32, create_cloud_xyz32
from py_src.ros_utils.image_utils import depth_msg_to_numpy, imgmsg_to_pil, depth2points, depth2xyzrgb, BackprojectDepth
from py_src.misc.plot_utils import make_open3d_point_cloud

class BagReader:
    def __init__(self, bag_path, enable = True):
        self.bag_path = bag_path
        self.bag = rosbag.Bag(self.bag_path, 'r')
        self.enable = enable
        # self.back_proj = BackprojectDepth(batch_size=1, height=480, width=640, intrinsic=intrinsic).cuda()

    def read_bag(self, topic_names):
        depth_list = {}
        rgb_list = {}
        for topic, msg, t in self.bag.read_messages(topics=topic_names):
            time_sec = float("%.6f" %  msg.header.stamp.to_sec())
            if topic == "/camera/aligned_depth_to_color/image_raw":
                depth = depth_msg_to_numpy(msg, scale=0.00025)
                depth_list[time_sec] = depth
            elif topic == "/camera/color/image_raw/compressed":
                rgb = np.array(imgmsg_to_pil(msg))
                rgb_list[time_sec] = rgb
            elif topic == "/camera/imu":
                pass
        print("finish reading bag file")

        rgb_times = sorted(rgb_list.keys())
        rgb_tKDTree = KDTree(np.array(rgb_times).reshape(-1, 1))
        cnt = 0
        for time_sec in depth_list.keys():
            cnt += 1
            if cnt % 100 != 0:
                continue
            depth = depth_list[time_sec]
            # find the corresponding rgb image
            rgb_time = rgb_times[rgb_tKDTree.query(np.array([time_sec]).reshape(-1, 1), k=1)[1][0]]
            rgb = rgb_list[rgb_time]
            xyzrgb = depth2xyzrgb(depth, rgb, intrinsic)
            pcd_rgb = make_open3d_point_cloud(xyzrgb[:, :3], xyzrgb[:, 3:])
            o3d.visualization.draw_geometries([pcd_rgb])

    def close(self):
        self.bag.close()

class BagWriter:
    def __init__(self, bag_path, enable = True):
        self.bag_path = bag_path
        dir_path = os.path.dirname(self.bag_path)
        mkdirs(dir_path)
        self.bag = rosbag.Bag(self.bag_path, 'w')
        self.enable = enable

    def write_odom(self, rotation, translation, topic_name, timestamp_micros):
        # pose: 4x4
        # timestamp: us
        if not self.enable:
            return
        odom = Odometry()
        odom.header.stamp = rospy.Time.from_seconds(timestamp_micros / 1e6)
        odom.header.frame_id = 'map'
        odom.pose.pose.position.x = translation[0]
        odom.pose.pose.position.y = translation[1]
        odom.pose.pose.position.z = translation[2]
        rotation = so3_to_quat(rotation)
        odom.pose.pose.orientation.x = rotation[0]
        odom.pose.pose.orientation.y = rotation[1]
        odom.pose.pose.orientation.z = rotation[2]
        odom.pose.pose.orientation.w = rotation[3]
        self.bag.write(topic_name, odom, odom.header.stamp)

    def write_pcd_xyzcvt(self, points, topic_name, timestamp_micros):
        # points: Nx6 numpy.darray or list
        if not self.enable:
            return
        stamp = rospy.Time.from_seconds(timestamp_micros / 1e6)
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = 'camera_body'
        cloud = create_cloud_xyzcvt32(header, points)
        self.bag.write(topic_name, cloud, stamp)

    def write_pcd_xyz(self, topic_name, points, stamp, frame_id):
        # points: Nx3 numpy.darray or list
        if not self.enable:
            return
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = frame_id
        cloud = create_cloud_xyz32(header, points)
        self.bag.write(topic_name, cloud, stamp)

    def write(self, topic_name, msg, stamp):
        if not self.enable:
            return
        self.bag.write(topic_name, msg, stamp)

    def close(self):
        self.bag.close()