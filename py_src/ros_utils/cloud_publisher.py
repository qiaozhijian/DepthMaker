#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import rospy
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2
import numpy as np
import std_msgs
from sensor_msgs.point_cloud2 import PointField, create_cloud, create_cloud_xyz32

def create_cloud_xyzcvt32(header, points):
    points = np.array(points, dtype=np.float32)
    fields = [PointField(name='x', offset=0,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='category', offset=12,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='visibilty', offset=16,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='track_id', offset=20,
                         datatype=PointField.FLOAT32, count=1)]
    return create_cloud(header, fields, points)

def create_cloud_xyzc32(header, points):

    points = np.array(points, dtype=np.float32)
    fields = [PointField(name='x', offset=0,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8,
                         datatype=PointField.FLOAT32, count=1),
              PointField(name='category', offset=12,
                         datatype=PointField.FLOAT32, count=1)]
    return create_cloud(header, fields, points)

def find_decimal(array):
    decimal = 0
    for i in array:
        if i % 1 != 0:
            decimal += 1
    return decimal

class CloudPublisher:

    def __init__(self, topic, frame = 'camera_body'):
        self.pcl_pub = rospy.Publisher(topic, PointCloud2, queue_size=1000)
        self.frame = frame

    def publish_cloud_xyzcvt(self, cloud_points, timestamp):
        # points: Nx6 numpy.darray or list
        if cloud_points.shape[1] == 4:
            cloud_points = np.hstack((cloud_points, np.zeros((cloud_points.shape[0], 2))))
        assert cloud_points.shape[1] == 6
        assert len(cloud_points.shape) == 2
        #create pcl from points
        header = std_msgs.msg.Header()
        header.stamp = timestamp
        header.frame_id = self.frame
        lane_points = create_cloud_xyzcvt32(header, cloud_points)

        self.pcl_pub.publish(lane_points)

    def publish_cloud_xyzc(self, cloud_points, timestamp):
        # points: Nx4 numpy.darray or list
        assert cloud_points.shape[1] == 4
        assert len(cloud_points.shape) == 2
        #create pcl from points
        header = std_msgs.msg.Header()
        header.stamp = timestamp
        header.frame_id = self.frame
        lane_points = create_cloud_xyzc32(header, cloud_points)

        self.pcl_pub.publish(lane_points)

    def publish_cloud_xyz(self, cloud_points, timestamp):
        # points: Nx3 numpy.darray or list
        assert cloud_points.shape[1] == 3
        assert len(cloud_points.shape) == 2
        #create pcl from points
        header = std_msgs.msg.Header()
        header.stamp = timestamp
        header.frame_id = self.frame
        lane_points = create_cloud_xyz32(header, cloud_points)

        self.pcl_pub.publish(lane_points)
