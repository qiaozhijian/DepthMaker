#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import yaml
import open3d as o3d
from scipy.spatial import KDTree

import rospy
import rospkg
from sensor_msgs.point_cloud2 import PointCloud2
from sensor_msgs.msg import CompressedImage, Image

from py_src.ros_utils.image_utils import depth_msg_to_numpy, imgmsg_to_pil, depth2points, depth2xyzrgb, BackprojectDepth
from py_src.ros_utils.pose_publisher import PosePublisher
from py_src.ros_utils.cloud_publisher import CloudPublisher
from py_src.ros_utils.bag_util import BagReader, BagWriter
from py_src.misc.utils import get_K_mat, xyzquat_to_pose
from py_src.misc.plot_utils import visualize_points_list, visualize_pcd, draw_registration_result
from py_src.misc.pcd_utils import make_open3d_point_cloud, inv_se3

class Mapping:
    def __init__(self):
        self.read_params()
        self.init_pub_sub()
        self.back_proj = BackprojectDepth(batch_size=1, height=480, width=640).cuda()
        self.bag_reader = BagReader(self.bag_path, enable=True)
        self.bag_writer = BagWriter(self.new_bag_path, enable=False)

        self.global_map = o3d.geometry.PointCloud()
        print('Init mapping')

    def read_params(self):
        self.base_dir = rospkg.RosPack().get_path('rgbd_inertial_slam')
        self.config = yaml.load(open(os.path.join(self.base_dir, 'config', 'template.yaml'), 'r'), Loader=yaml.FullLoader)
        self.lidar_topic = self.config['lidar_topic']
        self.map_topic = self.config['map_topic']
        self.K = get_K_mat(self.config['l515']['intrinsics']) # 3x3
        self.depth_topic = os.path.join('/ause', self.config['l515']['depth_topic'])
        self.trans_i_c = np.asarray(self.config['l515']['body_T_cam0']).reshape(4, 4)
        self.rot_diff = np.eye(4)
        self.rot_diff[:3,:3] = np.asarray(self.config['rot_diff']).reshape(3, 3)

        self.bag_path = rospy.get_param('/bag_path', '/home/zhijian/Downloads/2021-03-18-15-30-00.bag')
        bag_name = os.path.basename(self.bag_path).replace('.bag', '')
        self.new_bag_path = os.path.join(os.path.dirname(self.bag_path), bag_name + '_parsed.bag')

        self.vio = np.loadtxt(os.path.join(self.base_dir, 'vins_outputs', 'vio.csv'), delimiter=',', usecols=range(11))
        self.cam_poses = self.I_to_C(self.vio, self.trans_i_c)
        self.loop_vio = np.loadtxt(os.path.join(self.base_dir, 'vins_outputs', 'vio_loop.csv'), delimiter=',', usecols=range(8))

        rospy.loginfo('config: {}'.format(self.config))

    def I_to_C(self, vio, trans_i_c):
        # vio: [t, x, y, z, qx, qy, qz, qw, v_x, v_y, v_z]
        cam_poses = {}
        points = np.zeros((vio.shape[0],3))
        for i in range(vio.shape[0]):
            t = vio[i, 0] / 1e9
            T_i = xyzquat_to_pose(vio[i, 1:4], vio[i, 4:8])
            T_c = np.dot(T_i, trans_i_c)
            cam_poses[t] = inv_se3(self.rot_diff) @ T_c
            points[i,:] = cam_poses[t][:3,3].reshape(3)
        # visualize_points_list([points])
        return cam_poses

    def init_pub_sub(self):
        self.pub_cloud = CloudPublisher(self.map_topic, frame='map')

    def add_points(self, points, pose, voxel_size=0.05):
        # points: Nx3
        # pose: 4x4, T_w_c
        points = np.dot(points, pose[:3, :3].T) + pose[:3, 3]
        map_points = np.asarray(self.global_map.points)
        if map_points.shape[0] > 0:
            self.global_map.points = o3d.utility.Vector3dVector(np.vstack([map_points, points]))
        else:
            self.global_map.points = o3d.utility.Vector3dVector(points)
        self.global_map = self.global_map.voxel_down_sample(voxel_size=voxel_size)

        # cl, ind = self.global_map.remove_radius_outlier(nb_points=32, radius=0.5)
        # cl, ind = self.global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        # self.global_map = self.global_map.select_by_index(ind)

    def icp_odometry(self, points, Twc = np.eye(4)):
        source = make_open3d_point_cloud(points[:,:3])
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        self.global_map.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, self.global_map, 2.0, Twc,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        # print('est: \n', reg_p2l.transformation)
        # draw_registration_result(source, self.global_map, Twc)
        # draw_registration_result(source, self.global_map, reg_p2l.transformation)
        return reg_p2l.transformation

    def publish_all(self):
        map_points = np.asarray(self.global_map.points)
        self.pub_cloud.publish_cloud_xyz(map_points, rospy.Time.now())

    def mapping(self):
        topics = ["/camera/aligned_depth_to_color/image_raw"]
        pose_times = sorted(self.cam_poses.keys())
        tKDTree = KDTree(np.array(pose_times).reshape(-1, 1))
        last_Twc, cur_Twc, last_vio, cur_vio = None, None, None, None
        for i, (topic, msg, t) in enumerate(self.bag_reader.bag.read_messages(topics=topics)):
            if topic == "/camera/aligned_depth_to_color/image_raw":
                # if i%10 !=0:
                #     continue
                time_sec = msg.header.stamp.to_sec()
                dist, ind = tKDTree.query(np.array([time_sec]).reshape(-1, 1), k=1)
                if abs(dist[0]) > 0.1:
                    continue
                depth_img = depth_msg_to_numpy(msg).astype(np.float32)
                points = self.back_proj(depth_img, self.K)
                cur_vio = self.cam_poses[pose_times[ind[0]]]
                if cur_Twc is None:
                    cur_Twc = cur_vio
                else:
                    init_Twc = last_Twc @ inv_se3(last_vio) @ cur_vio
                    cur_Twc = self.icp_odometry(points, init_Twc)

                self.add_points(points, cur_Twc, voxel_size=0.01)
                if i % 10 == 0:
                    self.publish_all()
                last_vio = cur_vio
                last_Twc = cur_Twc
                if i > 500:
                    break
        visualize_pcd(self.global_map, axis_marker=False)


    def write_new_bag(self, msg):
        # topics = ["/camera/color/image_raw/compressed", "/camera/aligned_depth_to_color/image_raw", "/camera/imu"]
        for topic, msg, t in self.bag_reader.bag.read_messages():
            self.bag_writer.write(topic, msg, msg.header.stamp)
            if topic == "/camera/aligned_depth_to_color/image_raw":
                depth_img = depth_msg_to_numpy(msg).astype(np.float32)
                points = self.back_proj(depth_img, self.intrinsic)
                self.bag_writer.write_pcd_xyz(self.lidar_topic, points, msg.header.stamp, frame_id = 'base_link')
                print('write pcd to bag at {}'.format(msg.header.stamp.to_sec()))
        self.bag_writer.close()