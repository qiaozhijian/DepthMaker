#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import os
import glob
import cv2
import numpy as np
import open3d as o3d
from py_src.misc.plot_utils import visualize_pcds
from py_src.misc.pcd_utils import icp_plane_registration, draw_registration_result

if __name__ == '__main__':

    pcd_file = "/media/qzj/Extreme SSD/datasets/CV_Project/r3live/hkust_campus_seq_02/rgb_pt.pcd"
    pcd = o3d.io.read_point_cloud(pcd_file)
    ex_0c = np.asarray([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
    ex_1c = np.asarray([[0.00735877, 0.999971, 0.00171312, 0.0830676],
                        [-0.00520647, 0.00175146, -0.999985, 0.000456845],
                        [-0.999959, 0.00734974, 0.00521921, -0.0841266],
                        [0, 0, 0, 1]])
    ex_2c = np.asarray([[-0.00343483, -0.999979, -0.00551894, -0.0575728],
                        [0.0132888, 0.00547284, -0.999897, -0.000133631],
                        [0.999906, -0.00350781, 0.0132698, -0.0577581],
                        [0, 0, 0, 1]])
    ex_3c = np.asarray([[-0.999988, 0.00325354, -0.00355828, -0.0260468],
                        [0.00356684, 0.0026289, -0.99999, 0.000267131],
                        [-0.00324415, -0.999991, -0.00264047, -0.140323],
                        [0, 0, 0, 1]])

    pcd0 = pcd.transform(ex_0c).voxel_down_sample(voxel_size=1)
    pcd1 = pcd.transform(ex_1c).voxel_down_sample(voxel_size=1)
    pcd2 = pcd.transform(ex_2c).voxel_down_sample(voxel_size=1)
    pcd3 = pcd.transform(ex_3c).voxel_down_sample(voxel_size=1)

    visualize_pcds([pcd0])
    visualize_pcds([pcd1])
    visualize_pcds([pcd2])
    visualize_pcds([pcd3])

    # base_dir = "/media/qzj/Extreme SSD/datasets/CV_Project/omni_depth/project_2022-11-23-14-33-57_debug/rs_dep_rgb"
    # points_file = glob.glob(os.path.join(base_dir, "*.pcd"))
    # points_file.sort()
    # poses_file = glob.glob(os.path.join(base_dir, "*.txt"))
    # poses_file.sort()
    #
    # source = o3d.io.read_point_cloud(points_file[0])
    # pose_source = np.loadtxt(poses_file[0]).reshape(4, 4)
    #
    # for i in range(1, len(points_file), 10):
    #     target = o3d.io.read_point_cloud(points_file[i])
    #     pose_target = np.loadtxt(poses_file[i]).reshape(4, 4)
    #     rel_pose = np.dot(np.linalg.inv(pose_target), pose_source)
    #
    #     draw_registration_result(source, target, rel_pose)
        # est_rel_pose = icp_plane_registration(source, target, np.eye(4), 0.3)

        # print("rel_pose", rel_pose)
        # print("est_rel_pose", est_rel_pose)