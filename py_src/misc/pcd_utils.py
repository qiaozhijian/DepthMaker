#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
def dbscan_cluster(pcd, eps=0.02, min_points=10):
    # np.darray, (N, ), each element is the cluster label(0~max), if -1, it is noise
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    id = np.unique(labels)
    clusters = []
    for i in id:
        if i == -1:
            continue
        cluster = pcd.select_by_index(np.where(labels == i)[0])
        clusters.append(cluster)
    return clusters

def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color / 255.0)
    return pcd

def split_lane_by_id(lane_points):
    lane_points = lane_points[lane_points[:, -1] >= 0]
    track_ids = np.unique(np.round(lane_points[:, -1]))
    # print('track_ids', track_ids)
    # category = np.unique(lane_points[:, -3])
    # print('category', category)
    lane_points_split = []
    for track_id in sorted(track_ids):
        lane_points_split.append(lane_points[np.round(lane_points[:, -1]) == track_id, :])
    return lane_points_split

def split_lane_by_category(lane_points):
    place = 3
    lane_points = lane_points[lane_points[:, place] > 0]
    track_ids = np.unique(np.round(lane_points[:, place]))
    lane_points_split = []
    for track_id in track_ids:
        lane_points_split.append(lane_points[np.round(lane_points[:, place]) == track_id, :])
    return lane_points_split

def transform_points(points, transform):
    # transform: (4, 4)
    # points: (N, 3)
    points_3d = points[:, :3]
    points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_3d = np.dot(transform, points_3d.T).T
    points[:, :3] = points_3d[:, :3]

    return points

def inv_se3(transform):
    # transform: (4, 4)
    # points: (N, 3)
    inv_transform = np.eye(4)
    inv_transform[:3, :3] = transform[:3, :3].T
    inv_transform[:3, 3] = -np.dot(transform[:3, :3].T, transform[:3, 3])
    return inv_transform


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) #黄色
    target_temp.paint_uniform_color([0, 0.651, 0.929]) #蓝色
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def icp_plane_registration(source, target, trans_init = np.eye(4), radius = 0.3, max_iter = 100):

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max_iter, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max_iter, max_nn=30))

    threshold = 2
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    draw_registration_result(source, target, reg_p2l.transformation)
    return reg_p2l.transformation