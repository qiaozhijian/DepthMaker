#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from random import shuffle
import copy
import os
import glob
from py_src.misc.pcd_utils import split_lane_by_id, make_open3d_point_cloud

colors_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0],
               [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
               [0.25, 0.25, 0.25], [0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25], [0.25, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0.25],
               [0.75, 0.75, 0.75], [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.75, 0.75, 0], [0.75, 0, 0.75], [0, 0.75, 0.75],
               [0.125, 0.125, 0.125], [0.125, 0, 0], [0, 0.125, 0], [0, 0, 0.125], [0.125, 0.125, 0], [0.125, 0, 0.125], [0, 0.125, 0.125],
               [0.375, 0.375, 0.375], [0.375, 0, 0], [0, 0.375, 0], [0, 0, 0.375], [0.375, 0.375, 0], [0.375, 0, 0.375], [0, 0.375, 0.375],
               [0.625, 0.625, 0.625], [0.625, 0, 0], [0, 0.625, 0], [0, 0, 0.625], [0.625, 0.625, 0], [0.625, 0, 0.625], [0, 0.625, 0.625],
               [0.875, 0.875, 0.875], [0.875, 0, 0], [0, 0.875, 0], [0, 0, 0.875], [0.875, 0.875, 0], [0.875, 0, 0.875], [0, 0.875, 0.875],]

def visualize_points(points):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])

    pcds = [pcd]
    axis_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    pcds.append(axis_marker)
    o3d.visualization.draw_geometries(pcds)

def visualize_pcd(pcd, axis_marker = True):
    pcd.paint_uniform_color([1, 0.706, 0])
    pcds = [pcd]
    if axis_marker:
        axis_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
        pcds.append(axis_marker)
    o3d.visualization.draw_geometries(pcds)

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

def visualize_points_list(points_list, extra_pcd = None, axis_marker = True):

    num = len(colors_list)
    pcds = []
    for i, points in enumerate(points_list):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        pcd.paint_uniform_color(colors_list[i%num])
        pcds.append(pcd)
    if axis_marker:
        axis_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
        pcds.append(axis_marker)
    if extra_pcd is not None:
        pcds.append(extra_pcd)

    o3d.visualization.draw_geometries(pcds)

def visualize_points_list_size(points_list):

    num = len(colors_list)
    pcds = []

    for i, points in enumerate(points_list):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        pcd.paint_uniform_color(colors_list[i%num])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 9.0
        pcds.append({'name': 'pcd_{}'.format(i), 'geometry': pcd, 'material': mat})

    axis_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    pcds.append(axis_marker)
    # o3d.visualization.draw_geometries(pcds)
    o3d.visualization.draw(pcds, show_skybox=False)

def visualize_pcds(pcds_list, axis_marker = True):
    num = len(colors_list)
    pcds = []
    for i, pcd in enumerate(pcds_list):
        pcd.normals = o3d.utility.Vector3dVector([])
        if np.asarray(pcd.colors).shape[0] == 0:
            pcd.paint_uniform_color(colors_list[i%num])
        pcds.append(pcd)
    if axis_marker:
        axis_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
        pcds.append(axis_marker)
    o3d.visualization.draw_geometries(pcds)

def pointcloud_to_spheres(points, color=[0.7, 0.1, 0.1], sphere_size=0.6):
    spheres = o3d.geometry.TriangleMesh()
    s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    for i, p in enumerate(points):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p[:3]
        si.transform(trans)
        spheres += si
    return spheres

def check_association():

    file_path = os.path.dirname(os.path.abspath(__file__))
    visualization_path = os.path.join(file_path, '../visualization')
    for directionary in sorted(glob(os.path.join(visualization_path, '*0'))):
        pcds = []
        ins_pcd = {}
        for file_path in sorted(glob(os.path.join(directionary, '*.pcd'))):
            pcd = o3d.io.read_point_cloud(file_path)
            normal = np.asarray(pcd.normals)
            assert normal[:,0].mean()==normal[0,0] and normal[:,1].mean()==normal[0,1]
            ins = (normal[0,0], normal[0,1])
            xyz = np.asarray(pcd.points)
            if 'cluster' in file_path:
                xyz[:,2] = xyz[:,2] + np.ones(shape=(xyz.shape[0])) * 5
            if ins in ins_pcd.keys():
                ins_pcd[ins] = np.vstack([ins_pcd[ins], xyz])
            else:
                ins_pcd[ins] = xyz

        for ins, points in ins_pcd.items():
            pcd = make_open3d_point_cloud(points)
            pcds.append(pcd)

        visualize_pcds(pcds)
        # print(file)
        # pcd = o3d.io.read_point_cloud(file)
        # o3d.visualization.draw_geometries([pcd])
    # print(file_path)
    return