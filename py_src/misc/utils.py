#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def xyzquat_to_pose(xyz, quat):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(quat).as_matrix()
    pose[:3, 3] = xyz
    return pose

def get_K_mat(intrinsic):
    K = np.eye(3)
    K[0, 0] = intrinsic[0]
    K[1, 1] = intrinsic[1]
    K[0, 2] = intrinsic[2]
    K[1, 2] = intrinsic[3]
    return K

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    print('CUDNN initialized')
    torch.cuda.empty_cache()

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def so3_to_quat(rot):
    r = R.from_matrix(rot)
    return r.as_quat()

def polyline_fitting(xyz, interval=0.2, forward_axis='x'):
    # uniform interpolation, x forward
    # xyz: N x 3
    iter_num = 3
    if forward_axis == 'x':
        order = [0, 1, 2]
    elif forward_axis == 'y':
        order = [1, 2, 0]
    elif forward_axis == 'z':
        order = [2, 0, 1]
    while iter_num > 0:
        x_g, y_g, z_g = xyz[:, order[0]], xyz[:, order[1]], xyz[:, order[2]]

        f_yx = np.poly1d(np.polyfit(x_g,y_g,2))
        f_zx = np.poly1d(np.polyfit(x_g,z_g,2))
        y_fit = f_yx(x_g)
        z_fit = f_zx(x_g)
        xyz_fit = np.concatenate((x_g.reshape(-1,1), y_fit.reshape(-1,1), z_fit.reshape(-1,1)), axis=1)

        residual = np.linalg.norm(xyz_fit - xyz, axis=1)
        idx = np.where(residual > np.mean(residual) + 2 * np.std(residual))[0]
        xyz = np.delete(xyz, idx, axis=0)
        iter_num = iter_num - 1

    x_g = np.linspace(min(x_g), max(x_g), int((max(x_g) - min(x_g)) / interval))
    y_g = f_yx(x_g)
    z_g = f_zx(x_g)
    xyz = np.concatenate((x_g.reshape(-1,1), y_g.reshape(-1,1), z_g.reshape(-1,1)), axis=1)
    if forward_axis == 'x':
        xyz = xyz[:, [0, 1, 2]]
    elif forward_axis == 'y':
        xyz = xyz[:, [2, 0, 1]]
    elif forward_axis == 'z':
        xyz = xyz[:, [1, 2, 0]]

    return xyz