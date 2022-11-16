#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

from io import BytesIO   # python3

import torch
import sys
import array

import numpy as np
from PIL import Image
from PIL import ImageOps
from cv_bridge import CvBridge
import torch.nn as nn

def imgmsg_to_pil(img_msg, rgba=True):
    # ROS压缩图像格式转换为pil图像格式
    try:
        if img_msg._type == 'sensor_msgs/CompressedImage':
            pil_img = Image.open(BytesIO(img_msg.data))   # python3
            # pil_img = Image.open(StringIO(img_msg.data))  # python2
            if pil_img.mode != 'L':
                pil_img = pil_bgr2rgb(pil_img)
            pil_mode = 'RGB'
        else:
            alpha = False
            pil_mode = 'RGB'
            if img_msg.encoding == 'mono8':
                mode = 'L'
            elif img_msg.encoding == 'rgb8':
                mode = 'RGB'
            elif img_msg.encoding == 'bgr8':
                mode = 'BGR'
            elif img_msg.encoding in ['bayer_rggb8', 'bayer_bggr8', 'bayer_gbrg8', 'bayer_grbg8']:
                mode = 'L'
            elif img_msg.encoding in ['bayer_rggb16', 'bayer_bggr16', 'bayer_gbrg16', 'bayer_grbg16']:
                pil_mode = 'I;16'
                if img_msg.is_bigendian:
                    mode='I;16B'
                else:
                    mode='I;16L'
            elif img_msg.encoding == 'mono16' or img_msg.encoding == '16UC1':
                pil_mode = 'F'
                if img_msg.is_bigendian:
                    mode = 'F;16B'
                else:
                    mode = 'F;16'
            elif img_msg.encoding == 'rgba8':
                mode = 'BGR'
                alpha = True
            elif img_msg.encoding == 'bgra8':
                mode = 'RGB'
                alpha = True
            else:
                raise Exception("Unsupported image format: %s" % img_msg.encoding)
            pil_img = Image.frombuffer(
                pil_mode, (img_msg.width, img_msg.height), img_msg.data, 'raw', mode, 0, 1)

        # 16 bits conversion to 8 bits
        if pil_mode == 'I;16':
            pil_img = pil_img.convert('I').point(lambda i: i * (1. / 256.)).convert('L')
        if pil_img.mode == 'F':
            pil_img = pil_img.point(lambda i: i * (1. / 256.)).convert('L')
            pil_img = ImageOps.autocontrast(pil_img)

        if rgba and pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')

        return pil_img

    except Exception as ex:
        print('Can\'t convert image: %s' % ex, file=sys.stderr)
        return None


def pil_bgr2rgb(pil_img):
    rgb2bgr = (0, 0, 1, 0,
               0, 1, 0, 0,
               1, 0, 0, 0)
    return pil_img.convert('RGB', rgb2bgr)

def depth2points(depth, intrinsic):
    # 将深度图转换为点云
    # input: depth, intrinsic[4]
    # output: points[N, 3]
    K = np.eye(3)
    K[0, 0] = intrinsic[0]
    K[1, 1] = intrinsic[1]
    K[0, 2] = intrinsic[2]
    K[1, 2] = intrinsic[3]
    points = []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            z = depth[v, u]
            if z > 0:
                x = (u - K[0, 2]) * z / K[0, 0]
                y = (v - K[1, 2]) * z / K[1, 1]
                points.append([x, y, z])
    points = np.array(points)
    return points

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, K):
        inv_K = torch.from_numpy(np.linalg.inv(K)).unsqueeze(0).cuda().float()
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        depth = torch.from_numpy(depth).float().cuda()
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = cam_points[0].transpose(0, 1)
        cam_points = cam_points[cam_points[:, 2] > 0, :]
        return cam_points.detach().cpu().numpy()

    def get_inv_K(self, intrinsic):
        K = np.eye(3)
        K[0, 0] = intrinsic[0]
        K[1, 1] = intrinsic[1]
        K[0, 2] = intrinsic[2]
        K[1, 2] = intrinsic[3]
        inv_K = torch.from_numpy(np.linalg.inv(K)).unsqueeze(0).cuda().float()
        return inv_K

def depth2xyzrgb(depth, rgb, intrinsic):
    # 将深度图转换为点云
    # input: depth, intrinsic[4]
    # output: points[N, 6]
    K = np.eye(3)
    K[0, 0] = intrinsic[0]
    K[1, 1] = intrinsic[1]
    K[0, 2] = intrinsic[2]
    K[1, 2] = intrinsic[3]
    points = []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            z = depth[v, u]
            if z > 0:
                x = (u - K[0, 2]) * z / K[0, 0]
                y = (v - K[1, 2]) * z / K[1, 1]
                points.append([x, y, z, rgb[v, u, 0], rgb[v, u, 1], rgb[v, u, 2]])
    points = np.array(points)
    return points

def depth_msg_to_numpy(msg, scale=0.00025):
    # ROS消息转换为numpy数组
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg,"passthrough") * scale
    return img