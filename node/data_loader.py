#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk

import os
import glob
import cv2
import numpy as np

if __name__ == '__main__':

    # depth_re = "/media/qzj/Extreme SSD/datasets/CV_Project/omni_depth/project_2022-11-12-23-13-10-small/cam1/depth/1668266579.234421_re.tiff"
    # depth = "/media/qzj/Extreme SSD/datasets/CV_Project/omni_depth/project_2022-11-12-23-13-10-small/cam1/depth/1668266579.234421.tiff"
    omni = "/media/qzj/Extreme SSD/datasets/CV_Project/omni_depth/project_2022-11-12-23-13-10-small_debug/omni/1668266579.234421.tiff"
    omni = cv2.imread(omni, cv2.IMREAD_UNCHANGED)


    print(omni[158, 371])
    # # print("depth_re: ", depth_re)
    # # print("depth: ", depth)
    # depth_re = cv2.imread(depth_re, cv2.IMREAD_UNCHANGED)
    # depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
    # diff = depth_re - depth
    # print(depth-depth_re)
    # cv2.imshow('depth', depth)
    # cv2.waitKey(0)
    # cv2.imshow('depth_re', depth_re)
    # cv2.waitKey(0)
