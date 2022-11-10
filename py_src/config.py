#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import argparse

def get_config():
    parser = argparse.ArgumentParser(description='ROS package template')
    parser.add_argument('--dataset_name', type=str, default='', help='the dataset name')
    parser.add_argument('--batch', type=int, default=32, help='batch_size')

    args, unknown = parser.parse_known_args()

    return args