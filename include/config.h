//
// Created by qzj on 2021/4/25.
//

#ifndef SRC_CONFIG_H
#define SRC_CONFIG_H

#include <yaml-cpp/yaml.h>
#include "global_defination/global_defination.h"
#include <Eigen/Core>
#include <vector>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>      // std::ifstream
#include <stdio.h>
#include <unistd.h>

class Config {

public:

    Config(){}
    static void readConfig();

public:

    static std::string lidar_topic;

    static Eigen::Matrix4d imu_to_lidar_;
    static Eigen::Matrix4d lidar_to_imu_;

    static std::string init_method_;

    static YAML::Node config_node_;
};


#endif //SRC_CONFIG_H
