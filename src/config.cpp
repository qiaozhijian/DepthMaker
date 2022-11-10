//
// Created by qzj on 2021/4/25.
//

#include "config.h"
#include "CommonFunc.h"

YAML::Node Config::config_node_ = YAML::Node();
std::string Config::lidar_topic;

Eigen::Matrix4d Config::imu_to_lidar_;
Eigen::Matrix4d Config::lidar_to_imu_;

std::string Config::init_method_;


template <typename T>
Eigen::Matrix<T,4,4> EigenIsoInv(const Eigen::Matrix<T,4,4> &Tcw) {
    Eigen::Matrix<T,3,3> Rcw = Tcw.block(0, 0, 3, 3);
    Eigen::Matrix<T,3,1> tcw = Tcw.block(0, 3, 3, 1);
    Eigen::Matrix<T,3,3> Rwc = Rcw.transpose();
    Eigen::Matrix<T,3,1> twc = -Rwc * tcw;

    Eigen::Matrix<T,4,4> Twc = Eigen::Matrix<T,4,4>::Identity();

    Twc.block(0, 0, 3, 3) = Rwc;
    Twc.block(0, 3, 3, 1) = twc;

    return Twc;
}

void Config::readConfig(){

    config_node_ = YAML::LoadFile(rgbd_inertial_slam::MATCH_YAML_PATH);

    lidar_topic = config_node_["lidar_topic"].as<std::string>();

    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 4; j++) {
            imu_to_lidar_(i, j) = config_node_["imu_to_lidar"][4 * i + j].as<float>();
        }

    lidar_to_imu_ = EigenIsoInv(imu_to_lidar_);
}