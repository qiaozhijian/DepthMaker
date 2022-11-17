//
// Created by qzj on 2021/12/24.
//

#include "System.h"

namespace rgbd_inertial_slam {

System::System(ros::NodeHandle &nh){

    LoadParameters(nh);

    InitBag(nh);

    TravelBag();

//    一些系统初始化操作，比如初始化变量，新建线程
//    config_node_ = Config::config_node_;
////        接收原始点云信息
//    cloud_sub_ptr_ = std::make_shared<CloudSubscriber>(nh, config_node_["lidar_topic"].as<std::string>(), 5);

    LOG(INFO)<<"System init!";

}

}