//
// Created by qzj on 2021/12/24.
//

#ifndef CMAKE_TEMPLATE_SYSTEM_H
#define CMAKE_TEMPLATE_SYSTEM_H

#include <ros/ros.h>
// subscriber
#include "subscriber/cloud_subscriber.hpp"
#include "config.h"


namespace rgbd_inertial_slam {

    class System {

        public:

        System(ros::NodeHandle &nh);

        // subscriber
        std::shared_ptr<CloudSubscriber> cloud_sub_ptr_;

        private:
        //config
        YAML::Node config_node_;
    };
}



#endif //CMAKE_TEMPLATE_SYSTEM_H
