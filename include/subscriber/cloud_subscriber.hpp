/*
 * @Description: 订阅激光点云信息，并解析数据
 * @Author: Ren Qian
 * @Date: 2020-02-05 02:27:30
 */

#ifndef LIDAR_LOCALIZATION_SUBSCRIBER_CLOUD_SUBSCRIBER_HPP_
#define LIDAR_LOCALIZATION_SUBSCRIBER_CLOUD_SUBSCRIBER_HPP_

#include <deque>
#include <mutex>
#include <thread>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "sensor_data/cloud_data.hpp"

namespace rgbd_inertial_slam {
    class CloudSubscriber {
    public:
        CloudSubscriber(ros::NodeHandle &nh, std::string topic_name, size_t buff_size, bool use_lidar_time = true, bool use_ring = false);

        CloudSubscriber() = default;

        void ParseData(std::deque<CloudData> &deque_cloud_data);

        void ParseData(std::deque<CloudRingData> &deque_cloud_data);

    private:
        void msg_callback(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg_ptr);

    private:
        ros::NodeHandle nh_;
        ros::Subscriber subscriber_;
        std::deque<CloudData> new_cloud_data_;
        std::deque<CloudRingData> new_cloud_ring_data_;
        bool use_lidar_time_;
        bool use_ring_;
        std::mutex buff_mutex_;
    };
}

#endif