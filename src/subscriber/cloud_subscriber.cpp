/*
 * @Description: 订阅激光点云信息，并解析数据
 * @Author: Ren Qian
 * @Date: 2020-02-05 02:27:30
 */

#include "subscriber/cloud_subscriber.hpp"

#include "glog/logging.h"

namespace rgbd_inertial_slam {
    CloudSubscriber::CloudSubscriber(ros::NodeHandle &nh, std::string topic_name, size_t buff_size, bool use_lidar_time, bool use_ring)
            : nh_(nh), use_lidar_time_(use_lidar_time), use_ring_(use_ring) {
        subscriber_ = nh_.subscribe(topic_name, buff_size, &CloudSubscriber::msg_callback, this);
    }

    void CloudSubscriber::msg_callback(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg_ptr) {
        buff_mutex_.lock();
        if (!use_ring_)
        {
            CloudData cloud_data;
            cloud_data.time = cloud_msg_ptr->header.stamp.toSec();
//        LOG(INFO) << std::fixed << "cloud time: "<<cloud_data.time << ", cur: " << ros::Time::now().toSec();
//用当前时间戳而不是雷达的话，会有20-60ms延迟
            if(!use_lidar_time_)
                cloud_data.time = ros::Time::now().toSec();

            pcl::fromROSMsg(*cloud_msg_ptr, *(cloud_data.cloud_ptr));
            new_cloud_data_.push_back(cloud_data);
        }

        if(use_ring_){
            CloudRingData cloud_ring_data;
            cloud_ring_data.time = cloud_msg_ptr->header.stamp.toSec();
            if(!use_lidar_time_)
                cloud_ring_data.time = ros::Time::now().toSec(); // Ouster lidar and pandar users may need to uncomment this line

            pcl::fromROSMsg(*cloud_msg_ptr, *(cloud_ring_data.cloud_ptr));
            new_cloud_ring_data_.push_back(cloud_ring_data);
        }
        LOG(INFO) << "new_cloud_data_.size(): " << new_cloud_data_.size();
        buff_mutex_.unlock();
    }

    void CloudSubscriber::ParseData(std::deque<CloudData> &cloud_data_buff) {
        buff_mutex_.lock();
        if (new_cloud_data_.size() > 0) {
            cloud_data_buff.insert(cloud_data_buff.end(), new_cloud_data_.begin(), new_cloud_data_.end());
            new_cloud_data_.clear();
        }
        buff_mutex_.unlock();
    }

    void CloudSubscriber::ParseData(std::deque<CloudRingData> &cloud_data_buff) {
        buff_mutex_.lock();
        if (new_cloud_ring_data_.size() > 0) {
            cloud_data_buff.insert(cloud_data_buff.end(), new_cloud_ring_data_.begin(), new_cloud_ring_data_.end());
            new_cloud_ring_data_.clear();
        }
        buff_mutex_.unlock();
    }

} // namespace data_input