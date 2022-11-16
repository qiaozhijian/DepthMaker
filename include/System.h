//
// Created by qzj on 2021/12/24.
//

#ifndef CMAKE_TEMPLATE_SYSTEM_H
#define CMAKE_TEMPLATE_SYSTEM_H

#include <ros/ros.h>
// subscriber
#include "subscriber/cloud_subscriber.hpp"
#include "config.h"
#include "glog/logging.h"
#include "ros/package.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "CommonFunc.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"


namespace rgbd_inertial_slam {

    class System {

    public:

        System(ros::NodeHandle &nh);

        // subscriber
        std::shared_ptr<CloudSubscriber> cloud_sub_ptr_;

    private:

        void LoadParameters(ros::NodeHandle &nh){
            std::string config_file_path = ros::package::getPath("rgbd_inertial_slam") + "/config/template.yaml";
            auto yaml = YAML::LoadFile(config_file_path);

            nh.param<std::string>("/fisheye0/topic_name", fisheye_topic0_, "");
            nh.param<std::string>("/fisheye1/topic_name", fisheye_topic1_, "");
            nh.param<std::string>("/fisheye2/topic_name", fisheye_topic2_, "");
            nh.param<std::string>("/fisheye3/topic_name", fisheye_topic3_, "");

            Eigen::Matrix4d Ex_01 = MatFromArray<double >(yaml["fisheye0"]["extrinsic"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_12 = MatFromArray<double >(yaml["fisheye1"]["extrinsic"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_23 = MatFromArray<double >(yaml["fisheye2"]["extrinsic"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_c2 = MatFromArray<double >(yaml["fisheye3"]["extrinsic"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_c1 = Ex_c2 * EigenIsoInv(Ex_12);
            Eigen::Matrix4d Ex_c0 = Ex_c1 * EigenIsoInv(Ex_01);
            Eigen::Matrix4d Ex_c3 = Ex_c2 * Ex_23;
            Ex_vec_.push_back(Ex_c0);
            Ex_vec_.push_back(Ex_c1);
            Ex_vec_.push_back(Ex_c2);
            Ex_vec_.push_back(Ex_c3);

            m_camera0 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
                    ros::package::getPath("rgbd_inertial_slam") + "/config/cam0_mei.yaml");
            m_camera1 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
                    ros::package::getPath("rgbd_inertial_slam") + "/config/cam1_mei.yaml");
            m_camera2 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
                    ros::package::getPath("rgbd_inertial_slam") + "/config/cam2_mei.yaml");
            m_camera3 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
                    ros::package::getPath("rgbd_inertial_slam") + "/config/cam3_mei.yaml");
            m_camera_vec.push_back(m_camera0);
            m_camera_vec.push_back(m_camera1);
            m_camera_vec.push_back(m_camera2);
            m_camera_vec.push_back(m_camera3);

            nh.param<std::string>("/vins_path", vins_path_, "");
            std::string map_file = vins_path_ + "/map.pcd";
            map_ptr_.reset(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::io::loadPCDFile(map_file, *map_ptr_);
            load_csv(vins_path_ + "/vins_result_loop.csv", pose_map_);

            LOG(INFO)<<"package_path: "<<vins_path_;
        }

        void load_csv(std::string file_path, std::map<double, Eigen::Matrix4d> &pose_map){
            FILE * pFile = fopen (file_path.c_str(),"r");
            if (pFile == NULL){
                std::cout<<"file not found"<<std::endl;
                return;
            }
            double time_stamp, x, y, z, qx, qy, qz, qw;
            while (fscanf(pFile,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", &time_stamp,
                          &x, &y, &z, &qw, &qx, &qy, &qz) != EOF){
                Eigen::Quaterniond q(qw, qx, qy, qz);
                Eigen::Vector3d t(x, y, z);
                Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                pose.block<3, 3>(0, 0) = q.toRotationMatrix();
                pose.block<3, 1>(0, 3) = t;
                pose_map.insert(std::make_pair(time_stamp / 1e9, pose));
            }
        }

        void InitBag(ros::NodeHandle &nh){
            std::string bag_path;
            nh.param<std::string>("/bag_path", bag_path, "");
            bag_.open(bag_path, rosbag::bagmode::Read); //打开一个bag文件
            topics_.push_back(fisheye_topic0_);
            topics_.push_back(fisheye_topic1_);
            topics_.push_back(fisheye_topic2_);
            topics_.push_back(fisheye_topic3_);
            LOG(INFO)<<"Read topics: "<<fisheye_topic0_<<" "<<fisheye_topic1_<<" "<<fisheye_topic2_<<" "<<fisheye_topic3_;
            view_ = std::make_shared<rosbag::View>(bag_, rosbag::TopicQuery(topics_));
        }

        void TravelBag(){
            rosbag::View::iterator it = view_->begin(); //使用迭代器的方式遍历
            while (it != view_->end() && ros::ok()) {
                auto m = *it;
                std::string cur_topic = m.getTopic();
                std::vector<std::string>::iterator iter=std::find(topics_.begin(),topics_.end(),cur_topic);//返回的是一个迭代器指针
                int idx = std::distance(topics_.begin(),iter);
                if (cur_topic == fisheye_topic0_) {
                }
                it++;
            }
            bag_.close();
        }

        void ProjectMap(int idx){
            for (int i = 0; i < map_ptr_->points.size(); ++i) {
                Eigen::Vector3d p(map_ptr_->points[i].x, map_ptr_->points[i].y, map_ptr_->points[i].z);
                Eigen::Vector3d p_cam = Ex_vec_[idx].block<3, 3>(0, 0) * p + Ex_vec_[idx].block<3, 1>(0, 3);
                Eigen::Vector2d p_img;
                m_camera_vec[idx]->spaceToPlane(p_cam, p_img);
                cv::Point2d p_cv(p_img(0), p_img(1));
//                cv::circle(img_vec_[j], p_cv, 1, cv::Scalar(0, 255, 0), 1);
            }
        }

    private:

        std::string fisheye_topic0_, fisheye_topic1_, fisheye_topic2_, fisheye_topic3_;
        std::string vins_path_;
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Ex_vec_;

        camodocal::CameraPtr m_camera0, m_camera1, m_camera2, m_camera3;
        std::vector<camodocal::CameraPtr> m_camera_vec;

        pcl::PointCloud<pcl::PointXYZ>::Ptr map_ptr_;

        std::map<double, Eigen::Matrix4d> pose_map_;
        std::vector<double> extrinT_{3, 0.0};  // lidar-imu translation

//        bag read
        rosbag::Bag bag_;
        std::shared_ptr<rosbag::View> view_;
        std::vector <std::string> topics_;

        ros::Subscriber subscriber_;

        //config
        YAML::Node config_node_;
    };
}


#endif //CMAKE_TEMPLATE_SYSTEM_H
