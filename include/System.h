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
#include "cv_bridge/cv_bridge.h"
#include "CommonFunc.h"
#include "file_manager.hpp"
#include "algorithm"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include <execution>

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>


namespace rgbd_inertial_slam {

    class System {

    public:

        System(ros::NodeHandle &nh);

        // subscriber
        std::shared_ptr<CloudSubscriber> cloud_sub_ptr_;

    private:

        void LoadParameters(ros::NodeHandle &nh) {
            std::string config_file_path = ros::package::getPath("rgbd_inertial_slam") + "/config/template.yaml";
            auto yaml = YAML::LoadFile(config_file_path);

            Tic_ = MatFromArray<double>(yaml["l515"]["body_T_cam0"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_10 = MatFromArray<double>(yaml["Ex_10"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_21 = MatFromArray<double>(yaml["Ex_21"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_32 = MatFromArray<double>(yaml["Ex_32"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_c2 = MatFromArray<double>(yaml["Ex_c2"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_c1 = Ex_c2 * Ex_21;
            Eigen::Matrix4d Ex_c0 = Ex_c1 * Ex_10;
            Eigen::Matrix4d Ex_c3 = Ex_c2 * EigenIsoInv(Ex_32);

            Ex_vec_.push_back(Tic_ * Ex_c0);
            Ex_vec_.push_back(Tic_ * Ex_c1);
            Ex_vec_.push_back(Tic_ * Ex_c2);
            Ex_vec_.push_back(Tic_ * Ex_c3);

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
            load_csv(vins_path_ + "/vins_result_loop.csv", pose_map_, time_vec_);

            map_ptr_.reset(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::io::loadPCDFile(vins_path_ + "/map.pcd", *map_ptr_);

            LOG(INFO) << "package_path: " << vins_path_;
        }

        void
        load_csv(std::string file_path, std::map<double, Eigen::Matrix4d> &pose_map, std::vector<double> &time_vec) {
            FILE *pFile = fopen(file_path.c_str(), "r");
            if (pFile == NULL) {
                std::cout << "file not found" << std::endl;
                return;
            }
            double time_stamp, x, y, z, qx, qy, qz, qw;
            while (fscanf(pFile, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", &time_stamp,
                          &x, &y, &z, &qw, &qx, &qy, &qz) != EOF) {
                Eigen::Quaterniond q(qw, qx, qy, qz);
                Eigen::Vector3d t(x, y, z);
                Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                pose.block<3, 3>(0, 0) = q.toRotationMatrix();
                pose.block<3, 1>(0, 3) = t;
                double time_sec = time_stamp / 1e9;
                pose_map.insert(std::make_pair(time_sec, pose));
                time_vec.push_back(time_sec);
            }
            fclose(pFile);
            sort(time_vec.begin(), time_vec.end());
        }

        void InitBag(ros::NodeHandle &nh) {
            std::string bag_path;
            nh.param<std::string>("/bag_path", bag_path, "");
            nh.param<std::string>("/fisheye0/topic_name", fisheye_topic0_, "");
            nh.param<std::string>("/fisheye1/topic_name", fisheye_topic1_, "");
            nh.param<std::string>("/fisheye2/topic_name", fisheye_topic2_, "");
            nh.param<std::string>("/fisheye3/topic_name", fisheye_topic3_, "");
            nh.param<std::string>("/fisheye_topic", fisheye_topic_, "");
            bag_.open(bag_path, rosbag::bagmode::Read); //打开一个bag文件
            LOG(INFO) << "Open bag: " << bag_path;
//            topics_.push_back(fisheye_topic0_);
//            topics_.push_back(fisheye_topic1_);
//            topics_.push_back(fisheye_topic2_);
//            topics_.push_back(fisheye_topic3_);
//            LOG(INFO)<<"Read topics: "<<fisheye_topic0_<<" "<<fisheye_topic1_<<" "<<fisheye_topic2_<<" "<<fisheye_topic3_;
            topics_.push_back(fisheye_topic_);
            view_ = std::make_shared<rosbag::View>(bag_, rosbag::TopicQuery(topics_));

            nh.param<std::string>("/ground_truth_path", gt_dataset_path_, "");
            std::string bag_name = bag_path.substr(bag_path.find_last_of("/") + 1);
            bag_name = bag_name.substr(0, bag_name.find_last_of("."));
            omni_path_ = gt_dataset_path_ + bag_name + "/omni";
            FileManager::CreateDirectory(omni_path_);
            for (int i = 0; i < 4; i++) {
                image_paths.push_back(gt_dataset_path_ + bag_name + "/cam" + std::to_string(i + 1) + "/image");
                depth_paths.push_back(gt_dataset_path_ + bag_name + "/cam" + std::to_string(i + 1) + "/depth");
                FileManager::CreateDirectory(image_paths[i]);
                FileManager::CreateDirectory(depth_paths[i]);
            }
        }

        void SplitImageHorizon(const cv::Mat &img, std::vector<cv::Mat> &img_vec) {
            int width = img.cols / 4;
            int height = img.rows;
            for (int i = 0; i < 4; ++i) {
                cv::Mat img_i = img(cv::Rect(i * width, 0, width, height));
                img_vec.push_back(img_i);
            }
        }

        void SaveIndividualDepth(const Eigen::Matrix4d &cur_pose, const double &cur_time) {
            for (int idx = 0; idx < 4; ++idx) {
                cv::Mat depth_map = cv::Mat::zeros(m_camera_vec[idx]->imageHeight(),
                                                   m_camera_vec[idx]->imageWidth(), CV_32FC1);
                ProjectMap(cur_pose, depth_map, idx);
                cv::imwrite(depth_paths[idx] + "/" + std::to_string(cur_time) + ".png", depth_map);
            }
        }

        void SaveIndividualImage(const sensor_msgs::CompressedImageConstPtr &img_msg, const double &cur_time) {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
            cv::Mat img = cv_ptr->image;
            std::vector<cv::Mat> img_vec;
            SplitImageHorizon(img, img_vec);
            for (int idx = 0; idx < 4; ++idx) {
                cv::Mat img_i = img_vec[idx];
                cv::imwrite(image_paths[idx] + "/" + std::to_string(cur_time) + ".png", img_i);
#if 0
                cv::imshow("img_i", img_i);
                                cv::waitKey(0);
#endif
            }
        }

        void SaveOmniImage(const Eigen::Matrix4d &cur_pose, const double &cur_time, const double &height_res,
                           const double &width_res) {
//            omni_path_
            // regard the fisheye image 0 as the omni image
            Eigen::Matrix4d Twf = cur_pose * Ex_vec_[0];
            Eigen::Matrix4d Tfw = EigenIsoInv(Twf);
            pcl::PointCloud<pcl::PointXYZ>::Ptr fisheye_map_ptr(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*map_ptr_, *fisheye_map_ptr, Tfw);
            cv::Mat depth_map = cv::Mat::zeros(180.0 / height_res, 360.0 / width_res, CV_32FC1);

            std::vector<size_t> index(fisheye_map_ptr->size());
            for (int i = 0; i < fisheye_map_ptr->size(); ++i) {
                index[i] = i;
            }

            std::for_each(std::execution::par_unseq, index.begin(), index.end(),
                          [&fisheye_map_ptr, &depth_map, &height_res, &width_res, this](size_t idx) {
                              Eigen::Vector3d p_cam(fisheye_map_ptr->points[idx].x, fisheye_map_ptr->points[idx].y,
                                                    fisheye_map_ptr->points[idx].z);
                              // Spherical coordinate system
                              double r = p_cam.norm();
                              double theta = atan2(-p_cam(1), sqrt(p_cam(0) * p_cam(0) + p_cam(2) * p_cam(2)));
                              double phi = atan2(p_cam(2), p_cam(0));
                              int row = (theta + M_PI / 2.0) * 180.0 / M_PI / height_res;
                              int col = (phi + M_PI) * 180.0 / M_PI / width_res;
                              if (row >= 0 && row < depth_map.rows && col >= 0 && col < depth_map.cols) {
                                  if (depth_map.at<float>(row, col) == 0 || depth_map.at<float>(row, col) > r) {
                                      depth_map.at<float>(row, col) = r * 100.0;
                                  }
                              }
                          });
            cv::imwrite(omni_path_ + "/" + std::to_string(cur_time) + ".png", depth_map);
        }

        void TravelBag() {
            rosbag::View::iterator view_it = view_->begin(); //使用迭代器的方式遍历
            while (view_it != view_->end() && ros::ok()) {
                auto m = *view_it;
                std::string cur_topic = m.getTopic();
                if (cur_topic == fisheye_topic_) {
                    //interpolate pose
                    double cur_time = m.getTime().toSec();
                    auto it = lower_bound(time_vec_.begin(), time_vec_.end(), cur_time);
//                    LOG(INFO) << std::fixed << "cur_time: " << cur_time << " " << time_vec_.front() << " " << time_vec_.back();
                    if (it != time_vec_.end() && it != time_vec_.begin()) {
                        double upper_t = *it;
                        double lower_t = *(it - 1);
                        if (abs(upper_t - lower_t) < 1.0) {
                            double ratio = (cur_time - lower_t) / (upper_t - lower_t);
                            Eigen::Matrix4d cur_pose = InterpolatePose(pose_map_[lower_t], pose_map_[upper_t], ratio);

                            sensor_msgs::CompressedImageConstPtr img_msg = m.instantiate<sensor_msgs::CompressedImage>();
//                            SaveIndividualDepth(cur_pose, cur_time);
                            SaveIndividualImage(img_msg, cur_time);
                            SaveOmniImage(cur_pose, cur_time, 0.5625, 0.5625);
                            LOG(INFO) << std::fixed << "Write image at " << cur_time;
                        }
                    }
                }
                view_it++;
            }
            bag_.close();
        }

        void ProjectMap(const Eigen::Matrix4d &Twc, cv::Mat &depth_map, int idx) {
            Eigen::Matrix4d Twf = Twc * Ex_vec_[idx];
            Eigen::Matrix4d Tfw = EigenIsoInv(Twf);
            pcl::PointCloud<pcl::PointXYZ>::Ptr fisheye_map_ptr(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*map_ptr_, *fisheye_map_ptr, Tfw);
            for (int i = 0; i < fisheye_map_ptr->points.size(); ++i) {
                Eigen::Vector3d p_cam(fisheye_map_ptr->points[i].x, fisheye_map_ptr->points[i].y,
                                      fisheye_map_ptr->points[i].z);
                if (p_cam(2) > 0.1) {
                    Eigen::Vector2d px;
                    int u = px(0);
                    int v = px(1);
                    m_camera_vec[idx]->spaceToPlane(p_cam, px);
                    if (u >= 0 && u < depth_map.cols && v >= 0 && v < depth_map.rows) {
                        p_cam(2) = p_cam(2);
                        if (depth_map.at<float>(v, u) == 0 || depth_map.at<float>(v, u) > p_cam(2)) {
                            depth_map.at<float>(v, u) = p_cam(2);
                        }
                    }
                }
            }
        }

        bool InFov(Eigen::Vector3d &p_cam) {
//            double theta = atan2(p_cam(1), p_cam(0));
//            double phi = atan2(p_cam(2), p_cam(0));
//            if (theta > fov_h_ / 2.0 || theta < -fov_h_ / 2.0 || phi > fov_v_ / 2.0 || phi < -fov_v_ / 2.0){
//                return false;
//            }
//            return true;
        }

    private:

        std::string fisheye_topic_, fisheye_topic0_, fisheye_topic1_, fisheye_topic2_, fisheye_topic3_;
        std::string vins_path_, gt_dataset_path_, omni_path_;
        std::vector<std::string> image_paths, depth_paths;
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Ex_vec_;
        Eigen::Matrix4d Tic_;

        camodocal::CameraPtr m_camera0, m_camera1, m_camera2, m_camera3;
        std::vector<camodocal::CameraPtr> m_camera_vec;

        pcl::PointCloud<pcl::PointXYZ>::Ptr map_ptr_;

        std::map<double, Eigen::Matrix4d> pose_map_;
        std::vector<double> time_vec_;

//        bag read
        rosbag::Bag bag_;
        std::shared_ptr<rosbag::View> view_;
        std::vector<std::string> topics_;

        ros::Subscriber subscriber_;

        //config
        YAML::Node config_node_;
    };
}


#endif //CMAKE_TEMPLATE_SYSTEM_H
