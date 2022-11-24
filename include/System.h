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
#include <pcl/filters/crop_box.h>

typedef pcl::PointXYZI PointType;

namespace rgbd_inertial_slam {

    class System {

    private:

        std::string fisheye_topic_, depth_topic_, rs_rgb_topic_;
        std::string vins_path_, gt_dataset_path_, omni_path_, rs_dep_rgb_path_, points_path_, r3live_path_, r3_omni_depth_path_;
        std::vector<std::string> image_paths, depth_paths, dep_rgb_path_;
        std::vector<std::string> r3_image_paths, r3_depth_paths, r3_depth_vis_paths, r3_dep_rgb_path_;
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Ex_vec_;
        Eigen::Matrix4d Ex_10_, Ex_21_, Ex_32_;
        Eigen::Matrix4d Tic_;
        double depth_scale_;

        camodocal::CameraPtr m_camera0, m_camera1, m_camera2, m_camera3, m_rgbd_model_;
        std::vector<camodocal::CameraPtr> m_camera_vec;

        pcl::PointCloud<PointType>::Ptr map_ptr_, tmp_omni_ptr_;

        std::map<double, Eigen::Matrix4d> pose_map_;
        std::vector<double> time_vec_;

//        bag read
        rosbag::Bag bag_;
        std::shared_ptr<rosbag::View> view_;
        std::vector<std::string> topics_;

        ros::Subscriber subscriber_;

        //config
        YAML::Node config_node_;

        bool debug_ = false;
        bool use_map_ = true;
        std::mutex mutex;

    public:

        System(ros::NodeHandle &nh);

        System(std::string config_file) {

            config_node_ = YAML::LoadFile(config_file);

            LoadParameters();

//            InitBag();

            GenVirtualDataset();
//            if(!use_map_ && debug_){
//                ReGenMap();
//            }
//
//            TravelBag();
        }

        // subscriber
        std::shared_ptr<CloudSubscriber> cloud_sub_ptr_;

    private:

        void GenVirtualDataset() {
            std::string traj_path = r3live_path_ + "/trajectory.txt";
            std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> pose_map;
            load_csv_r3live(traj_path, pose_map);
            std::string pcd_path = r3live_path_ + "/rgb_pt.pcd";
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr global_map_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
            pcl::io::loadPCDFile(pcd_path, *global_map_ptr);

            Ex_vec_.clear();
            Eigen::Matrix4d Ex_0c = Eigen::Matrix4d::Identity();
            Ex_0c.block<3, 3>(0, 0) = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
            Ex_vec_.push_back(Ex_0c);
            Ex_vec_.push_back(Ex_10_ * Ex_0c);
            Ex_vec_.push_back(Ex_21_ * Ex_10_ * Ex_0c);
            Ex_vec_.push_back(Ex_32_ * Ex_21_ * Ex_10_ * Ex_0c);

            std::vector<size_t> index;
            int step = 2;
            for (int i = 0; i < pose_map.size() / step; i++) {
                index.push_back(i * step);
            }

            tmp_omni_ptr_.reset(new pcl::PointCloud<PointType>());
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&pose_map, &global_map_ptr, this](size_t i) {
                Eigen::Matrix4d pose = pose_map[i];
//                add random yaw
                double rand01 = (double) rand() / RAND_MAX;
                pose.block<3,3>(0,0) = pose.block<3,3>(0,0) * Eigen::AngleAxisd(M_PI / 2.0 * rand01, Eigen::Vector3d::UnitZ()).toRotationMatrix();
//                pcl box filter
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr local_map_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
                pcl::CropBox<pcl::PointXYZRGBA> box_filter;
                Eigen::Vector4f min_pt, max_pt;
                double box_size = 100.0;
                min_pt << pose(0, 3) - box_size, pose(1, 3) - box_size, pose(2, 3) - box_size, 1;
                max_pt << pose(0, 3) + box_size, pose(1, 3) + box_size, pose(2, 3) + box_size, 1;
                box_filter.setMin(min_pt);
                box_filter.setMax(max_pt);
                box_filter.setInputCloud(global_map_ptr);
                box_filter.filter(*local_map_ptr);
                if (local_map_ptr->size() == 0) {
                    return;
                }

                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp_map(new pcl::PointCloud<pcl::PointXYZRGBA>());
                pcl::transformPointCloud(*local_map_ptr, *tmp_map, EigenIsoInv(pose));
                std::vector<cv::Mat> rgb_vec, depth_vec;
                double height_res = 0.5625;
                double width_res = 0.5625;
                cv::Mat omni_depth_ = cv::Mat::zeros(180.0 / height_res, 360.0 / width_res, CV_32FC1);

                for (int j = 0; j < 4; ++j) {
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp_map_fisheye(new pcl::PointCloud<pcl::PointXYZRGBA>());
                    pcl::transformPointCloud(*tmp_map, *tmp_map_fisheye, Ex_vec_[j]);
                    cv::Mat rgb = cv::Mat::zeros(m_camera0->imageHeight(), m_camera0->imageWidth(),
                                                 CV_8UC3);
                    cv::Mat depth = cv::Mat::zeros(m_camera0->imageHeight(), m_camera0->imageWidth(),
                                                   CV_32FC1);
                    for (int k = 0; k < tmp_map_fisheye->points.size(); ++k) {
                        pcl::PointXYZRGBA p = tmp_map_fisheye->points[k];
                        Eigen::Vector3d p3d(p.x, p.y, p.z);
                        if (p3d(2) > 0.1) {
                            Eigen::Vector2d p2d;
                            m_camera_vec[j]->spaceToPlane(p3d, p2d);
                            if (p2d[0] < 0 || p2d[0] >= m_camera_vec[j]->imageWidth() || p2d[1] < 0 ||
                                p2d[1] >= m_camera_vec[j]->imageHeight())
                                continue;
                            int u = p2d[0], v = p2d[1];
                            if (depth.at<float>(v, u) == 0 || depth.at<float>(v, u) > p3d[2]) {
                                depth.at<float>(v, u) = p3d[2];
                                rgb.at<cv::Vec3b>(v, u) = cv::Vec3b(p.b, p.g, p.r);
                            }
                        }

                        if(j==0){
                            double r = p3d.norm();
                            double theta = atan2(-p3d(1), sqrt(p3d(0) * p3d(0) + p3d(2) * p3d(2)));
                            double phi = atan2(p3d(2), p3d(0));
                            int row = (theta + M_PI / 2.0) * 180.0 / M_PI / height_res;
                            int col = (phi + M_PI) * 180.0 / M_PI / width_res;
                            if (row >= 0 && row < omni_depth_.rows && col >= 0 && col < omni_depth_.cols) {
                                if (omni_depth_.at<float>(row, col) == 0 || omni_depth_.at<float>(row, col) > r) {
                                    omni_depth_.at<float>(row, col) = r;
                                }
                            }
                        }
                    }
                    rgb_vec.push_back(rgb);
                    depth_vec.push_back(depth);
                }
                std::string i_s = std::to_string(i);
                while (i_s.size() < 6) {
                    i_s = "0" + i_s;
                }
                for (int j = 0; j < 4; ++j) {
                    cv::Mat rgb = rgb_vec[j];
                    cv::imwrite(r3_image_paths[j] + "/" + i_s + ".png", rgb);
                    if(debug_) {
                        cv::Mat depth = depth_vec[j];
                        cv::imwrite(r3_depth_paths[j] + "/" + i_s + ".tiff", depth);
                        cv::imwrite(r3_depth_vis_paths[j] + "/" + i_s + ".png", depth * 10);
                        cv::Mat merge_img = cv::Mat::zeros(rgb.rows, rgb.cols, CV_8UC3);
                        MergeDepthRGB(depth, rgb, merge_img);
                        cv::imwrite(r3_dep_rgb_path_[j] + "/" + i_s + ".png", merge_img);
                    }
                }
                cv::imwrite(r3_omni_depth_path_ + "/" + i_s + ".tiff", omni_depth_);
                LOG(INFO) << "Gen " << i << " / " << pose_map.size() << " at " << pose_map[i].block<3, 1>(0, 3).transpose();
                if(debug_){
                    ReGenDepthUsingOmni(omni_depth_, i_s);
                }
            });
        }

        void ReGenDepthUsingOmni(const cv::Mat &omni_depth_, const std::string i_s){

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp_omni(new pcl::PointCloud<pcl::PointXYZRGBA>());
            GenPCLfromOmni(omni_depth_, tmp_omni);
            std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Ex_vec_fisheye;
            Ex_vec_fisheye.push_back(Eigen::Matrix4d::Identity());
            Ex_vec_fisheye.push_back(Ex_10_);
            Ex_vec_fisheye.push_back(Ex_21_ * Ex_10_);
            Ex_vec_fisheye.push_back(Ex_32_ * Ex_21_ * Ex_10_);
            std::vector<cv::Mat> depth_vec;
            for (int j = 0; j < 4; ++j) {
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp_map_fisheye(new pcl::PointCloud<pcl::PointXYZRGBA>());
                pcl::transformPointCloud(*tmp_omni, *tmp_map_fisheye, Ex_vec_fisheye[j]);
                cv::Mat depth = cv::Mat::zeros(m_camera0->imageHeight(), m_camera0->imageWidth(),
                                               CV_32FC1);
                for (int k = 0; k < tmp_map_fisheye->points.size(); ++k) {
                    pcl::PointXYZRGBA p = tmp_map_fisheye->points[k];
                    Eigen::Vector3d p3d(p.x, p.y, p.z);
                    if (p3d(2) > 0.1) {
                        Eigen::Vector2d p2d;
                        m_camera_vec[j]->spaceToPlane(p3d, p2d);
                        if (p2d[0] < 0 || p2d[0] >= m_camera_vec[j]->imageWidth() || p2d[1] < 0 ||
                            p2d[1] >= m_camera_vec[j]->imageHeight())
                            continue;
                        int u = p2d[0], v = p2d[1];
                        if (depth.at<float>(v, u) == 0 || depth.at<float>(v, u) > p3d[2]) {
                            depth.at<float>(v, u) = p3d[2];
                        }
                    }
                }
                depth_vec.push_back(depth);
            }
            for (int j = 0; j < 4; ++j) {
                cv::Mat depth = depth_vec[j];
                cv::imwrite(r3_depth_vis_paths[j] + "/" + i_s + "_re.png", depth * 10);
            }
        }

        void GenPCLfromOmni(const cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &pcd_ptr) {
            pcd_ptr->clear();
            double height_res = 0.5625;
            double width_res = 0.5625;
            for (int row = 0; row < depth.rows; ++row) {
                for (int col = 0; col < depth.cols; ++col) {
                    double r = depth.at<float>(row, col);
                    pcl::PointXYZRGBA p;
                    double theta = row * height_res * M_PI / 180.0 - M_PI / 2.0;
                    double phi = col * width_res * M_PI / 180.0 - M_PI;
                    p.x = r * cos(theta) * cos(phi);
                    p.y = -r * sin(theta);
                    p.z = r * cos(theta) * sin(phi);
                    pcd_ptr->push_back(p);
                }
            }
        }

        void LoadParameters() {
            debug_ = config_node_["debug"].as<bool>();
            use_map_ = config_node_["use_map"].as<bool>();

            depth_scale_ = config_node_["depth_scale"].as<double>();
            vins_path_ = config_node_["vins_path"].as<std::string>();
            load_csv(vins_path_ + "/vins_result_loop.csv", pose_map_, time_vec_);
            LOG(INFO) << "package_path: " << vins_path_;
            if (use_map_) {
                map_ptr_.reset(new pcl::PointCloud<PointType>());
                pcl::io::loadPCDFile(vins_path_ + "/map.pcd", *map_ptr_);
            } else {
                map_ptr_.reset(new pcl::PointCloud<PointType>());
            }
            Tic_ = MatFromArray<double>(config_node_["l515"]["body_T_cam0"].as<std::vector<double>>());
            for (auto pose_it = pose_map_.begin(); pose_it != pose_map_.end(); ++pose_it) {
                pose_it->second = pose_it->second * Tic_;
            }
            Ex_10_ = MatFromArray<double>(config_node_["Ex_10"].as<std::vector<double>>());
            Ex_21_ = MatFromArray<double>(config_node_["Ex_21"].as<std::vector<double>>());
            Ex_32_ = MatFromArray<double>(config_node_["Ex_32"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_c2 = MatFromArray<double>(config_node_["Ex_c2"].as<std::vector<double>>());
            Eigen::Matrix4d Ex_c1 = Ex_c2 * Ex_21_;
            Eigen::Matrix4d Ex_c0 = Ex_c1 * Ex_10_;
            Eigen::Matrix4d Ex_c3 = Ex_c2 * EigenIsoInv(Ex_32_);

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
            m_rgbd_model_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
                    ros::package::getPath("rgbd_inertial_slam") + "/config/l515.yaml");
            m_camera_vec.push_back(m_camera0);
            m_camera_vec.push_back(m_camera1);
            m_camera_vec.push_back(m_camera2);
            m_camera_vec.push_back(m_camera3);

            r3live_path_ = config_node_["r3live_path"].as<std::string>();
            std::string seq_name = r3live_path_.substr(r3live_path_.find_last_of("/") + 1);
            std::string base_dir = r3live_path_ + '/' + seq_name;
            r3_omni_depth_path_ = base_dir + "/omni_depth";
            FileManager::CreateDirectory(r3_omni_depth_path_);
            for (int i = 0; i < 4; i++) {
                r3_image_paths.push_back(base_dir + "/cam" + std::to_string(i + 1) + "/image");
                r3_depth_paths.push_back(base_dir + "/cam" + std::to_string(i + 1) + "/depth");
                r3_depth_vis_paths.push_back(base_dir + "/cam" + std::to_string(i + 1) + "/depth_vis");
                r3_dep_rgb_path_.push_back(base_dir + "/cam" + std::to_string(i + 1) + "/dep_rgb");
                FileManager::CreateDirectory(r3_image_paths[i]);
                FileManager::CreateDirectory(r3_depth_paths[i]);
                FileManager::CreateDirectory(r3_depth_vis_paths[i]);
                FileManager::CreateDirectory(r3_dep_rgb_path_[i]);
            }
        }

        void InitBag() {
            std::string bag_path = config_node_["bag_path"].as<std::string>();
            std::string bag_name = bag_path.substr(bag_path.find_last_of("/") + 1);
            bag_name = bag_name.substr(0, bag_name.find_last_of("."));
            if (debug_) {
                bag_name = bag_name + "_debug";
            }
            bag_.open(bag_path, rosbag::bagmode::Read); //打开一个bag文件
            LOG(INFO) << "Open bag: " << bag_path;

            fisheye_topic_ = config_node_["fisheye_topic"].as<std::string>();
            depth_topic_ = config_node_["depth_topic"].as<std::string>();
            rs_rgb_topic_ = config_node_["rs_rgb_topic"].as<std::string>();
            topics_.push_back(fisheye_topic_);
            topics_.push_back(depth_topic_);
            topics_.push_back(rs_rgb_topic_);

            view_ = std::make_shared<rosbag::View>(bag_, rosbag::TopicQuery(topics_));
            if (debug_) {
                ros::Time start_time = view_->getBeginTime() + ros::Duration(10.0);
                ros::Time end_time = start_time + ros::Duration(1.0);
                view_ = std::make_shared<rosbag::View>(bag_, rosbag::TopicQuery(topics_), start_time, end_time);
            }

            gt_dataset_path_ = config_node_["ground_truth_path"].as<std::string>();
            omni_path_ = gt_dataset_path_ + bag_name + "/omni";
            rs_dep_rgb_path_ = gt_dataset_path_ + bag_name + "/rs_dep_rgb";
            points_path_ = gt_dataset_path_ + bag_name + "/benchmark_realworld";
            FileManager::CreateDirectory(omni_path_);
            FileManager::CreateDirectory(rs_dep_rgb_path_);
            FileManager::CreateDirectory(points_path_);
            for (int i = 0; i < 4; i++) {
                image_paths.push_back(gt_dataset_path_ + bag_name + "/cam" + std::to_string(i + 1) + "/image");
                depth_paths.push_back(gt_dataset_path_ + bag_name + "/cam" + std::to_string(i + 1) + "/depth");
                dep_rgb_path_.push_back(gt_dataset_path_ + bag_name + "/cam" + std::to_string(i + 1) + "/dep_rgb");
                FileManager::CreateDirectory(image_paths[i]);
                FileManager::CreateDirectory(depth_paths[i]);
                FileManager::CreateDirectory(dep_rgb_path_[i]);
            }
        }

        void ReGenMap() {
            rosbag::View::iterator view_it = view_->begin(); //使用迭代器的方式遍历
            bool first_frame = true;
            std::ofstream fout;
            FileManager::CreateFile(fout, points_path_ + "/alidarPose.csv");
            int frame_count = 0;
            Eigen::Matrix4d T_init = Eigen::Matrix4d::Identity();
            pcl::PointCloud<PointType>::Ptr global_map_ptr(new pcl::PointCloud<PointType>());
            while (view_it != view_->end()) {
                auto m = *view_it;
                std::string cur_topic = m.getTopic();
                //interpolate pose
                double cur_time = m.getTime().toSec();
                Eigen::Matrix4d cur_Twc = Eigen::Matrix4d::Identity();
                auto it = lower_bound(time_vec_.begin(), time_vec_.end(), cur_time);
                bool pose_available = false;
                if (it != time_vec_.end() && it != time_vec_.begin()) {
                    double upper_t = *it;
                    double lower_t = *(it - 1);
                    if (abs(upper_t - lower_t) < 1.0) {
                        pose_available = true;
                        double ratio = (cur_time - lower_t) / (upper_t - lower_t);
                        cur_Twc = InterpolatePose(pose_map_[lower_t], pose_map_[upper_t], ratio);
                    }
                }
                if (cur_topic == depth_topic_ && pose_available) {
                    sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
                    cv::Mat depth = GetDepthFromImage(img_msg, depth_scale_);
                    pcl::PointCloud<PointType>::Ptr tmp_map_ptr_(new pcl::PointCloud<PointType>());
                    Depth2Points(depth, tmp_map_ptr_, m_rgbd_model_);
                    if (first_frame) {
                        pcl::transformPointCloud(*tmp_map_ptr_, *map_ptr_, cur_Twc);
                        T_init = cur_Twc;
                        first_frame = false;
                    }
                    pcl::io::savePCDFileASCII(points_path_ + "/full" + std::to_string(frame_count++) + ".pcd",
                                              *tmp_map_ptr_);
                    Eigen::Matrix4d T_cur = EigenIsoInv(T_init) * cur_Twc;
                    for (int i = 0; i < 4; ++i) {
                        fout << T_cur(i, 0) << "," << T_cur(i, 1) << "," << T_cur(i, 2) << "," << T_cur(i, 3) << ",\n";
                    }
                    LOG(INFO) << "T_cur: \n" << T_cur;
                    pcl::transformPointCloud(*tmp_map_ptr_, *tmp_map_ptr_, T_cur);
                    *global_map_ptr = *global_map_ptr + *tmp_map_ptr_;
                }
                view_it++;
            }
            pcl::io::savePCDFileASCII(points_path_ + "/full.pcd", *global_map_ptr);
            fout.close();
        }

        void TravelBag() {
            rosbag::View::iterator view_it = view_->begin(); //使用迭代器的方式遍历
            while (view_it != view_->end()) {
                auto m = *view_it;
                std::string cur_topic = m.getTopic();
                //interpolate pose
                double cur_time = m.getTime().toSec();
                Eigen::Matrix4d cur_Twc = Eigen::Matrix4d::Identity();
                auto it = lower_bound(time_vec_.begin(), time_vec_.end(), cur_time);
                bool pose_available = false;
                if (it != time_vec_.end() && it != time_vec_.begin()) {
                    double upper_t = *it;
                    double lower_t = *(it - 1);
                    if (abs(upper_t - lower_t) < 1.0) {
                        pose_available = true;
                        double ratio = (cur_time - lower_t) / (upper_t - lower_t);
                        cur_Twc = InterpolatePose(pose_map_[lower_t], pose_map_[upper_t], ratio);
                    }
                }

                if (cur_topic == fisheye_topic_ && pose_available) {
                    if (it != time_vec_.end() && it != time_vec_.begin()) {
                        std::vector<cv::Mat> img_vec;
                        std::vector<cv::Mat> dep_vec;
                        sensor_msgs::CompressedImageConstPtr img_msg = m.instantiate<sensor_msgs::CompressedImage>();
                        SaveIndividualImage(img_msg, cur_time, img_vec);
                        SaveOmniImage(cur_Twc, cur_time, 0.5625, 0.5625);
                        LOG(INFO) << std::fixed << "Write image at " << cur_time;

                        if (debug_) {
                            SaveIndividualDepth(cur_Twc, cur_time, dep_vec);
                            SaveRGBWithDepth(img_vec, dep_vec, cur_time);
                            ReProjectUsingOmni(cur_Twc, cur_time);
                        }
                    }
                }
                if (cur_topic == rs_rgb_topic_ && debug_ && pose_available) {
                    cv::Mat rgb_img;
                    if (cur_topic.find("compressed") != std::string::npos) {
                        sensor_msgs::CompressedImagePtr img_msg = m.instantiate<sensor_msgs::CompressedImage>();
                        rgb_img = GetImageFromCompressed(img_msg);
                    } else {
                        sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
                        rgb_img = GetImageFromImage(img_msg);
                    }
                    SaveRsRGBWithDepth(rgb_img, cur_time, cur_Twc);
                }

                view_it++;
            }
            bag_.close();
        }

        void SplitImageHorizon(const cv::Mat &img, std::vector<cv::Mat> &img_vec) {
            int width = img.cols / 4;
            int height = img.rows;
            for (int i = 0; i < 4; ++i) {
                cv::Mat img_i = img(cv::Rect(i * width, 0, width, height));
                img_vec.push_back(img_i);
            }
        }

        void
        SaveIndividualDepth(const Eigen::Matrix4d &cur_Twc, const double &cur_time, std::vector<cv::Mat> &dep_vec) {
            for (int idx = 0; idx < 4; ++idx) {
                cv::Mat depth_map = cv::Mat::zeros(m_camera_vec[idx]->imageHeight(),
                                                   m_camera_vec[idx]->imageWidth(), CV_32FC1);

                Eigen::Matrix4d Twf = cur_Twc * Ex_vec_[idx];
                Eigen::Matrix4d Tfw = EigenIsoInv(Twf);
                pcl::PointCloud<PointType>::Ptr fisheye_map_ptr(new pcl::PointCloud<PointType>());
                pcl::transformPointCloud(*map_ptr_, *fisheye_map_ptr, Tfw);
                for (int i = 0; i < fisheye_map_ptr->points.size(); ++i) {
                    Eigen::Vector3d p_cam = fisheye_map_ptr->points[i].getVector3fMap().cast<double>();
                    if (p_cam(2) > 0.1) {
                        Eigen::Vector2d px;
                        int u = px(0);
                        int v = px(1);
                        m_camera_vec[idx]->spaceToPlane(p_cam, px);
                        if (u >= 0 && u < depth_map.cols && v >= 0 && v < depth_map.rows) {
                            if (debug_)
                                p_cam(2) = p_cam(2) * 1000;
                            if (depth_map.at<float>(v, u) == 0 || depth_map.at<float>(v, u) > p_cam(2)) {
                                depth_map.at<float>(v, u) = p_cam(2);
                            }
                        }
                    }
                }

                dep_vec.push_back(depth_map);
                if (debug_) {
                    cv::imwrite(depth_paths[idx] + "/" + std::to_string(cur_time) + ".png", depth_map);
                    cv::imwrite(depth_paths[idx] + "/" + std::to_string(cur_time) + ".tiff", depth_map);
                } else
                    cv::imwrite(depth_paths[idx] + "/" + std::to_string(cur_time) + ".tiff", depth_map);
            }
        }

        void SaveRGBWithDepth(const std::vector<cv::Mat> &img_vec, const std::vector<cv::Mat> &dep_vec,
                              const double &cur_time) {
            for (int idx = 0; idx < 4; ++idx) {
                cv::Mat img = img_vec[idx];
                cv::Mat dep = dep_vec[idx];

                cv::Mat merge_img = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
                MergeDepthRGB(dep, img, merge_img);
                cv::rotate(merge_img, merge_img, cv::ROTATE_180);
                cv::imwrite(dep_rgb_path_[idx] + "/aligned_" + std::to_string(cur_time) + ".png", merge_img);
            }
        }

        void ReProjectUsingOmni(const Eigen::Matrix4d &cur_Twc, const double &cur_time) {

            Eigen::Matrix4d Twf = cur_Twc * Ex_vec_[0];
//          LOG(INFO) << "tmp_omni_ptr_ first point: " << tmp_omni_ptr_->points[0].x << " " << tmp_omni_ptr_->points[0].y << " " << tmp_omni_ptr_->points[0].z;
            pcl::transformPointCloud(*tmp_omni_ptr_, *tmp_omni_ptr_, Twf); //转到世界坐标系
            for (int idx = 0; idx < 4; ++idx) {
                cv::Mat depth_map = cv::Mat::zeros(m_camera_vec[idx]->imageHeight(),
                                                   m_camera_vec[idx]->imageWidth(), CV_32FC1);

                Eigen::Matrix4d Twf_idx = cur_Twc * Ex_vec_[idx];
                Eigen::Matrix4d Tfw = EigenIsoInv(Twf_idx);
                pcl::PointCloud<PointType>::Ptr fisheye_map_ptr(new pcl::PointCloud<PointType>());
                pcl::transformPointCloud(*tmp_omni_ptr_, *fisheye_map_ptr, Tfw);
//              LOG(INFO) << "idx: " << idx << ", trans mat: \n" << Tfw * Twf;
//              LOG(INFO) << "fisheye_map_ptr first point: " << fisheye_map_ptr->points[0].x << " " << fisheye_map_ptr->points[0].y << " " << fisheye_map_ptr->points[0].z;
                for (int i = 0; i < fisheye_map_ptr->points.size(); ++i) {
                    Eigen::Vector3d p_cam(fisheye_map_ptr->points[i].x, fisheye_map_ptr->points[i].y,
                                          fisheye_map_ptr->points[i].z);
                    if (p_cam(2) > 0.1) {
                        Eigen::Vector2d px;
                        int u = px(0);
                        int v = px(1);
                        m_camera_vec[idx]->spaceToPlane(p_cam, px);
                        if (u >= 0 && u < depth_map.cols && v >= 0 && v < depth_map.rows) {
                            if (debug_)
                                p_cam(2) = p_cam(2) * 1000;
                            if (depth_map.at<float>(v, u) == 0 || depth_map.at<float>(v, u) > p_cam(2)) {
                                depth_map.at<float>(v, u) = p_cam(2);
                            }
                        }
                    }
                }
                cv::imwrite(depth_paths[idx] + "/" + std::to_string(cur_time) + "_re.png", depth_map);
                cv::imwrite(depth_paths[idx] + "/" + std::to_string(cur_time) + "_re.tiff", depth_map);
            }
        }

        void SaveRsRGBWithDepth(const cv::Mat &rs_rgb_img, const double &cur_time, const Eigen::Matrix4d &cur_Twc) {
            Eigen::Matrix4d Twc = cur_Twc;
            pcl::PointCloud<PointType>::Ptr rs_map_ptr(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*map_ptr_, *rs_map_ptr, EigenIsoInv(Twc));
            cv::Mat depth_map = cv::Mat::zeros(rs_rgb_img.rows, rs_rgb_img.cols, CV_32FC1);
            for (int i = 0; i < rs_map_ptr->points.size(); ++i) {
                Eigen::Vector3d p_cam(rs_map_ptr->points[i].x, rs_map_ptr->points[i].y, rs_map_ptr->points[i].z);
                if (p_cam(2) > 0.1) {
                    Eigen::Vector2d px;
                    m_rgbd_model_->spaceToPlane(p_cam, px);
                    int u = px(0);
                    int v = px(1);
                    if (u >= 0 && u < depth_map.cols && v >= 0 && v < depth_map.rows) {
                        if (debug_)
                            p_cam(2) = p_cam(2) * 1000;
                        if (depth_map.at<float>(v, u) == 0 || depth_map.at<float>(v, u) > p_cam(2)) {
                            depth_map.at<float>(v, u) = p_cam(2);
                        }
                    }
                }
            }
            cv::Mat merge_img = cv::Mat::zeros(rs_rgb_img.rows, rs_rgb_img.cols, CV_8UC3);
            MergeDepthRGB(depth_map, rs_rgb_img, merge_img);
            cv::imwrite(rs_dep_rgb_path_ + "/aligned_" + std::to_string(cur_time) + ".png", merge_img);
        }

        void SaveIndividualImage(const sensor_msgs::CompressedImageConstPtr &img_msg, const double &cur_time,
                                 std::vector<cv::Mat> &img_vec) {
            cv::Mat img = GetImageFromCompressed(img_msg);
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

        void SaveOmniImage(const Eigen::Matrix4d &cur_Twc, const double &cur_time, const double &height_res,
                           const double &width_res) {
//            omni_path_
            // regard the fisheye image 0 as the omni image
            Eigen::Matrix4d Twf = cur_Twc * Ex_vec_[0];
            Eigen::Matrix4d Tfw = EigenIsoInv(Twf);
            pcl::PointCloud<PointType>::Ptr fisheye_map_ptr(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*map_ptr_, *fisheye_map_ptr, Tfw);
            cv::Mat depth_map = cv::Mat::zeros(180.0 / height_res, 360.0 / width_res, CV_32FC1);

            std::vector<size_t> index(fisheye_map_ptr->size());
            for (int i = 0; i < fisheye_map_ptr->size(); ++i) {
                index[i] = i;
            }

            tmp_omni_ptr_.reset(new pcl::PointCloud<PointType>());
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
                                  mutex.lock();
                                  if (debug_)
                                      r = r * 1000;
                                  if (depth_map.at<float>(row, col) == 0 || depth_map.at<float>(row, col) > r) {
                                      if (debug_) {
                                          PointType p;
                                          theta = row * height_res * M_PI / 180.0 - M_PI / 2.0;
                                          phi = col * width_res * M_PI / 180.0 - M_PI;
                                          p.x = r * cos(theta) * cos(phi) / 1000.0;
                                          p.y = -r * sin(theta) / 1000.0;
                                          p.z = r * cos(theta) * sin(phi) / 1000.0;
//                                          if (row==158 && col==371) {
//                                              LOG(INFO) << "row: " << row << " col: " << col << " theta: " << theta << " phi: " << phi << " r: " << r;
//                                              LOG(INFO) << "p: " << p.x << " " << p.y << " " << p.z;
//                                          }
                                          this->tmp_omni_ptr_->points.push_back(p);
                                      }
                                      depth_map.at<float>(row, col) = r;
                                  }
                                  mutex.unlock();
                              }
                          });
            if (debug_) {
//                LOG(INFO) << "row: " << 158 << " col: " << 371 << " r: " << depth_map.at<float>(158, 371);
                cv::imwrite(omni_path_ + "/" + std::to_string(cur_time) + ".png", depth_map);
                cv::imwrite(omni_path_ + "/" + std::to_string(cur_time) + ".tiff", depth_map);
            } else
                cv::imwrite(omni_path_ + "/" + std::to_string(cur_time) + ".tiff", depth_map);
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

        void load_csv_r3live(std::string file_path,
                             std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &pose_map) {
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
//                LOG(INFO) << pose.block<3, 1>(0, 3).transpose();
                pose_map.push_back(pose);
            }
            fclose(pFile);
        }
    };
}


#endif //CMAKE_TEMPLATE_SYSTEM_H
