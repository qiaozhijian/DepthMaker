//
// Created on 19-3-27.
//

#ifndef PROJECT_COMMONFUNC_H
#define PROJECT_COMMONFUNC_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <tf/transform_broadcaster.h>
#include <vector>
#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include "sensor_data/cloud_data.hpp"
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/opencv.hpp>

namespace rgbd_inertial_slam {

    typedef Eigen::Matrix<double, 6, 1> Vector6d;

    inline Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    inline void SaveImgToTiff(const cv::Mat &img, const std::string &path) {
        cv::Mat img_16bit;
        img.convertTo(img_16bit, CV_16UC1);
        cv::imwrite(path, img_16bit);
    }

    inline Eigen::Matrix4d InterpolatePose(const Eigen::Matrix4d &pose1, const Eigen::Matrix4d &pose2, double ratio){
        // Interpolate the pose between pose1 and pose2, ratio is the ratio of pose2
        assert(ratio >= 0 && ratio <= 1 && "ratio should be in [0, 1]");
        Eigen::Matrix4d pose;
        Eigen::Quaterniond q1(pose1.block<3, 3>(0, 0));
        Eigen::Quaterniond q2(pose2.block<3, 3>(0, 0));
        Eigen::Quaterniond q = q1.slerp(ratio, q2);
        pose.block<3, 3>(0, 0) = q.toRotationMatrix();
        pose.block<3, 1>(0, 3) = pose1.block<3, 1>(0, 3) * (1 - ratio) + pose2.block<3, 1>(0, 3) * ratio;
        pose(3, 0) = pose(3, 1) = pose(3, 2) = 0;
        pose(3, 3) = 1;
        return pose;
    }

    template <typename S>
    inline Eigen::Matrix<S, 4, 4> MatFromArray(const std::vector<double> &v) {
        Eigen::Matrix<S, 4, 4> m;
        m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
                , v[9], v[10], v[11], v[12], v[13], v[14], v[15];
        return m;
    }

    template<typename PointT>
    inline void publishCLoudMsg(ros::Publisher &publisher,
                                const pcl::PointCloud<PointT> &cloud,
                                std::string frameID) {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.frame_id = frameID;

        publisher.publish(msg);
    };

    template<typename T>
    inline Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1> &axis) {
        Eigen::Matrix<T, 3, 3> skew_matrix = Eigen::Matrix<T, 3, 3>::Identity();

        skew_matrix<< 0, -axis(2,0), axis(1,0),
                      axis(2,0), 0, -axis(0,0),
                      -axis(1,0), axis(0,0), 0;

        return skew_matrix;
    }

    template<typename T>
    inline Eigen::Matrix<T, 3, 1> dcm2rpy(const Eigen::Matrix<T, 3, 3> &R) {
        Eigen::Matrix<T, 3, 1> rpy;
        rpy[1] = atan2(-R(2, 0), sqrt(pow(R(0, 0), 2) + pow(R(1, 0), 2)));
        if (fabs(rpy[1] - M_PI / 2) < 0.00001) {
            rpy[2] = 0;
            rpy[0] = -atan2(R(0, 1), R(1, 1));
        } else {
            if (fabs(rpy[1] + M_PI / 2) < 0.00001) {
                rpy[2] = 0;
                rpy[0] = -atan2(R(0, 1), R(1, 1));
            } else {
                rpy[2] = atan2(R(1, 0) / cos(rpy[1]), R(0, 0) / cos(rpy[1]));
                rpy[0] = atan2(R(2, 1) / cos(rpy[1]), R(2, 2) / cos(rpy[1]));
            }
        }
        return rpy;
    }

    template<typename T>
    void toEulerAngle(const Eigen::Quaternion<T>& q, T& roll, T& pitch, T& yaw)
    {
// roll (x-axis rotation)
        T sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
        T cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
        roll = atan2(sinr_cosp, cosr_cosp);

// pitch (y-axis rotation)
        T sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
        if (fabs(sinp) >= 1)
            pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
        else
            pitch = asin(sinp);

// yaw (z-axis rotation)
        T siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        T cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        yaw = atan2(siny_cosp, cosy_cosp);
    }

    inline Eigen::Matrix3d rpy2dcm(const Eigen::Vector3d &rpy)//(yaw)
    {
        Eigen::Matrix3d R1;
        R1(0, 0) = 1.0;
        R1(0, 1) = 0.0;
        R1(0, 2) = 0.0;
        R1(1, 0) = 0.0;
        R1(1, 1) = cos(rpy[0]);
        R1(1, 2) = -sin(rpy[0]);
        R1(2, 0) = 0.0;
        R1(2, 1) = -R1(1, 2);
        R1(2, 2) = R1(1, 1);

        Eigen::Matrix3d R2;
        R2(0, 0) = cos(rpy[1]);
        R2(0, 1) = 0.0;
        R2(0, 2) = sin(rpy[1]);
        R2(1, 0) = 0.0;
        R2(1, 1) = 1.0;
        R2(1, 2) = 0.0;
        R2(2, 0) = -R2(0, 2);
        R2(2, 1) = 0.0;
        R2(2, 2) = R2(0, 0);

        Eigen::Matrix3d R3;
        R3(0, 0) = cos(rpy[2]);
        R3(0, 1) = -sin(rpy[2]);
        R3(0, 2) = 0.0;
        R3(1, 0) = -R3(0, 1);
        R3(1, 1) = R3(0, 0);
        R3(1, 2) = 0.0;
        R3(2, 0) = 0.0;
        R3(2, 1) = 0.0;
        R3(2, 2) = 1.0;

        return R3 * R2 * R1;
    }


    inline Vector6d toVector6d(Eigen::Matrix4d &matT) {

        Eigen::Matrix3d rot = matT.block(0, 0, 3, 3);
        Eigen::Vector3d angle = dcm2rpy(rot);
        Eigen::Vector3d trans(matT(0, 3), matT(1, 3), matT(2, 3));
        Vector6d pose;
        pose(0) = trans(0);
        pose(1) = trans(1);
        pose(2) = trans(2);
        pose(3) = angle(0) * 180 / M_PI;
        pose(4) = angle(1) * 180 / M_PI;
        pose(5) = angle(2) * 180 / M_PI;
        return pose;
    }

    inline Eigen::Isometry3d rosOdoMsg2Iso3d(nav_msgs::Odometry &odom) {

        Eigen::Quaterniond qua(odom.pose.pose.orientation.w,
                               odom.pose.pose.orientation.x,
                               odom.pose.pose.orientation.y,
                               odom.pose.pose.orientation.z);
        Eigen::Vector3d trans(odom.pose.pose.position.x,
                              odom.pose.pose.position.y,
                              odom.pose.pose.position.z);

        Eigen::Isometry3d pose;
        pose.setIdentity();
        pose.translate(trans);
        pose.rotate(qua);

        return pose;

    }

    inline Eigen::Matrix3f Euler2DCM(float roll, float pitch, float yaw) {
//        输入的角度是弧度制
        Eigen::Vector3f ea(yaw, pitch, roll);
        //欧拉角转换为旋转矩阵
        Eigen::Matrix3f rotation_matrix3;
        rotation_matrix3 = Eigen::AngleAxisf(ea[0], Eigen::Vector3f::UnitZ()) *
                           Eigen::AngleAxisf(ea[1], Eigen::Vector3f::UnitY()) *
                           Eigen::AngleAxisf(ea[2], Eigen::Vector3f::UnitX());
        return rotation_matrix3;
    }

    template <typename T>
    inline Eigen::Matrix<T,4,4> EigenIsoInv(const Eigen::Matrix<T,4,4> &Tcw) {
        Eigen::Matrix<T,3,3> Rcw = Tcw.block(0, 0, 3, 3);
        Eigen::Matrix<T,3,1> tcw = Tcw.block(0, 3, 3, 1);
        Eigen::Matrix<T,3,3> Rwc = Rcw.transpose();
        Eigen::Matrix<T,3,1> twc = -Rwc * tcw;

        Eigen::Matrix<T,4,4> Twc = Eigen::Matrix<T,4,4>::Identity();

        Twc.block(0, 0, 3, 3) = Rwc;
        Twc.block(0, 3, 3, 1) = twc;

        return Twc;
    }


    inline Eigen::Matrix4d rosOdoMsg2Mat4d(const nav_msgs::Odometry &odom) {

        Eigen::Quaterniond qua(odom.pose.pose.orientation.w,
                               odom.pose.pose.orientation.x,
                               odom.pose.pose.orientation.y,
                               odom.pose.pose.orientation.z);
        Eigen::Vector3d trans(odom.pose.pose.position.x,
                              odom.pose.pose.position.y,
                              odom.pose.pose.position.z);

        Eigen::Matrix4d odoPose;
        odoPose.setIdentity();
        odoPose.block(0, 0, 3, 3) = qua.toRotationMatrix();
        odoPose(0, 3) = trans(0);
        odoPose(1, 3) = trans(1);
        odoPose(2, 3) = trans(2);

        return odoPose;

    }

    inline void Mat4d2OdoMsg(Eigen::Matrix4d &pose, nav_msgs::Odometry &poseRos) {

        Eigen::Vector3d transV(pose(0, 3), pose(1, 3), pose(2, 3));
        Eigen::Matrix3d rotMat = pose.block(0, 0, 3, 3);
        Eigen::Quaterniond qua(rotMat);

        poseRos.pose.pose.position.x = transV(0);
        poseRos.pose.pose.position.y = transV(1);
        poseRos.pose.pose.position.z = transV(2);
        poseRos.pose.pose.orientation.x = qua.x();
        poseRos.pose.pose.orientation.y = qua.y();
        poseRos.pose.pose.orientation.z = qua.z();
        poseRos.pose.pose.orientation.w = qua.w();

    }

    inline Eigen::Matrix4d rosGeoMsg2Mat4d(const geometry_msgs::PoseStamped &Pose) {


        Eigen::Quaterniond qua(Pose.pose.orientation.w,
                               Pose.pose.orientation.x,
                               Pose.pose.orientation.y,
                               Pose.pose.orientation.z);
        Eigen::Vector3d trans(Pose.pose.position.x,
                              Pose.pose.position.y,
                              Pose.pose.position.z);

        Eigen::Matrix4d pose;
        pose.setIdentity();
        pose.block(0, 0, 3, 3) = qua.toRotationMatrix();
        pose(0, 3) = trans(0);
        pose(1, 3) = trans(1);
        pose(2, 3) = trans(2);

        return pose;

    };

    inline geometry_msgs::PoseStamped Mat4d2rosGeoMsg(Eigen::Matrix4d &Pose) {


        Eigen::Matrix3d rot = Pose.block(0, 0, 3, 3);

        Eigen::Quaterniond qua(rot);

        geometry_msgs::PoseStamped poseMsg;

        poseMsg.pose.position.x = Pose(0, 3);
        poseMsg.pose.position.y = Pose(1, 3);
        poseMsg.pose.position.z = Pose(2, 3);

        poseMsg.pose.orientation.x = qua.x();
        poseMsg.pose.orientation.y = qua.y();
        poseMsg.pose.orientation.z = qua.z();
        poseMsg.pose.orientation.w = qua.w();

        return poseMsg;

    };

    inline Eigen::Isometry3d Mat4d2Iso3d(Eigen::Matrix4d &mat) {

        Eigen::Isometry3d matIso;
        matIso.setIdentity();

        Eigen::Matrix3d rotMat = mat.block(0, 0, 3, 3);
        Eigen::Vector3d trans(mat(0, 3), mat(1, 3), mat(2, 3));

        matIso.translate(trans);
        matIso.rotate(rotMat);

        return matIso;
    }

    inline Eigen::Matrix4d Iso3d2Mat4d(Eigen::Isometry3d &iso) {

        Eigen::Matrix3d rot = iso.rotation();
        Eigen::Vector3d trans = iso.translation();
        Eigen::Matrix4d mat;
        mat.setIdentity();
        mat.block(0, 0, 3, 3) = rot;
        mat(0, 3) = trans(0);
        mat(1, 3) = trans(1);
        mat(2, 3) = trans(2);

        return mat;

    }

    inline cv::Mat toCVMat(Eigen::Matrix4d &eMat) {

        cv::Mat cvMat(4, 4, CV_32FC1);

        for (int i = 0; i < 4; i++)

            for (int j = 0; j < 4; j++)

                cvMat.at<float>(i, j) = eMat(i, j);

        return cvMat;

    }

    inline Eigen::Matrix4d toEiMat(cv::Mat &cvMat) {

        Eigen::Matrix4d eMat;
        eMat.setIdentity();

        for (int i = 0; i < 4; i++)

            for (int j = 0; j < 4; j++)

                eMat(i, j) = cvMat.at<float>(i, j);

        return eMat;

    }

    inline void transformImuBase(const sensor_msgs::Imu imuIn, sensor_msgs::Imu &imuOut, Eigen::Matrix4f T)
    {
        Eigen::Matrix3f R = T.block<3,3>(0,0);
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn.orientation, orientation);
        Eigen::Quaternionf orientation_eigen(orientation.w(),orientation.x(),orientation.y(),orientation.z());
        orientation_eigen = R * orientation_eigen.toRotationMatrix() * R.transpose();

        Eigen::Vector3f linear_acceleration(imuIn.linear_acceleration.x,imuIn.linear_acceleration.y,imuIn.linear_acceleration.z);
        Eigen::Vector3f angular_velocity(imuIn.angular_velocity.x,imuIn.angular_velocity.y,imuIn.angular_velocity.z);
        angular_velocity = R * angular_velocity;
        linear_acceleration = R * linear_acceleration;

        orientation = tf::Quaternion(orientation_eigen.x(), orientation_eigen.y(), orientation_eigen.z(), orientation_eigen.w());
        tf::quaternionTFToMsg(orientation, imuOut.orientation);
        imuOut.header = imuIn.header;
        imuOut.angular_velocity.x = angular_velocity[0];
        imuOut.angular_velocity.y = angular_velocity[1];
        imuOut.angular_velocity.z = angular_velocity[2];
        imuOut.linear_acceleration.x = linear_acceleration[0];
        imuOut.linear_acceleration.y = linear_acceleration[1];
        imuOut.linear_acceleration.z = linear_acceleration[2];
        imuOut.angular_velocity_covariance = imuIn.angular_velocity_covariance;
        imuOut.orientation_covariance = imuIn.orientation_covariance;
    }

    inline CloudData::CLOUD_PTR transformPointCloud(const CloudData::CLOUD_PTR cloudIn, CloudData::CLOUD_PTR cloudOut,
                                                       const Eigen::Matrix4f transformIn) {

        CloudData::POINT *pointFrom;
        CloudData::POINT pointTo;

        int cloudSize = cloudIn->points.size();

        if (cloudIn == cloudOut)  // cloud_in = cloud_out
        {
            CloudData::CLOUD output_temp;
            output_temp.clear();
            output_temp.resize(cloudSize);
            for (int i = 0; i < cloudSize; ++i) {
                pointFrom = &cloudIn->points[i];
                pointTo.x = transformIn(0, 0) * pointFrom->x + transformIn(0, 1) * pointFrom->y +
                            transformIn(0, 2) * pointFrom->z + transformIn(0, 3);
                pointTo.y = transformIn(1, 0) * pointFrom->x + transformIn(1, 1) * pointFrom->y +
                            transformIn(1, 2) * pointFrom->z + transformIn(1, 3);
                pointTo.z = transformIn(2, 0) * pointFrom->x + transformIn(2, 1) * pointFrom->y +
                            transformIn(2, 2) * pointFrom->z + transformIn(2, 3);
                pointTo.intensity = pointFrom->intensity;
                output_temp.points[i] = pointTo;
            }
            output_temp.header = cloudIn->header;
            output_temp.sensor_origin_ = cloudIn->sensor_origin_;
            output_temp.sensor_orientation_ = cloudIn->sensor_orientation_;
            pcl::copyPointCloud (output_temp, *cloudOut);   //output_temp类型 pcl::PointCloud<POINT>;
        }
        else
        {
            cloudOut->clear();
            cloudOut->resize(cloudSize);
            for (int i = 0; i < cloudSize; ++i) {
                pointFrom = &cloudIn->points[i];
                pointTo.x = transformIn(0, 0) * pointFrom->x + transformIn(0, 1) * pointFrom->y +
                            transformIn(0, 2) * pointFrom->z + transformIn(0, 3);
                pointTo.y = transformIn(1, 0) * pointFrom->x + transformIn(1, 1) * pointFrom->y +
                            transformIn(1, 2) * pointFrom->z + transformIn(1, 3);
                pointTo.z = transformIn(2, 0) * pointFrom->x + transformIn(2, 1) * pointFrom->y +
                            transformIn(2, 2) * pointFrom->z + transformIn(2, 3);
                pointTo.intensity = pointFrom->intensity;
                cloudOut->points[i] = pointTo;
            }
            cloudOut->header = cloudIn->header;
            cloudOut->sensor_origin_ = cloudIn->sensor_origin_;
            cloudOut->sensor_orientation_ = cloudIn->sensor_orientation_;
        }

        return cloudOut;
    }

}

#endif //PROJECT_COMMONFUNC_H
