/*
 * @Description:
 * @Author: Ren Qian
 * @Date: 2019-07-17 18:17:49
 */
#ifndef LIDAR_LOCALIZATION_SENSOR_DATA_CLOUD_DATA_HPP_
#define LIDAR_LOCALIZATION_SENSOR_DATA_CLOUD_DATA_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "global_defination/global_defination.h"


/*
    * A point cloud type that has "ring" channel
    */
struct PointXYZIR {
    PCL_ADD_POINT4D

    PCL_ADD_INTENSITY;
    uint16_t ring;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIR,
                                   (float, x, x)(float, y, y)
                                           (float, z, z)(float, intensity, intensity)
                                           (uint16_t, ring, ring)
)

namespace rgbd_inertial_slam {

    class CloudData {
    public:
        using POINT = pcl::PointXYZI;
        using CLOUD = pcl::PointCloud<POINT>;
        using CLOUD_PTR = CLOUD::Ptr;

    public:
        CloudData()
                : cloud_ptr(new CLOUD()) {
            cloud_ptr->clear();
        }

    public:
        double time = 0.0;
        CLOUD_PTR cloud_ptr;
    };

    class CloudRingData {
    public:
        using POINT = PointXYZIR;
        using CLOUD = pcl::PointCloud<POINT>;
        using CLOUD_PTR = CLOUD::Ptr;

    public:
        CloudRingData()
                : cloud_ptr(new CLOUD()) {
            cloud_ptr->clear();
        }

    public:
        double time = 0.0;
        CLOUD_PTR cloud_ptr;
    };

    typedef CloudRingData CloudDataPre; //在matching.yaml 文件里还要改一下use_ring的值
}

#endif