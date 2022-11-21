#include <iostream>
#include <string>
#include "glog/logging.h"
#include "file_manager.hpp"
#include <Eigen/Core>
#include "global_defination/global_defination.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

///////主函数
int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = rgbd_inertial_slam::WORK_SPACE_PATH + "/Log";
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_prefix = true;
    FLAGS_logbufsecs = 0;
    FileManager::CreateDirectory(FLAGS_log_dir);

    camodocal::CameraPtr m_camera0 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
            "/media/qzj/Dataset/code/catkin/catkin_roadslam/src/RGBD-Inertial/config/cam0_mei.yaml");

//    仿真一个3d点
    Eigen::Vector3d p_cam(2, 1, 5);
    Eigen::Vector2d px(0.0, 0.0);
//    投影到像素坐标
    m_camera0->spaceToPlane(p_cam, px);
//    可以注意一下，像素还在图像上，图像宽度是480，高度是752
    Eigen::Vector3d p_cam2(0, 0, 0);
//    此处得到的p_cam2是一条射线
    m_camera0->liftProjective(px, p_cam2);
//    乘以网络预测的深度，得到真实的3d点
    double d = p_cam(2);
    p_cam2(0) = p_cam2(0) * d / p_cam2(2);
    p_cam2(1) = p_cam2(1) * d / p_cam2(2);
    p_cam2(2) = d;
    std::cout << std::fixed << "use d(true Z) and X_ray,Y_ray,Z_ray to recover true XYZ: " << p_cam2.transpose() << std::endl;

    return 0;
}
