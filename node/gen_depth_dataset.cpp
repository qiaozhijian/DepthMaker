#include <iostream>
#include <string>
#include "glog/logging.h"
#include "file_manager.hpp"
#include <Eigen/Core>
#include "global_defination/global_defination.h"
#include "System.h"
#include "config.h"

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

    std::string config_file = "";
    if(argc != 2){
        LOG(ERROR)<<"Usage: ./try_mei_model path_to_config_file";
        return 1;
    }else{
        config_file = argv[1];
        LOG(INFO)<<"config file path: "<<argv[1];
    }

    rgbd_inertial_slam::System system(config_file);

    return 0;
}
