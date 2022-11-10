cmake_minimum_required(VERSION 3.5)
project(rgbd_inertial_slam)

message("start make rgbd_inertial_slam")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_STANDARD 17)

set(ALL_TARGET_LIBRARIES "")
include(cmake/boost.cmake)
include(cmake/global_defination.cmake)
include(cmake/YAML.cmake)
include(cmake/glog.cmake)
include(cmake/PCL.cmake)
include(cmake/catkin.cmake)
include(cmake/OpenCV.cmake)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${catkin_INCLUDE_DIRS}
)

file(GLOB_RECURSE ALL_SRCS "src/*.cpp")

add_executable(subs_pcl node/subs_pcl.cpp ${ALL_SRCS})
target_link_libraries(subs_pcl ${ALL_TARGET_LIBRARIES})

catkin_install_python(PROGRAMS
        node/parse_rosbag.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )


