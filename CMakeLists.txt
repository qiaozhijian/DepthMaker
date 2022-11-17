cmake_minimum_required(VERSION 3.5)
project(rgbd_inertial_slam)

message("start make rgbd_inertial_slam")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")

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

add_executable(gen_depth_dataset node/gen_depth_dataset.cpp ${ALL_SRCS})
target_link_libraries(gen_depth_dataset ${ALL_TARGET_LIBRARIES})

catkin_install_python(PROGRAMS
        node/parse_rosbag.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )


