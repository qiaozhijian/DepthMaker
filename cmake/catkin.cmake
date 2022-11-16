find_package(catkin REQUIRED COMPONENTS
        tf
        roscpp
        rospy
        rosbag
        camera_model
        )
catkin_package()

include_directories(
        ${catkin_INCLUDE_DIRS}
)

list(APPEND ALL_TARGET_LIBRARIES ${catkin_LIBRARIES})
