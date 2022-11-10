find_package(catkin REQUIRED COMPONENTS
        tf
        roscpp
        rospy
        )
catkin_package()

include_directories(
        ${catkin_INCLUDE_DIRS}
)

list(APPEND ALL_TARGET_LIBRARIES ${catkin_LIBRARIES})
