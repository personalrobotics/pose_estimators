cmake_minimum_required(VERSION 2.8.3)
project(deep_pose_estimators)

find_package(catkin REQUIRED)
catkin_package()
catkin_python_setup()

install(
    PROGRAMS
        rcnn_moped.py
        rcnn_projection.py
    DESTINATION
        "${CATKIN_PACKAGE_BIN_DESTINATION}"
)
