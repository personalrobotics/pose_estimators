#!/usr/bin/python2
# Supports only python2 due to ros dependency
# Example script for running a perception module
from __future__ import absolute_import

from pose_estimators.sim_pose_estimator import SimPoseEstimator
from pose_estimators.perception_module import PerceptionModule
from pose_estimators.marker_manager import MarkerManager
from pose_estimators.run_perception_module import run_detection

import rospy


# Run in command line a static transform between the detection frame to
# destination frame, e.g.
# rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 camera_color_optical_frame <base_link> 1000

if __name__ == "__main__":
    detection_frame = "camera_color_optical_frame"
    destination_frame = "map"
    # Change to Robot Base Link, e.g.:
    # destination_frame = "j2n6s200_link_base"

    rospy.init_node("sim_perception")

    pose_estimator = SimPoseEstimator(detection_frame)
    marker_manager = MarkerManager()

    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=marker_manager,
        detection_frame_marker_topic=None,  # Not used since pose estimator is provided.
        detection_frame=detection_frame,
        destination_frame=destination_frame,
        purge_all_markers_per_update=False)

    destination_frame_marker_topic = "simulated_pose"
    frequency = 5
    run_detection(destination_frame_marker_topic, frequency, perception_module)
