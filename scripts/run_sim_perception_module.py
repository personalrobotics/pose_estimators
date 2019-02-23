#!/usr/bin/python
# Example script for running a perception module

from deep_pose_estimators.pose_estimators import SimPoseEstimator
from deep_pose_estimators.perception_module import PerceptionModule
from deep_pose_estimators.marker_manager import MarkerManager
from deep_pose_estimators.run_perception_module import run_detection

import rospy


# Run in command line a static transform between the detection frame to
# destination frame, e.g.
# rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 camera_color_optical_frame <base_link> 1000

if __name__ == "__main__":
    # Normally these would be reversed
    # But here, we want to hard-code our test points in
    #   world frame for convenience.
    # And send them to Aikido as if they were coming
    #   from the camera.
    detection_frame = "map"
    destination_frame = "camera_color_optical_frame"

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
