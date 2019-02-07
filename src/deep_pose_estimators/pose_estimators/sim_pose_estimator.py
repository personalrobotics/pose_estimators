#!/usr/bin/env python

import rospy
import numpy
from deep_pose_estimators.pose_estimators import PoseEstimator
from deep_pose_estimators.detected_item import DetectedItem

class SimPoseEstimator(PoseEstimator):
    def __init__(self, frame_id):
        pose1 = numpy.eye(4)
        pose1[0:3, 3] = [0.3, 0.4, 0.5]
        self.item1 = DetectedItem(
                        frame_id=frame_id,
                        marker_namespace="cantaloupe",
                        marker_id=1,
                        db_key="",
                        pose=pose1,
                        detected_time=rospy.Time.now())
        print(pose1)
        pose2 = numpy.eye(4)
        pose2[0:3, 3] = [0.1, 0.2, 0.3]
        print(pose2)
        self.item2 = DetectedItem(
                        frame_id=frame_id,
                        marker_namespace="cantaloupe",
                        marker_id=2,
                        db_key="",
                        pose=pose2,
                        detected_time=rospy.Time.now())

    def detect_objects(self):
        self.item1.detected_time = rospy.Time.now()
        self.item2.detected_time = rospy.Time.now()
        return [self.item1, self.item2]

