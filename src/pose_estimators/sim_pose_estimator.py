#!/usr/bin/env python
from __future__ import absolute_import

import rospy
import numpy
from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem


class SimPoseEstimator(PoseEstimator):
    def __init__(self, frame_id):
        pose1 = numpy.eye(4)
        pose1[0:3, 3] = [0.3, 0.4, 0.5]
        self.item1 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="cantaloupe",
            marker_id=1,
            db_key="food_item",
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
            db_key="food_item",
            pose=pose2,
            detected_time=rospy.Time.now())

    def detect_objects(self):
        self.item1.detected_time = rospy.Time.now()
        self.item2.detected_time = rospy.Time.now()
        return [self.item1, self.item2]
