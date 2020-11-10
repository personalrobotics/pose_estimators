#!/usr/bin/python2
from __future__ import absolute_import

import numpy as np
import rospy
import cv2

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from tf import TransformListener

from pose_estimators.detected_item import DetectedItem
from pose_estimators.utils.ros_utils import get_transform_matrix


class PerceptionException(Exception):
    pass


class PerceptionModule(object):
    """
    PerceptionModule is a wrapper for ROS communication with aikido's
    PoseEstimatorModule. It mainly converts detected items' poses
    from the detection frame to the destination frame.

    Each perception module listens to a single detection frame's marker topic
    and transforms the detected items into the destination frame.

    It is upto the high-level user to integrate multiple perception modules'
    detection results.
    """
    def __init__(self,
                 pose_estimator,
                 marker_manager,
                 detection_frame,
                 destination_frame='map',
                 detection_frame_marker_topic=None,
                 timeout=2.0,
                 purge_all_markers_per_update=True):
        """
        This initializes a Perception module.
        @param pose_estimator If pose_estimator is provided, its detect_objects
        will be called instead of subscribing to a marker topic.
        @param destination_frame_marker_topic The ROS topic to publish markers
        to, in destination frame.
        @param detection_frame The TF frame of the camera.
        @param destination_frame The desired world TF frame
        @param detection_frame_marker_topic The ROS topic to read markers from.
        Typically the output topic for detectors.
        If this is provided, PerceptionModule will subscribe to this topic.
        @param timeout Timeout for detector or tf listener
        @param purge_all_markers_per_update If True, add a delete-all marker as
        part of the update so that all previous markers get deleted
        """
        # Only one of these should be provided
        if not (
            (pose_estimator is None and
                detection_frame_marker_topic is not None) or
            (pose_estimator is not None and
                detection_frame_marker_topic is None)):
            raise ValueError(
                "Only one of pose_estimator or "
                "detection_frame_marker_topic should be provided.")

        self.pose_estimator = pose_estimator
        self.detection_frame_marker_topic = detection_frame_marker_topic
        self.marker_manager = marker_manager

        self.detection_frame = detection_frame
        self.destination_frame = destination_frame

        self.listener = TransformListener()
        self.timeout = timeout
        self.purge_all_markers_per_update = purge_all_markers_per_update

    def __str__(self):
        return self.__class__.__name__

    def get_detected_objects_as_markers(self, raw_img=None):
        """
        Returns a list of markers, each corresponding to a detected item in
        destination frame.
        @return A list of DetectedItems
        """
        markers = list()
        if self.purge_all_markers_per_update:
            # delete all markers counted by marker_manager
            for uid_ns, uid_id in self.marker_manager.get_uid_pair_list_iter():
                purge_marker = Marker()
                purge_marker.header.frame_id = self.destination_frame
                purge_marker.header.stamp = rospy.Time.now()
                purge_marker.id = uid_id
                purge_marker.ns = uid_ns
                purge_marker.type = Marker.CUBE
                purge_marker.action = Marker.DELETE
                markers.append(purge_marker)
            self.marker_manager.clear()
        
        bbox_img_msg = None
        if self.pose_estimator is None:
            marker_message = rospy.wait_for_message(
                self.marker_topic, MarkerArray, timeout=self.timeout)

            items = [DetectedItem.from_marker(marker)
                     for marker in marker_message.markers]
        else:
            items, bbox_img_msg = self.pose_estimator.detect_objects(raw_img)

        # Get the transform from destination to detection frame
        # print("self.destination_frame = ", self.destination_frame,  "self.detection_frame = ", self.detection_frame)
        frame_offset = get_transform_matrix(
            self.listener, self.destination_frame, self.detection_frame,
            self.timeout)

        # Convert items to be in desination frame
        for item in items:
            item.pose = np.dot(frame_offset, item.pose)
            item.frame_id = self.destination_frame
            markers += [self.marker_manager.item_to_marker(item)]

        return markers, bbox_img_msg
