#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import with_statement

import numpy as np
import os
import sys
import json
import collections
import rospy
import rospkg

from visualization_msgs.msg import Marker, MarkerArray


class TestMarkerManager:
    def __init__(self, frame_id):
        self.frame_id = frame_id

        self._marker_list = list()
        self._default_offset = 5000

    def add_object(obj_id, obj_key, x, y, z,
                   qx=0, qy=0, qz=0, qw=1,
                   marker_type=Marker.CUBE):
        info_dict = {
            'id': obj_id
        }
        marker = Marker()

        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = self._default_offset + len(self._marker_list)
        marker.ns = obj_key
        marker.text = json.dumps(info_dict)
        marker.type = marker_type
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.lifetime = rospy.Duration(0)

        self._marker_list.append(marker)

    def get_objects():
        for m in self._marker_list:
            m.header.stamp = rospy.Time.now()
        return self._marker_list


def publish_test_markers():
    node_title = 'deep_pose'
    camera_frame_id = 'camera_color_optical_frame'
    frequency = 5

    rospy.init_node(node_title)

    test_marker_manager = TestMarkerManager(frame_id=camera_frame_id)
    test_marker_manager.add_object('cantaloupe_1', 'food_item', 0.1, 0.1, 0.3)
    test_marker_manager.add_object('cantaloupe_2', 'food_item', 0.2, 0.3, 0.3)

    try:
        pub_pose = rospy.Publisher(
            '{}/marker_array'.format(node_title),
            MarkerArray,
            queue_size=1)

        rate = rospy.Rate(frequency)

        while not rospy.is_shutdown():
            pub_pose.publish(test_marker_manager.get_objects())
            rate.sleep()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    publish_test_markers()
