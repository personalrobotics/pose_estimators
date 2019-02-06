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

from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from detected_item import DetectedItem
from marker_manager import MarkerManager


def run_detection_with_tracking():
    print('run_detection_with_tracking')


def run_detection():
    rospy.init_node(conf.node_title)

    estimator = conf.estimator(
        title=conf.node_title, use_cuda=conf.use_cuda)

    try:
        pub_pose = rospy.Publisher(
            '{}/marker_array'.format(conf.node_title),
            MarkerArray,
            queue_size=1)

        rate = rospy.Rate(conf.frequency)

        while not rospy.is_shutdown():
            markers = estimator.detect_objects()

            pub_pose.publish(markers.marker_list)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    args = sys.argv

    config_filename = None
    if len(args) == 2:
        config_filename = args[1]
    else:
        ros_param_name = '/pose_estimator/config_filename'
        if rospy.has_param(ros_param_name):
            config_filename = rospy.get_param(ros_param_name)

    if config_filename is None:
        print_usage('Invalid arguments')
        exit(0)

    if config_filename.startswith('ada_food_manipulation'):
        from robot_conf import ada as conf
    elif config_filename.startswith('herb_sodahandoff'):
        from robot_conf import herb as conf
    else:
        print_usage('Invalid arguments')
        exit(0)

    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    run_detection()
