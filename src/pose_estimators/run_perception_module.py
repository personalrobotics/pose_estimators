#!/usr/bin/env python

import rospy

from visualization_msgs.msg import MarkerArray


# Should be called after initializing a ros node
def run_detection(destination_frame_marker_topic,
                  frequency,
                  perception_module):
    try:
        pub_pose = rospy.Publisher(
            '{}/marker_array'.format(destination_frame_marker_topic),
            MarkerArray,
            queue_size=1)

        rate = rospy.Rate(frequency)

        while not rospy.is_shutdown():
            markers = perception_module.get_detected_objects_as_markers()
            pub_pose.publish(markers)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
