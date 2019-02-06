import rospy
from visualization_msgs.msg import Marker

from detected_item import DetectedItem


class MarkerManager:
    def __init__(self, title, frame_id,
                 marker_type=Marker.CUBE,
                 scale=[0.01, 0.01, 0.01],
                 color=[0.5, 1.0, 0.5, 0.1]):
        self.marker_list = list()

        self.title = title
        self.frame_id = frame_id
        self.marker_type = marker_type
        self.scale = scale
        self.color = color

    def add_item(item):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = item.namespace
        marker.id = item.id
        marker.text = item.info_map
        marker.pose.position.x = item.x
        marker.pose.position.y = item.y
        marker.pose.position.z = item.z
        marker.pose.orientation.x = item.ox
        marker.pose.orientation.y = item.oy
        marker.pose.orientation.z = item.oz
        marker.pose.orientation.w = item.ow
        marker.type = self.marker_type
        marker.scale.x = self.scale[0]
        marker.scale.y = self.scale[1]
        marker.scale.z = self.scale[2]
        marker.color.a = self.color[0]
        marker.color.r = self.color[1]
        marker.color.g = self.color[2]
        marker.color.b = self.color[3]
        marker.lifetime = rospy.Duration(0)

        self.marker_list.append(marker)

