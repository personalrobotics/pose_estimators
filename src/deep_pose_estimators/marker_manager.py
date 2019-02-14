import rospy
import yaml
import collections

from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_matrix


# TODO: When count_items is true, Markermanager is in charge of putting id to
# the markers.
# Currently, no postprocessing is done to assign the same id to the same object
# in consecutive frames.
# clear() must be called at every frame to reset the counter.
class MarkerManager(object):
    """
    Converts DetectedItem to a marker. Fills up basic information for marker
    in addition to what DetectedItem provides.
    PerceptionModule may optionally use this marker manager as a convenient way
    to populate markers with particular type, scale, and color.

    """
    def __init__(self,
                 marker_type=Marker.CUBE,
                 scale=[0.01, 0.01, 0.01],
                 color=[0.5, 1.0, 0.5, 0.1],
                 count_items=True,
                 **kwargs):
        """
        @param count_items: if True, MarkerManager is in charge of counting
        the items.

        """
        self.marker_type = marker_type
        self.scale = scale
        self.color = color
        self.kwargs = kwargs
        self.count_items = count_items

        self.item_counter = None
        self.clear()

    def clear(self):
        if self.count_items:
            self.item_counter = collections.defaultdict(int)

    def item_to_marker(self, item):
        marker = Marker()
        marker.header.frame_id = item.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = item.marker_namespace
        marker.text = yaml.dump(item.info_map)

        if self.count_items:
            self.item_counter[item.marker_namespace] += 1
            marker.id = self.item_counter[item.marker_namespace]
        else:
            marker.id = item.marker_id

        # Get the pose
        quaternion = quaternion_from_matrix(item.pose)
        marker.pose.position.x = item.pose[0, 3]
        marker.pose.position.y = item.pose[1, 3]
        marker.pose.position.z = item.pose[2, 3]
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        # Auxiliary information
        marker.type = self.marker_type
        marker.scale.x = self.scale[0]
        marker.scale.y = self.scale[1]
        marker.scale.z = self.scale[2]
        marker.color.a = self.color[0]
        marker.color.r = self.color[1]
        marker.color.g = self.color[2]
        marker.color.b = self.color[3]
        marker.lifetime = rospy.Duration(0)

        for key in self.kwargs:
            if hasattr(marker, key):
                setattr(marker, key, self.kwargs[key])

        return marker

    def get_uid_pair_list(self):
        return [uid for uid in self.get_uid_pair_list_iter()]

    def get_uid_pair_list_iter(self):
        if self.count_items:
            for ns, count in self.item_counter.items():
                for idx in range(1, count + 1):
                    yield (ns, idx)
