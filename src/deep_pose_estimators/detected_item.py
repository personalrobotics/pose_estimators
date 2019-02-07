import numpy
import rospy

from visualization_msgs.msg import Marker
from tf.transformations import quaternion_matrix


class DetectedItem(object):
    """
    Represents an item detected by a pose estimator.
    """
    def __init__(self, frame_id,
                 marker_namespace,
                 marker_id,
                 db_key,
                 pose,
                 detected_time,
                 info_map=dict()):
        self.frame_id = frame_id
        self.marker_namespace = marker_namespace
        self.marker_id = marker_id
        self.pose = pose
        self.detected_time = detected_time
        self.info_map = info_map
        self.info_map['db_key'] = db_key


    @classmethod
    def from_marker(cls, marker):
        """
        This method is to be used if a pose estimator not an internal object
        but is communicating via ROS.
        """
        if not isinstance(marker, Marker):
            raise ValueError("The provided marker is not Marker.")

        # TODO: parse from the marker_message
        frame_id, namespace, idx, db_key, info_map = cls.parse_marker_message(marker)

        marker_pose = numpy.array(quaternion_matrix([
                marker.pose.orientation.x,
                marker.pose.orientation.y,
                marker.pose.orientation.z,
                marker.pose.orientation.w]))
        marker_pose[0,3] = marker.pose.position.x
        marker_pose[1,3] = marker.pose.position.y
        marker_pose[2,3] = marker.pose.position.z

        # TODO: this currently assumes that the marker pose is
        # the reference pose of the object, but if there's any offset,
        # (e.g. marker reports edge-of-and object)
        # it should be applied here.
        # The information of offset can be found in pr_assets,
        # accesssible via db_key.
        object_pose = marker_pose

        return DetectedItem(frame_id, namespace, idx, db_key, info_map, object_pose)
