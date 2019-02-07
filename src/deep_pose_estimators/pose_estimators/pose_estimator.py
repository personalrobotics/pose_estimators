import abc


class PoseEstimator(object):
    """
    PoseEstimator is an abstract class for detection in camera frame.
    It optionally  publishes a marker topic which contains pose and
    any other auxiliary information (e.g. object class).

    It supports either directly calling detect_objects or publishing a ROS topic.

    PerceptionModule is expected to either directly call detect_objects
    or subscribe to the ROS topic.

    Note that some C++-based pose estimator may not support directly calling
    detect_objects; those estimators are expected to be launched as a separate
    server and publish to a ROS marker topic.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def detect_objects(self):
        raise NotImplementedError
