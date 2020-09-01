import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge


class CameraSubscriber(object):
    """
    A class which subscribes to camera topics and publishes detected images.
    """
    def __init__(self, image_topic, image_msg_type,
                 depth_image_topic, pointcloud_topic, camera_info_topic):

        self.image_topic = image_topic
        self.image_msg_type = image_msg_type
        self.depth_image_topic = depth_image_topic
        self.pointcloud_topic = pointcloud_topic
        self.camera_info_topic = camera_info_topic

        self.img_msg = None
        self.push_str = None
        
        self.depth_img_msg = None

        self.init_ros_subscribers()

        self.bridge = CvBridge()

    def init_ros_subscribers(self):
        # # subscribe image topic
        # if self.image_msg_type == 'compressed':
        #     self.img_subscriber = rospy.Subscriber(
        #         self.image_topic, CompressedImage,
        #         self.sensor_compressed_image_callback, queue_size=1)
        # else:  # raw
        #     self.img_subscriber = rospy.Subscriber(
        #         self.image_topic, Image,
        #         self.sensor_image_callback, queue_size=1)
        # print('Subscribed to {}'.format(self.image_topic))

        if self.depth_image_topic:
            # subscribe depth topic, only raw for now
            self.depth_subscriber = rospy.Subscriber(
                self.depth_image_topic, Image,
                self.sensor_depth_callback, queue_size=1)
            print('Subscribed to {}'.format(self.depth_image_topic))

        if self.pointcloud_topic:
            self.pointcloud_subscriber = rospy.Subscriber(
                self.pointcloud_topic, pc2.PointCloud2,
                self.lidar_scan_callback, queue_size=10)
            print('Subscribed to {}'.format(self.pointcloud_topic))

        # subscribe camera info topic
        self.subscriber = rospy.Subscriber(
                self.camera_info_topic, CameraInfo,
                self.camera_info_callback)
        print('Subscribed to {}'.format(self.camera_info_topic))

    def sensor_compressed_image_callback(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        new_msg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img_msg = cv2.cvtColor(new_msg, cv2.COLOR_BGR2RGB)

    def sensor_image_callback(self, ros_data):
        self.img_msg = self.bridge.imgmsg_to_cv2(ros_data, 'rgb8')

    def sensor_depth_callback(self, ros_data):
        self.depth_img_msg = self.bridge.imgmsg_to_cv2(ros_data, '16UC1')

    def camera_info_callback(self, ros_data):
        self.camera_info = ros_data
