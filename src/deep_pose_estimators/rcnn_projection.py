#!/usr/bin/env python

from __future__ import division
from __future__ import with_statement

import numpy as np
import os
import tensorflow as tf
import cv2
import rospy
import Math.isclose

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point

from cv_bridge import CvBridge
from scipy import misc
from matplotlib import pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util
from schunk_neck.srv import *


class RcnnProjection:
    class Model:
        def __init__(self, point3f_list, descriptors):
            self.point3f_list = point3f_list
            self.descriptors = descriptors

    def __init__(self, title='RcnnProjection', base_dir=None,
                 image_topic_name=None, msg_type='compressed'):
        self.title = title
        self.base_dir = base_dir
        self.image_topic_name = image_topic_name
        self.msg_type = msg_type

        self.img_msg = None
        self.sess = None
        self.graph = None
        self.category_index = None

        self.neck_service_name = '/schunk_neck/get_state'
        self.neck_tilt = 30

        self.init_image_subscriber()
        self.pub_img = rospy.Publisher(
                '%s/detection_image' % self.title,
                Image,
                queue_size=2)
        self.bridge = CvBridge()

        self.pub_people = rospy.Publisher(
                '%s/people_floatarray' % title,
                Float32MultiArray,
                queue_size=1)

    def init_image_subscriber(self):
        if self.msg_type == 'compressed':
            self.subscriber = rospy.Subscriber(
                    self.image_topic_name, CompressedImage,
                    self.sensor_compressed_image_callback, queue_size=1)
        else:  # raw
            self.subscriber = rospy.Subscriber(
                    self.image_topic_name, Image,
                    self.sensor_image_callback, queue_size=1)
        print('subscribed to %s' % self.image_topic_name)
        # rospy.wait_for_message(self.image_topic_name, CompressedImage)

    def sensor_compressed_image_callback(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        self.img_msg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img_msg = cv2.cvtColor(self.img_msg, cv2.COLOR_BGR2RGB)

    def sensor_image_callback(self, ros_data):
        self.img_msg = self.bridge.imgmsg_to_cv2(ros_data, 'rgb8')

    def visualize_boxes(self, img, boxes, classes, scores, category_index):
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
                img,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2)

    def init_sess(self):
        ckpt_path = os.path.join(self.base_dir, 'demo_data/graph',
                                 'frozen_inference_graph.pb')
        label_path = os.path.join(self.base_dir, 'demo_data/data',
                                  'demo_label_map.pbtxt')

        num_classes = 13

        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.visible_device_list = '0'
        tf_config.gpu_options.allow_growth = True

        self.graph.as_default()
        self.sess = tf.Session(graph=self.graph, config=tf_config)

    def get_box_coordinates(self, box, img_shape):
        txmin = int(box[0] * img_shape[0])
        tymin = int(box[1] * img_shape[1])
        txmax = int(box[2] * img_shape[0])
        tymax = int(box[3] * img_shape[1])
        return txmin, tymin, txmax, tymax

    def cleanup_detections(self, boxes, scores):
        box_idx_list = list()
        threshold = 0.05
        for box_idx in range(scores.shape[1]):
            if scores[0][box_idx] < 0.5:
                break
            t_box = boxes[0][box_idx]
            is_overlapped = False
            for prev_idx in box_idx_list:
                p_box = boxes[0][prev_idx]
                if (abs(p_box[0] - t_box[0]) < threshold
                        and abs(p_box[1] - t_box[1]) < threshold
                        and abs(p_box[2] - t_box[2]) < threshold
                        and abs(p_box[3] - t_box[3]) < threshold):
                    is_overlapped = True
                    break
            if not is_overlapped:
                box_idx_list.append(box_idx)

        return box_idx_list

    def update_neck_states(self):
        rospy.wait_for_service(self.neck_service_name)
        try:
            neck_states = rospy.ServiceProxy(
                    self.neck_service_name, GetJointStates)
            resp = neck_states()
            return resp.pan, resp.tilt
        except rospy.ServiceException, e:
            print('service call failed: %s' % e)

    def detect_objects(self, target_coords):
        if self.img_msg is None:
            print('no input stream')
            return

        if self.sess is None:
            self.init_sess()

        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.graph.get_tensor_by_name('num_detections:0')

        detections = list()

        img = self.img_msg.copy()
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
                [boxes, scores,
                 classes, num_detections],
                feed_dict={image_tensor: img_expanded})

        img_vis = img.copy()

        self.visualize_boxes(img_vis, boxes, classes, scores,
                             self.category_index)

        camera_matrix = np.array([[574.1943359375, 0.0, 507.0],
                                  [0.0, 574.1943359375, 278.0],
                                  [0.0, 0.0, 1.0]], dtype='double')
        cam_fx = camera_matrix[0, 0]
        cam_fy = camera_matrix[1, 1]
        cam_cx = camera_matrix[0, 2]
        cam_cy = camera_matrix[1, 2]

        # _, self.neck_tilt = self.update_neck_states()

        neck_to_table = 0.575  # 0.5875
        # when neck_tilt = 30, z0 = 1.175
        z0 = (neck_to_table /
              (np.cos(np.radians(90 - self.neck_tilt)) + 0.1 ** 10))
        rvec = np.array([1.08, 0.0, 0.0])
        tan_theta = np.tan(self.neck_tilt * np.pi / 180.)

        box_idx_list = self.cleanup_detections(boxes, scores)
        people = list()

        for box_idx in box_idx_list:
            t_class = classes[0][box_idx]
            t_class_name = self.category_index[t_class]['name']

            if (t_class_name.startswith('can_') and
                    (self.neck_tilt > 5) and
                    (z0 < 10)):
                txmin, tymin, txmax, tymax = self.get_box_coordinates(
                        boxes[0][box_idx], img.shape)

                pt = [txmax, (tymax + tymin) * 0.5]
                y0 = (z0 / cam_fy) * (pt[0] - cam_cy)
                tan_alpha = -y0 / z0
                tz = z0 - (y0 / (tan_theta - tan_alpha))
                tx = (tz / cam_fx) * (pt[1] - cam_cx)
                ty = (tz / cam_fy) * (pt[0] - cam_cy)
                tvec = np.array([tx, ty, tz])
                if (target_coords is not None and isclose(target_coords.x, tx, rel_tol=.05) and isclose(target_coords.y, ty, rel_tol=.05) and isclose(target_coords.z, tz, rel_tol=.05):
                    cv2.line(img_vis, (txmin, tymin), (txmax, tymax), 2) 

                rst_vecs = [rvec, tvec, t_class_name, t_class]
                detections.append(rst_vecs)

            elif t_class_name == 'person':
                people.extend(boxes[0][box_idx])

        msg_img = self.bridge.cv2_to_imgmsg(img_vis, "rgb8")
        self.pub_img.publish(msg_img)

        msg_people = Float32MultiArray()
        msg_people.data = people
        self.pub_people.publish(msg_people)

        return detections


def print_usage(err_msg):
    print(err_msg)
    print('Usage:')
    print('\t./rcnn_projection.py <config_filename (e.g. herb.json)>\n')


def load_configs():
    import sys
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
        return None

    import rospkg
    rospack = rospkg.RosPack()
    pkg_base = rospack.get_path('deep_pose_estimators')

    config_filepath = os.path.join(
        pkg_base, 'src/deep_pose_estimators/config', config_filename)
    print('config with: {}'.format(config_filepath))

    try:
        with open(config_filepath, 'r') as f:
            import simplejson
            from easydict import EasyDict
            config = EasyDict(simplejson.loads(f.read()))
            config.base_data_dir = os.path.expanduser(config.base_data_dir)
            return config
    except EnvironmentError:
        print_usage('Cannot find config file')

    return None


class target_object_listener:
    target_coords = None

    def listener():
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber('/alexa_target', Point, callback)
        rospy.spin()

    def callback(data):
        target_coords = data
        
        
def run_detection():
    config = load_configs()
    if config is None:
        return

    rospy.init_node(config.node_title)
    rcnn_projection = RcnnProjection(
        title=config.node_title,
        base_dir=config.base_data_dir,
        image_topic_name=config.image_topic,
        msg_type=config.msg_type)

    try:
        pub_pose = rospy.Publisher(
                '%s/marker_array' % config.node_title,
                MarkerArray,
                queue_size=1)

        rate = rospy.Rate(config.frequency)  # 1 hz
        target_object_listener.listener()

        while not rospy.is_shutdown():
            update_timestamp_str = 'update: %s' % rospy.get_time()
            rst = rcnn_projection.detect_objects(target_object_listener.target_coords)

            item_dict = dict()
            poses = list()
            if rst is not None:
                for item in rst:
                    if item[2] in item_dict:
                        item_dict[item[2]] += 1
                    else:
                        item_dict[item[2]] = 1

                    pose = Marker()
                    pose.header.frame_id = config.camera_tf
                    pose.header.stamp = rospy.Time.now()
                    pose.id = item[3]
                    pose.text = item[2]
                    pose.ns = '%s_%d' % (item[2], item_dict[item[2]])
                    pose.type = Marker.CYLINDER
                    pose.pose.position.x = item[1][0]
                    pose.pose.position.y = item[1][1]
                    pose.pose.position.z = item[1][2]
                    pose.pose.orientation.x = item[0][0]
                    pose.pose.orientation.y = item[0][1]
                    pose.pose.orientation.z = item[0][2]
                    pose.pose.orientation.w = 1
                    pose.scale.x = 0.1
                    pose.scale.y = 0.1
                    pose.scale.z = 0.23
                    pose.color.a = 1.0
                    pose.color.r = 1.0
                    pose.color.g = 0.1
                    pose.color.b = 0.1
                    pose.lifetime = rospy.Duration(0)
                    poses.append(pose)

            pub_pose.publish(poses)

            rospy.loginfo(update_timestamp_str)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    run_detection()
