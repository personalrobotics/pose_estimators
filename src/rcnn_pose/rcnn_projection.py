#!/usr/bin/env python

import numpy as np
import os
import tensorflow as tf
import cv2
import rospy

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from scipy import misc
from matplotlib import pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util


class RcnnProjection:
    class Model:
        def __init__(self, point3f_list, descriptors):
            self.point3f_list = point3f_list
            self.descriptors = descriptors

    def __init__(self, title='rcnn_pose', base_dir=None,
                 image_topic_name=None):
        self.title = title
        self.base_dir = base_dir
        self.image_topic_name = image_topic_name

        self.img_msg = None
        self.sess = None
        self.graph = None
        self.category_index = None

        self.init_image_subscriber()
        self.pub_img = rospy.Publisher(
            '%s/detection_image' % self.title,
            Image, queue_size=2)
        self.bridge = CvBridge()

    def init_image_subscriber(self):
        self.subscriber = rospy.Subscriber(
            self.image_topic_name, CompressedImage,
            self.sensor_image_callback, queue_size=1)
        print('subscribed to %s' % self.image_topic_name)
        # rospy.wait_for_message(self.image_topic_name, CompressedImage)

    def sensor_image_callback(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        self.img_msg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img_msg = cv2.cvtColor(self.img_msg, cv2.COLOR_BGR2RGB)

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
        tf_config.gpu_options.allow_growth = False

        self.graph.as_default()
        self.sess = tf.Session(graph=self.graph, config=tf_config)

    def detect_objects(self):
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

        for box_idx in range(scores.shape[1]):
            if scores[0][box_idx] < 0.5:
                break
            t_class = classes[0][box_idx]
            t_class_name = self.category_index[t_class]['name']
            t_box = boxes[0][box_idx]
            txmin = int(t_box[0] * img.shape[0])
            tymin = int(t_box[1] * img.shape[1])
            txmax = int(t_box[2] * img.shape[0])
            tymax = int(t_box[3] * img.shape[1])

            pt = [txmax, (tymax + tymin) * 0.5]
            rvec = np.array([1.05, 0.0, 0.0])
            z0 = 1.15
            y0 = (z0 / cam_fy) * (pt[0] - cam_cy)
            tan_theta = np.tan(30 * np.pi / 180.)
            tan_alpha = -y0 / z0
            tz = z0 - (y0 / (tan_theta - tan_alpha))
            # ty = tan_theta * y0 / (tan_theta - tan_alpha) 
            tx = (tz / cam_fx) * (pt[1] - cam_cx)
            ty = (tz / cam_fy) * (pt[0] - cam_cy)
            tvec = np.array([tx, ty, tz])

            rst_vecs = [rvec, tvec, t_class_name, t_class]

            if rst_vecs is not None:
                detections.append(rst_vecs)
        
        msg = self.bridge.cv2_to_imgmsg(img_vis, "rgb8")
        self.pub_img.publish(msg)

        return detections


if __name__ == '__main__':
    title = 'rcnn_demo'
    rospy.init_node(title)
    rcnn_demo = RcnnProjection(
        title=title,
        base_dir=os.path.join(os.path.expanduser('~'), 'Data/rcnn'),
        image_topic_name='/multisense/left/image_rect_color/compressed')

    try:
        pub = rospy.Publisher('%s/marker_array' % title, MarkerArray,
                              queue_size=2)
        rate = rospy.Rate(2)  # 2 hz

        while not rospy.is_shutdown():
            update_timestamp_str = 'update: %s' % rospy.get_time()
            rst = rcnn_demo.detect_objects()

            item_dict = dict()
            poses = list()
            if rst is not None:
                for item in rst:
                    if item[2] in item_dict:
                        item_dict[item[2]] += 1
                    else:
                        item_dict[item[2]] = 1
                    item_dict[item[2]]
                    pose = Marker()
                    pose.header.frame_id = 'multisense/left_camera_optical_frame'
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

            pub.publish(poses)

            rospy.loginfo(update_timestamp_str)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

# End of script
