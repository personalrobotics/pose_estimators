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


class RcnnMoped:
    class Model:
        def __init__(self, point3f_list, descriptors):
            self.point3f_list = point3f_list
            self.descriptors = descriptors

    def __init__(self, title='rcnn_moped', base_dir=None,
                 image_topic_name=None):
        self.title = title
        self.base_dir = base_dir
        self.image_topic_name = image_topic_name

        self.img_msg = None
        self.models = None
        self.sess = None
        self.graph = None
        self.category_index = None

        self.init_image_subscriber()
        self.pub_img = rospy.Publisher(
            'rcnn_moped/detection_image',
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

    def load_models(self):
        models = dict()
        for filename in os.listdir(os.path.join(self.base_dir, 'db/can')):
            filepath = os.path.join(self.base_dir, 'db/can', filename)
            if os.path.isfile(filepath):
                print(filepath)
                fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
                new_model = self.Model(fs.getNode('points_3d').mat(),
                                       fs.getNode('descriptors').mat())
                models[filename[:-4]] = new_model
        return models

    def get_feature_points(self, img, alg='SIFT', num_key_points=300):
        if alg == 'SURF':
            surf = cv2.xfeatures2d.SURF_create(num_key_points)
            kp, des = surf.detectAndCompute(img, None)
        else:  # alg == 'SIFT'
            sift = cv2.xfeatures2d.SIFT_create(num_key_points)
            kp, des = sift.detectAndCompute(img, None)
        # rst = cv2.drawKeypoints(img, kp, None)
        return kp, des

    def get_knn_matches(self, des1, des2,
                        trees=5, checks=128, threshold=0.78):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
        search_params = dict(checks=checks)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        matchesMask = [[0, 0] for i in range(len(matches))]

        for i, (m, n) in enumerate(matches):
            if m.distance < threshold * n.distance:
                matchesMask[i] = [1, 0]

        return matches, matchesMask

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

    def estimate_pose(self, img, mask, class_name, class_idx,
                      img_vis=None, visualize=False):
        rvec = np.array([1.0, 0.0, 0.0])
        tvec = np.array([0.0, 1.0, -1.0])

        return [rvec, tvec, class_name, class_idx]

        if class_name not in self.models:
            print('invalid class name: %s' % class_name)
            return

        model = self.models[class_name]

        num_key_points = 500
        threshold = 0.82
        camera_matrix = np.array([[574.1943359375, 0.0, 507.0], 
                                  [0.0, 574.1943359375, 278.0],
                                  [0.0, 0.0, 1.0]], dtype='double')
        dist_coeffs = np.zeros((4, 1))
        iterations_count = 500
        reprojection_error = 2.0
        confidence = 0.95
        pnp_algorithm = cv2.SOLVEPNP_DLS

        # SIFT features
        img_kp, img_des = self.get_feature_points(
            img * mask, num_key_points=num_key_points)

        # print(len(img_kp), class_name)

        # knn match
        index_params = dict(algorithm=1, trees=3)
        search_params = dict(checks=128)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        this_matches = flann.knnMatch(img_des, model.descriptors, k=2)
        # print(len(this_matches))

        matched_p3d = list()
        matched_p2d = list()
        for i, (m, n) in enumerate(this_matches):
            if m.distance < threshold * n.distance:
                matched_p3d.append(model.point3f_list[m.trainIdx])
                matched_p2d.append(img_kp[m.queryIdx].pt)
        matched_p3d = np.asarray(matched_p3d)
        matched_p2d = np.asarray(matched_p2d)

        # print(len(matched_p3d))
        if len(matched_p3d) < 6:
            print('not enough matches: %d' % len(matched_p3d))
            return

        # PnPRANSAC
        (success, rvec, tvec, inliers) = cv2.solvePnPRansac(
            matched_p3d, matched_p2d,
            camera_matrix, dist_coeffs,
            iterationsCount=iterations_count,
            reprojectionError=reprojection_error,
            confidence=confidence,
            flags=pnp_algorithm)
        if not success:
            print('pnpransac failed')
            return

        if visualize:
            points = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3],
                               [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                               [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                              dtype=np.float)
            points = points * 0.06;
            points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix,
                                          dist_coeffs)

            p = list()
            for pidx in range(len(points)):
                p.append((int(points[pidx][0][0]),
                          int(points[pidx][0][1])))

            cv2.line(img_vis, p[4], p[5], (0, 255, 200), 1)
            cv2.line(img_vis, p[5], p[6], (0, 255, 200), 1)
            cv2.line(img_vis, p[6], p[7], (0, 255, 200), 1)
            cv2.line(img_vis, p[7], p[4], (0, 255, 200), 1)
            cv2.line(img_vis, p[8], p[9], (0, 255, 200), 1)
            cv2.line(img_vis, p[9], p[10], (0, 255, 200), 1)
            cv2.line(img_vis, p[10], p[11], (0, 255, 200), 1)
            cv2.line(img_vis, p[11], p[8], (0, 255, 200), 1)
            cv2.line(img_vis, p[4], p[8], (0, 255, 200), 1)
            cv2.line(img_vis, p[5], p[9], (0, 255, 200), 1)
            cv2.line(img_vis, p[6], p[10], (0, 255, 200), 1)
            cv2.line(img_vis, p[7], p[11], (0, 255, 200), 1)

            cv2.line(img_vis, p[0], p[1], (255, 0, 0), 2)
            cv2.line(img_vis, p[0], p[2], (0, 255, 0), 2)
            cv2.line(img_vis, p[0], p[3], (0, 0, 255), 2)

        return [rvec, tvec, class_name, class_idx]

    def init_sess(self):
        ckpt_path = os.path.join(self.base_dir, 'demo_data/graph',
                                 'frozen_inference_graph.pb')
        label_path = os.path.join(self.base_dir, 'demo_data/data',
                                  'demo_label_map.pbtxt')

        num_classes = 10

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

        if self.models is None:
            self.models = self.load_models()

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

            mask = np.zeros_like(img)
            mask[txmin:txmax, tymin:tymax, :] = 1
            rst_vecs = self.estimate_pose(
                img, mask, t_class_name, t_class,
                img_vis=img_vis, visualize=True)

            if rst_vecs is not None:
                detections.append(rst_vecs)
        
        msg = self.bridge.cv2_to_imgmsg(img_vis, "rgb8")
        self.pub_img.publish(msg)

        return detections


if __name__ == '__main__':
    rospy.init_node('rcnn_moped')
    rcnn_moped = RcnnMoped(
        base_dir=os.path.join(os.path.expanduser('~'), 'Data/rcnn'),
        image_topic_name='/multisense/left/image_rect_color/compressed')

    try:
        pub = rospy.Publisher('rcnn_moped/marker_array', MarkerArray,
                              queue_size=2)
        rate = rospy.Rate(2)  # 2 hz

        while not rospy.is_shutdown():
            update_timestamp_str = 'update: %s' % rospy.get_time()
            rst = rcnn_moped.detect_objects()

            item_dict = dict()
            poses = list()
            if rst is not None:
                for item in rst:
                    if item[2] in item_dict:
                        item_dict[item[2]] += 1
                    else:
                        item_dict[item[2]] = 1
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
