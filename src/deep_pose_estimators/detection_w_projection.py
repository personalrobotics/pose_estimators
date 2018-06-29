#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import with_statement

import numpy as np
import os
import sys
import cv2
import rospy
import rospkg
try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch
import torchvision.transforms as transforms

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CompressedImage, Image, CameraInfo

from cv_bridge import CvBridge

from model.retinanet import RetinaNet
from utils.encoder import DataEncoder


config = None  # configurations saved in a json file (e.g. config/ada.json)


# An application for the food manipulation project
class DetectionWithProjection:
    class Model:
        def __init__(self, point3f_list, descriptors):
            self.point3f_list = point3f_list
            self.descriptors = descriptors

    def __init__(self, title='DetectionWithProjection'):
        self.title = title

        self.img_msg = None
        self.depth_img_msg = None
        self.net = None
        self.transform = None
        self.label_map = None
        self.encoder = None

        self.camera_tilt = 1e-5

        self.init_ros_subscribers()

        self.pub_img = rospy.Publisher(
                '{}/detection_image'.format(self.title),
                Image,
                queue_size=2)
        self.bridge = CvBridge()

    def init_ros_subscribers(self):
        # subscribe image topic
        if config.msg_type == 'compressed':
            self.subscriber = rospy.Subscriber(
                    config.image_topic, CompressedImage,
                    self.sensor_compressed_image_callback, queue_size=1)
        else:  # raw
            self.subscriber = rospy.Subscriber(
                    config.image_topic, Image,
                    self.sensor_image_callback, queue_size=1)
        print('subscribed to {}'.format(config.image_topic))

        # subscribe depth topic, only raw for now
        self.subscriber = rospy.Subscriber(
            config.depth_image_topic, Image,
                self.sensor_depth_callback, queue_size=1)
        print('subscribed to {}'.format(config.depth_image_topic))

        # subscribe camera info topic
        self.subscriber = rospy.Subscriber(
                config.camera_info_topic, CameraInfo,
                self.camera_info_callback)
        print('subscribed to {}'.format(config.camera_info_topic))

    def sensor_compressed_image_callback(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        self.img_msg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img_msg = cv2.cvtColor(self.img_msg, cv2.COLOR_BGR2RGB)

    def sensor_image_callback(self, ros_data):
        self.img_msg = self.bridge.imgmsg_to_cv2(ros_data, 'rgb8')

    def sensor_depth_callback(self, ros_data):
        self.depth_img_msg = self.bridge.imgmsg_to_cv2(ros_data, '16UC1')

    def camera_info_callback(self, ros_data):
        self.camera_info = ros_data

    def init_retinanet(self):
        self.net = RetinaNet(config.num_classes)
        ckpt = torch.load(config.checkpoint)
        self.net.load_state_dict(ckpt['net'])
        self.net.eval()
        self.net.cuda()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.encoder = DataEncoder()

    def load_label_map(self):
        with open(config.label_map, 'r') as f:
            label_map = pickle.load(f)
        assert label_map is not None, 'cannot load label map'
        self.label_map = label_map

    def get_box_coordinates(self, box, img_shape):
        txmin = int(box[0] * img_shape[0])
        tymin = int(box[1] * img_shape[1])
        txmax = int(box[2] * img_shape[0])
        tymax = int(box[3] * img_shape[1])
        return txmin, tymin, txmax, tymax

    def calculate_depth(self, xmin, ymin, xmax, ymax, dimg):
        dimg_sliced = np.array(dimg)[int(xmin):int(xmax), int(ymin):int(ymax)]
        z0 = np.mean(dimg_sliced)
        # mm to m
        return z0 / 1000.0

    def detect_objects(self):
        if self.img_msg is None:
            print('no input stream')
            return list()

    if self.depth_img_msg is None:
        print('no input depth stream')
        return list()

        if self.net is None:
            self.init_retinanet()

        if self.label_map is None:
            self.load_label_map()

        img = PILImage.fromarray(self.img_msg.copy())
    depth_img = PILImage.fromarray(self.depth_img_msg.copy())
        w, h = img.size

        x = self.transform(img)
        x = x.unsqueeze(0)
        with torch.no_grad():
            loc_preds, cls_preds = self.net(x.cuda())

            boxes, labels, scores = self.encoder.decode(
                loc_preds.cpu().data.squeeze(),
                cls_preds.cpu().data.squeeze(),
                (w, h))

        if boxes is None or len(boxes) == 0:
            print('no detection')
            msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
            self.pub_img.publish(msg_img)
            return list()

        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        # Projects 3D points in the camera coordinate frame to 2D pixel
        # coordinates using the focal lengths (fx, fy) and principal point
        # (cx, cy).
        camera_matrix = np.asarray(self.camera_info.K).reshape(3, 3)
        cam_fx = camera_matrix[0, 0]
        cam_fy = camera_matrix[1, 1]
        cam_cx = camera_matrix[0, 2]
        cam_cy = camera_matrix[1, 2]

        rvec = np.array([0.0, 0.0, 0.0])
        tan_theta = np.tan((self.camera_tilt - 90) * np.pi / 180.)

        # box_idx_list = self.cleanup_detections(boxes, scores)

        detections = list()

        # for box_idx in box_idx_list:
        for box_idx in range(len(boxes)):
            t_class = labels[box_idx].item()
            t_class_name = self.label_map[t_class]

            txmin, tymin, txmax, tymax = boxes[box_idx].numpy()
            z0 = self.calculate_depth(txmin, tymin, txmax, tymax, depth_img)

            pt = [(txmax + txmin) * 0.5, (tymax + tymin) * 0.5]
            y0 = (z0 / cam_fy) * (pt[0] - cam_cy)
            tan_alpha = -y0 / z0
            tz = z0 - (y0 / (tan_theta - tan_alpha))
            tx = (tz / cam_fx) * (pt[1] - cam_cx) + 0.045
            ty = (tz / cam_fy) * (pt[0] - cam_cy) - 0.085
            tvec = np.array([ty, tx, tz])

            rst_vecs = [rvec, tvec, t_class_name, t_class]
            detections.append(rst_vecs)

        # visualize detections
        draw = ImageDraw.Draw(img, 'RGBA')
        fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 11)
        for idx in range(len(boxes)):
            box = boxes[idx]
            label = labels[idx]
            draw.rectangle(list(box), outline=(255, 0, 0, 200))

            item_tag = '{0}: {1:.2f}'.format(
                self.label_map[label.item()],
                scores[idx])
            iw, ih = fnt.getsize(item_tag)
            ix, iy = list(box[:2])
            draw.rectangle((ix, iy, ix + iw, iy + ih), fill=(255, 0, 0, 100))
            draw.text(
                list(box[:2]),
                item_tag,
                font=fnt, fill=(255, 255, 255, 255))

        msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
        self.pub_img.publish(msg_img)

        return detections


def print_usage(err_msg):
    print(err_msg)
    print('Usage:')
    print('\t./detection_w_projection.py <config_filename (e.g. ada.json)>\n')


def load_configs():
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
            return config
    except EnvironmentError:
        print_usage('Cannot find config file')

    return None


def run_detection():
    global config
    config = load_configs()
    if config is None:
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

    rospy.init_node(config.node_title)
    rcnn_projection = DetectionWithProjection(title=config.node_title)

    try:
        pub_pose = rospy.Publisher(
                '{}/marker_array'.format(config.node_title),
                MarkerArray,
                queue_size=1)

        rate = rospy.Rate(config.frequency)  # 1 hz

        while not rospy.is_shutdown():
            update_timestamp_str = 'update: {}'.format(rospy.get_time())
            rst = rcnn_projection.detect_objects()

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
                    pose.text = 'food_item'
                    pose.ns = '{}_{}'.format(item[2], item_dict[item[2]])
                    pose.type = Marker.CUBE
                    pose.pose.position.x = item[1][0]
                    pose.pose.position.y = item[1][1]
                    pose.pose.position.z = item[1][2]
                    pose.pose.orientation.x = item[0][0]
                    pose.pose.orientation.y = item[0][1]
                    pose.pose.orientation.z = item[0][2]
                    pose.pose.orientation.w = 1
                    pose.scale.x = 0.04
                    pose.scale.y = 0.04
                    pose.scale.z = 0.04
                    pose.color.a = 0.5
                    pose.color.r = 1.0
                    pose.color.g = 0.5
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
