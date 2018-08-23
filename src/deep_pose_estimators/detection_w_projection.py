#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import with_statement

import numpy as np
import os
import sys
import json
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

rospack = rospkg.RosPack()
pkg_base = rospack.get_path('deep_pose_estimators')

sys.path.append(os.path.join(
    pkg_base, 'src/deep_pose_estimators/external'))
from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.utils.encoder import DataEncoder
from pytorch_retinanet.utils import utils

from bite_selection_package.model.spnet import SPNet


# An application for the food manipulation project
class DetectionWithProjection:
    class Model:
        def __init__(self, point3f_list, descriptors):
            self.point3f_list = point3f_list
            self.descriptors = descriptors

    def __init__(self, title='DetectionWithProjection', use_spnet=False):
        self.title = title

        self.img_msg = None
        self.depth_img_msg = None
        self.net = None
        self.transform = None
        self.label_map = None
        self.encoder = None

        self.use_spnet = use_spnet
        self.spnet = None
        self.spnet_transform = None

        self.init_ros_subscribers()

        self.pub_img = rospy.Publisher(
            '{}/detection_image'.format(self.title),
            Image,
            queue_size=2)
        self.bridge = CvBridge()

    def init_ros_subscribers(self):
        # subscribe image topic
        if conf.msg_type == 'compressed':
            self.subscriber = rospy.Subscriber(
                conf.image_topic, CompressedImage,
                self.sensor_compressed_image_callback, queue_size=1)
        else:  # raw
            self.subscriber = rospy.Subscriber(
                conf.image_topic, Image,
                self.sensor_image_callback, queue_size=1)
        print('subscribed to {}'.format(conf.image_topic))

        if (conf.depth_image_topic is not None and
                len(conf.depth_image_topic) > 0):
            # subscribe depth topic, only raw for now
            self.subscriber = rospy.Subscriber(
                conf.depth_image_topic, Image,
                self.sensor_depth_callback, queue_size=1)
            print('subscribed to {}'.format(conf.depth_image_topic))

        # subscribe camera info topic
        self.subscriber = rospy.Subscriber(
                conf.camera_info_topic, CameraInfo,
                self.camera_info_callback)
        print('subscribed to {}'.format(conf.camera_info_topic))

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

    def init_retinanet(self):
        self.net = RetinaNet(conf.num_classes)
        ckpt = torch.load(conf.checkpoint)
        self.net.load_state_dict(ckpt['net'])
        self.net.eval()
        self.net.cuda()

        print('Loaded RetinaNet.')
        print(self.net)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.encoder = DataEncoder()

    def init_spnet(self):
        self.spnet = SPNet()
        ckpt = torch.load(conf.spnet_checkpoint)
        self.spnet.load_state_dict(ckpt['net'])
        self.spnet.eval()
        self.spnet.cuda()

        self.spnet_transform = transforms.Compose([
            transforms.ToTensor()])
            # transforms.Normalize((0.562, 0.370, 0.271), (0.332, 0.302, 0.281))])

    def load_label_map_from_pkl(self):
        with open(conf.label_map, 'r') as f:
            label_map = pickle.load(f)
        assert label_map is not None, 'cannot load label map'
        self.label_map = label_map

    def load_label_map(self):
        with open(conf.label_map, 'r') as f:
            content = f.read().splitlines()
            f.close()
        assert content is not None, 'cannot find label map'

        temp = list()
        for line in content:
            line = line.strip()
            if (len(line) > 2 and
                    (line.startswith('id') or
                     line.startswith('name'))):
                temp.append(line.split(':')[1].strip())

        label_dict = dict()
        for idx in range(0, len(temp), 2):
            item_id = int(temp[idx])
            item_name = temp[idx + 1][1:-1]
            label_dict[item_name] = item_id

        self.label_map = label_dict

    def get_box_coordinates(self, box, img_shape):
        txmin = int(box[0] * img_shape[0])
        tymin = int(box[1] * img_shape[1])
        txmax = int(box[2] * img_shape[0])
        tymax = int(box[3] * img_shape[1])
        return txmin, tymin, txmax, tymax

    def calculate_depth_from_depth_image(self, xmin, ymin, xmax, ymax, dimg):
        dimg_sliced = np.array(dimg)[int(xmin):int(xmax), int(ymin):int(ymax)]
        depth = dimg_sliced.flatten()
        depth = depth[depth > 0]
        if depth is None or len(depth) == 0:
            return -1
        z0 = np.mean(depth)
        return z0 / 1000.0  # mm to m

    def calculate_depth(self, depth_img):
        depth = depth_img.flatten()
        depth = depth[depth > 0]
        depth = depth[abs(depth - np.mean(depth)) < np.std(depth)]
        if depth is None or len(depth) == 0:
            return -1
        z0 = np.mean(depth)
        return z0 / 1000.0  # mm to m

    def detect_objects(self):
        if self.img_msg is None:
            print('no input stream')
            return list()

        if self.depth_img_msg is None:
            print('no input depth stream')
            return list()

        if self.net is None:
            self.init_retinanet()

        if self.use_spnet and self.spnet is None:
            self.init_spnet()

        if self.label_map is None:
            self.load_label_map()

        img = PILImage.fromarray(self.img_msg.copy())
        depth_img = self.depth_img_msg.copy()
        # depth_img = PILImage.fromarray(depth)
        w, h = img.size

        x = self.transform(img)
        x = x.unsqueeze(0)
        with torch.no_grad():
            loc_preds, cls_preds = self.net(x.cuda())

            boxes, labels, scores = self.encoder.decode(
                loc_preds.cpu().data.squeeze(),
                cls_preds.cpu().data.squeeze(),
                (w, h))

        draw = ImageDraw.Draw(img, 'RGBA')

        if self.depth_img_msg is not None:
            # visualize depth
            depth_pixels = list(depth_img.getdata())
            depth_pixels = [depth_pixels[i * w:(i + 1) * w] for i in xrange(h)]
            for x in range(0, img.size[0]):
                for y in range(0, img.size[1]):
                    if (depth_pixels[y][x] < 0.001):
                        draw.point((x,y), fill=(0,0,0))

        if boxes is None or len(boxes) == 0:
            # print('no detection')
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
        # cam_cx = 320
        # cam_cy = 240

        rvec = np.array([0.0, 0.0, 0.0])

        z0 = (conf.camera_to_table /
              (np.cos(np.radians(90 - conf.camera_tilt)) + 1e-10))
        # z0 = conf.camera_to_table

        detections = list()

        for box_idx in range(len(boxes)):
            t_class = labels[box_idx].item()
            t_class_name = self.label_map[t_class]

            txmin, tymin, txmax, tymax = boxes[box_idx].numpy()

            cropped_depth = depth_img[tymin:tymax, txmin:txmax]

            z0 = self.calculate_depth(cropped_depth)
            if z0 < 0:
                print("skipping " + t_class_name + " because depth invalid")
                continue

            if self.use_spnet:
                cropped_img = img[tymin:tymax, txmin:txmax]
                pred_position, pred_angle = self.spnet(cropped_img)
            else:
                pred_position = np.array([0.5, 0.5])
                pred_angle = 0.0

            txoff = (txmax - txmin) * pred_position[0]
            tyoff = (tymax - tymin) * pred_position[1]
            pt = [txmin + txoff, tymin + tyoff]

            x, y, z, w = utils.angles_to_quaternion(pred_angle, 0., 0.)
            rvec = np.array([x, y, z, w])

            # y0 = (z0 / cam_fy) * (pt[1] - cam_cy)
            # tan_alpha = -y0 / z0
            tz = z0  # - (y0 / (tan_theta - tan_alpha))
            tx = (tz / cam_fx) * (pt[0] - cam_cx)
            ty = (tz / cam_fy) * (pt[1] - cam_cy)
            tvec = np.array([tx, ty, tz])

            rst_vecs = [rvec, tvec, t_class_name, t_class]
            detections.append(rst_vecs)

        # visualize detections
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
    print('\t./detection_w_projection.py <config_filename (e.g. herb)>\n')


def run_detection():
    rospy.init_node(conf.node_title)
    rcnn_projection = DetectionWithProjection(
        title=conf.node_title,
        use_spnet=True)

    try:
        pub_pose = rospy.Publisher(
                '{}/marker_array'.format(conf.node_title),
                MarkerArray,
                queue_size=1)

        rate = rospy.Rate(conf.frequency)  # 1 hz

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

                    obj_info = dict()
                    obj_info['id'] = '{}_{}'.format(item[2], item_dict[item[2]])

                    pose = Marker()
                    pose.header.frame_id = conf.camera_tf
                    pose.header.stamp = rospy.Time.now()
                    pose.id = item[3]
                    pose.ns = conf.marker_ns
                    pose.text = json.dumps(obj_info)
                    pose.type = Marker.CYLINDER
                    pose.pose.position.x = item[1][0]
                    pose.pose.position.y = item[1][1]
                    pose.pose.position.z = item[1][2]
                    pose.pose.orientation.x = item[0][0]
                    pose.pose.orientation.y = item[0][1]
                    pose.pose.orientation.z = item[0][2]
                    pose.pose.orientation.w = item[0][3]
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

            #rospy.loginfo(update_timestamp_str)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
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
        exit(0)

    if config_filename.startswith('ada'):
        from robot_conf import ada as conf
    elif config_filename.startswith('herb'):
        from robot_conf import herb as conf
    else:
        print_usage('Invalid arguments')
        exit(0)

    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    run_detection()
