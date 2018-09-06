#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import with_statement

import pdb
import numpy as np
import os
import sys
import json
import cv2
import rospy
import rospkg
import math
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

from utils import math_utils

rospack = rospkg.RosPack()
pkg_base = rospack.get_path('deep_pose_estimators')

external_path = os.path.join(
    pkg_base, 'src/deep_pose_estimators/external')
sys.path.append(external_path)
from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.utils.encoder import DataEncoder
from pytorch_retinanet.utils import utils

from bite_selection_package.model.spnet import SPNet
import time


config = None  # configurations saved in a json file (e.g. config/ada.json)


# An application for the food manipulation project
class DetectionWithProjection:
    class Model:
        def __init__(self, point3f_list, descriptors):
            self.point3f_list = point3f_list
            self.descriptors = descriptors

    def __init__(self, title='DetectionWithProjection', use_spnet=True, use_cuda=True):
        self.title = title

        self.img_msg = None
        self.depth_img_msg = None
        self.net = None
        self.transform = None
        self.label_map = None
        self.encoder = None

        self.use_spnet = use_spnet
        self.use_cuda = use_cuda
        self.spnet = None
        self.spnet_transform = None

        self.camera_tilt = 1e-5

        self.init_ros_subscribers()

        self.pub_img = rospy.Publisher(
            '{}/detection_image'.format(self.title),
            Image,
            queue_size=2)
        self.pub_spnet_img = rospy.Publisher(
            '{}/spnet_image'.format(self.title),
            Image,
            queue_size=2)
        self.bridge = CvBridge()

        self.selector_food_names = ["carrot", "melon", "apple", "banana", "strawberry"]
        self.selector_index = 0

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
        new_msg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img_msg = cv2.cvtColor(new_msg, cv2.COLOR_BGR2RGB)

    def sensor_image_callback(self, ros_data):
        self.img_msg = self.bridge.imgmsg_to_cv2(ros_data, 'rgb8')

    def sensor_depth_callback(self, ros_data):
        self.depth_img_msg = self.bridge.imgmsg_to_cv2(ros_data, '16UC1')

    def camera_info_callback(self, ros_data):
        self.camera_info = ros_data

    def init_retinanet(self):
        self.net = RetinaNet(config.num_classes)
        if self.use_cuda:
            ckpt = torch.load(config.checkpoint)
        else:
            ckpt = torch.load(config.checkpoint, map_location='cpu')
        self.net.load_state_dict(ckpt['net'])
        self.net.eval()
        if self.use_cuda:
            net = self.net.cuda()
        
        print('Loaded RetinaNet.')
        # print(self.net)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.encoder = DataEncoder()

    def init_spnet(self):
        self.spnet = SPNet()
        if self.use_cuda:
            ckpt = torch.load(config.spnet_checkpoint)
        else:
            ckpt = torch.load(config.spnet_checkpoint, map_location='cpu')
        self.spnet.load_state_dict(ckpt['net'])
        self.spnet.eval()
        if self.use_cuda:
            spnet = self.spnet.cuda()


        self.spnet_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.562, 0.370, 0.271), (0.332, 0.302, 0.281))
        ])

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

    def publish_spnet(self, sliced_img, identity, actuallyPublish = False):
        img_org = PILImage.fromarray(sliced_img.copy())

        target_size = 136
        ratio = float(target_size / max(img_org.size))
        new_size = tuple([int(x * ratio) for x in img_org.size])
        pads = [(target_size - new_size[0]) // 2,
                (target_size - new_size[1]) // 2]
        img_org = img_org.resize(new_size, PILImage.ANTIALIAS)
        img = PILImage.new('RGB', (target_size, target_size))
        img.paste(img_org, pads)

        transform = transforms.Compose([transforms.ToTensor()])
        pred_bmasks, pred_rmasks = self.spnet(torch.stack([transform(img)]).cuda())


        # print(pred_bmasks)

        angle_res = 6
        mask_size = 17
        final_size = 512

        img = img.resize((final_size,final_size), PILImage.ANTIALIAS)
        draw = ImageDraw.Draw(img, 'RGBA')
        positives = pred_bmasks[0].data.cpu().numpy() < -1
        rmask = pred_rmasks[0].data.cpu().numpy()
        rmask = np.argmax(rmask, axis=1)
        rmask = rmask * 180 / angle_res
        rmask[rmask < 0] = 0
        rmask[positives] = -1
        rmask = rmask.reshape(mask_size, mask_size)
        sp_poses = []
        sp_angles = []
        done = False
        for ri in range(mask_size):
            for ci in range(mask_size):
                rotation = rmask[ri][ci]
                if rotation >= 0:
                    ix = ci * final_size / mask_size
                    iy = ri * final_size / mask_size
                    # draw.rectangle((ix - iw, iy - ih, ix + iw, iy + ih), fill=(30, 30, 250, 255))
                    iw = -math.sin(rotation / 180.0 * 3.1415) * 5 * (final_size / target_size)
                    ih = math.cos(rotation / 180.0 * 3.1415) * 5 * (final_size / target_size)
                    # print(str(ix) + ', ' + str(iy) + ', ' + str(iw) + ', ' + str(ih))
                    draw.line((ix - iw, iy - ih, ix + iw, iy + ih), fill=(30, 30, 250, 255), width = int(float(final_size) / float(target_size)))
                    sp_poses.append([ci / float(mask_size), ri / float(mask_size)])

                    x1 = iw
                    y1 = ih
                    x2 = 0.5 - ci / float(mask_size)
                    y2 = 0.5 - ri / float(mask_size)
                    a = x1*y2 - x2*y1
                    if a > 0:
                        rotation += 180
                    sp_angles.append(rotation)
                    # done = True
                if done: break
            if done: break

        if actuallyPublish:
            msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
            self.pub_spnet_img.publish(msg_img)
        return sp_poses, sp_angles

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

        copied_img_msg = self.img_msg.copy()
        img = PILImage.fromarray(copied_img_msg.copy())
        depth_img = self.depth_img_msg.copy()
        # depth_img = PILImage.fromarray(depth)
        width, height = img.size

        x = self.transform(img)
        x = x.unsqueeze(0)
        with torch.no_grad():
            if self.use_cuda:
                loc_preds, cls_preds = self.net(x.cuda())
            else:
                loc_preds, cls_preds = self.net(x)

            boxes, labels, scores = self.encoder.decode(
                loc_preds.cpu().data.squeeze(),
                cls_preds.cpu().data.squeeze(),
                (width, height))

        # sp_poses = [[0.0, 0.0], [1.0, 1.0]]
        sp_poses = [[0.5, 0.5]]
        sp_angles = [45.0, 45.0]

        # visualize depth
        draw = ImageDraw.Draw(img, 'RGBA')
        # depth_pixels = list(depth_img.getdata())
        # depth_pixels = [depth_pixels[i * w:(i + 1) * w] for i in xrange(h)]
        # for x in range(0, img.size[0]):
        #     for y in range(0, img.size[1]):
        #         if (depth_pixels[y][x] < 0.001):
        #             draw.point((x,y), fill=(0,0,0))

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

        # z0 = (config.camera_to_table /
        #       (np.cos(np.radians(90 - self.camera_tilt)) + 0.1 ** 10))
        z0 = config.camera_to_table

        detections = list()

        bbox_offset = 5

        first_food_item = True
                    


        found = False
        spBoxIdx = -1
        for _ in range(len(self.selector_food_names)):
            for box_idx in range(len(boxes)):
                t_class = labels[box_idx].item()
                t_class_name = self.label_map[t_class]
                if (t_class_name == self.selector_food_names[self.selector_index]) or True:
                    txmin, tymin, txmax, tymax = boxes[box_idx].numpy() - bbox_offset
                    if (txmin < 0 or tymin < 0 or txmax > width or tymax > height):
                        break
                    found = True
                    spBoxIdx = box_idx
                    cropped_img = copied_img_msg[int(tymin):int(tymax), int(txmin):int(txmax)]
                    sp_poses, sp_angles = self.publish_spnet(cropped_img, t_class_name, True)
                    self.last_class_name = t_class_name
                    break

            self.selector_index = (self.selector_index + 1) % len(self.selector_food_names)
            if found:
                break
                        

        for box_idx in range(len(boxes)):
            t_class = labels[box_idx].item()
            t_class_name = self.label_map[t_class]

            txmin, tymin, txmax, tymax = boxes[box_idx].numpy() - bbox_offset
            if (txmin < 0 or tymin < 0 or txmax > width or tymax > height):
                break
            cropped_img = copied_img_msg[int(tymin):int(tymax), int(txmin):int(txmax)]
            sp_poses, sp_angles = self.publish_spnet(cropped_img, t_class_name, False)


            cropped_depth = depth_img[int(tymin):int(tymax), int(txmin):int(txmax)]

            z0 = self.calculate_depth(cropped_depth)
            if z0 < 0:
                print("skipping " + t_class_name + " because depth invalid")
                continue

            if spBoxIdx >= 0:
                
                for sp_idx in range(len(sp_poses)):
                    box_key = '{}_{}_{}_{}'.format(t_class_name, int(txmin), int(tymin), sp_idx)
                    this_pos = sp_poses[sp_idx]
                    this_ang = sp_angles[sp_idx]

                    txoff = (txmax - txmin) * this_pos[0]
                    tyoff = (tymax - tymin) * this_pos[1]
                    pt = [txmin + txoff, tymin + tyoff]

                    diff = 30
                    cropped_depth = depth_img[int(pt[1]-diff):int(pt[1]+diff), int(pt[0]-diff):int(pt[1]+diff)]
                    current_z0 = self.calculate_depth(cropped_depth)
                    if (current_z0 < 0):
                        diff = 70
                        cropped_depth = depth_img[int(pt[1]-diff):int(pt[1]+diff), int(pt[0]-diff):int(pt[1]+diff)]
                        current_z0 = self.calculate_depth(cropped_depth)
                        if current_z0 < 0:
                            current_z0 = z0

                    x, y, z, w = math_utils.angles_to_quaternion(this_ang + 90, 0., 0.)
                    rvec = np.array([x, y, z, w])

                    tz = current_z0
                    tx = (tz / cam_fx) * (pt[0] - cam_cx)
                    ty = (tz / cam_fy) * (pt[1] - cam_cy)
                    tvec = np.array([tx, ty, tz])

                    rst_vecs = [rvec, tvec, t_class_name, t_class, box_key, sp_idx]
                    detections.append(rst_vecs)

        # visualize detections
        fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 11)
        for idx in range(len(boxes)):
            box = boxes[idx].numpy() - bbox_offset
            label = labels[idx]
            draw.rectangle(box, outline=(255, 0, 0, 200))

            item_tag = '{0}: {1:.2f}'.format(
                self.label_map[label.item()],
                scores[idx])
            iw, ih = fnt.getsize(item_tag)
            ix, iy = box[:2]
            draw.rectangle((ix, iy, ix + iw, iy + ih), fill=(255, 0, 0, 100))
            draw.text(
                box[:2],
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
    rcnn_projection = DetectionWithProjection(
        title=config.node_title,
        use_spnet=True)

    try:
        pub_pose = rospy.Publisher(
                '{}/marker_array'.format(config.node_title),
                MarkerArray,
                queue_size=1)

        rate = rospy.Rate(config.frequency)  # 1 hz

        while not rospy.is_shutdown():
            update_timestamp_str = 'update: {}'.format(rospy.get_time())
            rst = rcnn_projection.detect_objects()

            poses = list()
            pose = Marker()
            pose.header.frame_id = config.camera_tf
            pose.header.stamp = rospy.Time.now()
            pose.id = 0
            pose.ns = 'food_item'
            pose.type = Marker.CUBE
            pose.action = Marker.DELETEALL
            poses.append(pose)

            item_dict = dict()
            if rst is not None:
                for item in rst:
                    if item[2] in item_dict:
                        item_dict[item[2]] += 1
                    else:
                        item_dict[item[2]] = 1

                    obj_info = dict()
                    obj_info['id'] = '{}_{}'.format(item[2], item_dict[item[2]])
                    obj_info['box_key'] = item[3] * 1000 + item[5]

                    pose = Marker()
                    pose.header.frame_id = config.camera_tf
                    pose.header.stamp = rospy.Time.now()
                    pose.id = item[3] * 1000 + item[5]
                    pose.ns = 'food_item'
                    pose.text = json.dumps(obj_info)
                    pose.type = Marker.CUBE
                    pose.pose.position.x = item[1][0]
                    pose.pose.position.y = item[1][1]
                    pose.pose.position.z = item[1][2]
                    pose.pose.orientation.x = item[0][0]
                    pose.pose.orientation.y = item[0][1]
                    pose.pose.orientation.z = item[0][2]
                    pose.pose.orientation.w = item[0][3]
                    pose.scale.x = 0.01
                    pose.scale.y = 0.01
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
    run_detection()
