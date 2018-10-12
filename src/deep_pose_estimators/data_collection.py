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

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CompressedImage, Image, CameraInfo

from cv_bridge import CvBridge

from utils import math_utils


rospack = rospkg.RosPack()
pkg_base = rospack.get_path('deep_pose_estimators')

config = None  # configurations saved in a json file (e.g. config/ada.json)


# An application for the food manipulation project
class DataCollection:

    def __init__(self, title='DataCollection'):
        self.title = title

        self.img_msg = None
        self.depth_img_msg = None

        self.camera_tilt = 1e-5

        self.init_ros_subscribers()

        self.pub_img_full = rospy.Publisher(
            '{}/target_image_full'.format(self.title),
            Image,
            queue_size=2)
        self.pub_img = rospy.Publisher(
            '{}/target_image'.format(self.title),
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
        new_msg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img_msg = cv2.cvtColor(new_msg, cv2.COLOR_BGR2RGB)

    def sensor_image_callback(self, ros_data):
        self.img_msg = self.bridge.imgmsg_to_cv2(ros_data, 'rgb8')

    def sensor_depth_callback(self, ros_data):
        self.depth_img_msg = self.bridge.imgmsg_to_cv2(ros_data, '16UC1')

    def camera_info_callback(self, ros_data):
        self.camera_info = ros_data

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

    def sample_cadidate(self):
        if self.img_msg is None:
            print('no input stream')
            return list()

        if self.depth_img_msg is None:
            print('no input depth stream')
            self.depth_img_msg = np.ones(self.img_msg.shape[:2])
            # return list()

        copied_img_msg = self.img_msg.copy()
        img = PILImage.fromarray(copied_img_msg.copy())
        depth_img = self.depth_img_msg.copy()
        # depth_img = PILImage.fromarray(depth)
        width, height = img.size


        xoffset = 55
        yoffset = -35
        cropped_area_size = 110
        xmin = int((width - cropped_area_size) * 0.5) + xoffset
        ymin = int((height - cropped_area_size) * 0.5) + yoffset
        xmax = xmin + cropped_area_size
        ymax = ymin + cropped_area_size

        box = [xmin, ymin, xmax, ymax]

        sp_pose = [0.5, 0.5]
        sp_angle = 45.0

        draw = ImageDraw.Draw(img, 'RGBA')

        camera_matrix = np.asarray(self.camera_info.K).reshape(3, 3)
        cam_fx = camera_matrix[0, 0]
        cam_fy = camera_matrix[1, 1]
        cam_cx = camera_matrix[0, 2]
        cam_cy = camera_matrix[1, 2]

        rvec = np.array([0.0, 0.0, 0.0])

        z0 = config.camera_to_table

        detections = list()

        bbox_offset = 5

        t_class = 1
        t_class_name = 'sample'
        txmin, tymin, txmax, tymax = box

        cropped_img = copied_img_msg[
            int(tymin):int(tymax), int(txmin):int(txmax)]

        cropped_depth = depth_img[int(tymin):int(tymax), int(txmin):int(txmax)]
        z0 = self.calculate_depth(cropped_depth)

        box_key = '{}_{}_{}_{}'.format(
            t_class_name, int(txmin), int(tymin), 0)
        this_pos = sp_pose
        this_ang = sp_angle

        txoff = (txmax - txmin) * this_pos[0]
        tyoff = (tymax - tymin) * this_pos[1]
        pt = [txmin + txoff, tymin + tyoff]

        x, y, z, w = math_utils.angles_to_quaternion(
            this_ang + 90, 0., 0.)
        rvec = np.array([x, y, z, w])

        tz = z0
        tx = (tz / cam_fx) * (pt[0] - cam_cx)
        ty = (tz / cam_fy) * (pt[1] - cam_cy)
        tvec = np.array([tx, ty, tz])

        rst_vecs = [rvec, tvec,
                    t_class_name, t_class,
                    box_key, 0]
        detections.append(rst_vecs)

        # visualize detections
        fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 11)
        draw.rectangle(box, outline=(255, 0, 0, 200))

        item_tag = 'sample'
        iw, ih = fnt.getsize(item_tag)
        ix, iy = box[:2]
        draw.rectangle((ix, iy, ix + iw, iy + ih), fill=(255, 0, 0, 100))
        draw.text(
            box[:2],
            item_tag,
            font=fnt, fill=(255, 255, 255, 255))

        msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
        self.pub_img_full.publish(msg_img)

        msg_img = self.bridge.cv2_to_imgmsg(np.array(cropped_img), "rgb8")
        self.pub_img.publish(msg_img)

        return detections


def print_usage(err_msg):
    print(err_msg)
    print('Usage:')
    print('\t./data_collection.py <config_filename (e.g. ada.json)>\n')


def load_configs():
    args = sys.argv

    config_filename = None
    if len(args) == 2:
        config_filename = args[1]
    else:
        ros_param_name = '/data_collector/config_filename'
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

    this_node_title = 'data_collection'
    rospy.init_node(this_node_title)
    data_collection = DataCollection(title=this_node_title)

    try:
        pub_pose = rospy.Publisher(
                '{}/marker_array'.format(this_node_title),
                MarkerArray,
                queue_size=1)

        rate = rospy.Rate(config.frequency)  # 1 hz

        while not rospy.is_shutdown():
            update_timestamp_str = 'update: {}'.format(rospy.get_time())
            rst = data_collection.sample_cadidate()

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
                    obj_info['id'] = '{}_{}'.format(
                        item[2], item_dict[item[2]])
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
