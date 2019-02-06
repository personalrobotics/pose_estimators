#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import with_statement

import numpy as np
import os
import sys
import json
import cv2
import pcl
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
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

from utils import math_utils
from utils import pcl_utils

rospack = rospkg.RosPack()
pkg_base = rospack.get_path('deep_pose_estimators')

sys.path.append(os.path.join(
    pkg_base, 'src/deep_pose_estimators/external'))
from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.utils.encoder import DataEncoder
from pytorch_retinanet.utils import utils as retinanet_utils


# An application for the food manipulation project
class CanDetection:
    def __init__(self, title='CanDetection', detection_mode='lidar'):
        self.title = title
        self.detection_mode = detection_mode

        self.img_msg = None
        self.depth_img_msg = None
        self.net = None
        self.transform = None
        self.label_map = None
        self.encoder = None

        self.agg_pc_data = list()
        self.agg_pc_counter = 0
        self.camera_to_table = conf.camera_to_table
        self.buf_len = 18500
        self.pos_by_lidar = None
        self.table_models = None
        self.table_models_idx = 0

        self.init_ros_subscribers()
        self.init_ros_publishers()

        self.bridge = CvBridge()

    def init_ros_subscribers(self):
        # subscribe image topic
        if conf.msg_type == 'compressed':
            self.img_subscriber = rospy.Subscriber(
                conf.image_topic, CompressedImage,
                self.sensor_compressed_image_callback, queue_size=1)
        else:  # raw
            self.img_subscriber = rospy.Subscriber(
                conf.image_topic, Image,
                self.sensor_image_callback, queue_size=1)
        print('subscribed to {}'.format(conf.image_topic))

        if conf.depth_image_topic is not None:
            # subscribe depth topic, only raw for now
            self.depth_subscriber = rospy.Subscriber(
                conf.depth_image_topic, Image,
                self.sensor_depth_callback, queue_size=1)
            print('subscribed to {}'.format(conf.depth_image_topic))

        if (conf.pointcloud_topic is not None and
                len(conf.pointcloud_topic) > 0):
            self.pointcloud_subscriber = rospy.Subscriber(
                conf.pointcloud_topic, pc2.PointCloud2,
                self.lidar_scan_callback, queue_size=10)
            print('subscribed to {}'.format(conf.pointcloud_topic))

        # subscribe camera info topic
        self.subscriber = rospy.Subscriber(
                conf.camera_info_topic, CameraInfo,
                self.camera_info_callback)
        print('subscribed to {}'.format(conf.camera_info_topic))

    def init_ros_publishers(self):
        topic_name = '{}/detection_image'.format(self.title)
        self.pub_img = rospy.Publisher(topic_name, Image, queue_size=2)
        print('publishing {}'.format(topic_name))

        topic_name = '{}/points2_table'.format(self.title)
        self.pub_table = rospy.Publisher(
            topic_name, pc2.PointCloud2, queue_size=2)
        print('publishing {}'.format(topic_name))

        topic_name = '{}/points2_can'.format(self.title)
        self.pub_cans = rospy.Publisher(
            topic_name, pc2.PointCloud2, queue_size=2)
        print('publishing {}'.format(topic_name))

    def sensor_compressed_image_callback(self, ros_data):
        print('sensor_compressed_image_callback')
        np_arr = np.fromstring(ros_data.data, np.uint8)
        new_msg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img_msg = cv2.cvtColor(new_msg, cv2.COLOR_BGR2RGB)

    def sensor_image_callback(self, ros_data):
        print('sensor_image_callback')
        self.img_msg = self.bridge.imgmsg_to_cv2(ros_data, 'rgb8')

    def sensor_depth_callback(self, ros_data):
        self.depth_img_msg = self.bridge.imgmsg_to_cv2(ros_data, '16UC1')

    def camera_info_callback(self, ros_data):
        self.camera_info = ros_data

    def lidar_scan_callback(self, ros_data):
        for data in pc2.read_points(ros_data, skip_nans=True):
            if (data[0] > -0.5 and data[0] < 0.5 and
                    data[1] > -0.05 and data[1] < 0.5 and
                    data[2] > 0.55 and data[2] < 1.1):
                    # data[3] > 2500 and data[3] < 4000):
                self.agg_pc_data.append([data[0], data[1], data[2], data[3]])

        if len(self.agg_pc_data) > self.buf_len:
            self.agg_pc_data = self.agg_pc_data[
                len(self.agg_pc_data) - self.buf_len:]

            self.agg_pc_counter += 1
            if self.agg_pc_counter > 15:
                self.agg_pc_counter = 0
            else:
                return

            pcl_cloud = pcl.PointCloud_PointXYZI()
            pcl_cloud.from_list(self.agg_pc_data)

            # vgrid = pcl_cloud.make_voxel_grid_filter()
            # vgrid.set_leaf_size(0.02, 0.02, 0.02)
            # pcl_cloud = vgrid.filter()

            seg = pcl_cloud.make_segmenter_normals(ksearch=100)
            seg.set_optimize_coefficients(True)
            seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
            seg.set_normal_distance_weight(0.1)
            seg.set_method_type(pcl.SAC_RANSAC)
            seg.set_max_iterations(100)
            seg.set_distance_threshold(0.05)
            indices, table_model = seg.segment()
            if indices is None or table_model is None:
                return

            table_cloud = pcl_cloud.extract(indices, negative=False)

            table_model = np.array(table_model)
            table_model[table_model == np.NaN] = 0

            if self.table_models is None:
                self.table_models = list()
                for _ in range(30):
                    self.table_models.append(table_model)
                self.table_models_idx = 0
            else:
                self.table_models[self.table_models_idx] = table_model
                self.table_models_idx += 1
                if self.table_models_idx >= len(self.table_models):
                    self.table_models_idx = 0
            mean_table_model = np.mean(np.asarray(self.table_models), axis=0)

            self.camera_to_table = mean_table_model[3]  # ax + by + cz + d = 0

            pcl_cloud = pcl_cloud.extract(indices, negative=True)

            seg = pcl_cloud.make_segmenter_normals(ksearch=100)
            seg.set_optimize_coefficients(True)
            seg.set_model_type(pcl.SACMODEL_CYLINDER)
            seg.set_normal_distance_weight(0.12)
            seg.set_method_type(pcl.SAC_RANSAC)
            seg.set_max_iterations(5000)
            seg.set_distance_threshold(0.05)
            seg.set_radius_limits(0.018, 0.042)
            indices, can_model = seg.segment()
            can_cloud = pcl_cloud.extract(indices, negative=False)

            cp_list = can_cloud.to_array()
            if cp_list is None or len(cp_list) == 0:
                return

            cp_mean = np.mean(cp_list, axis=0)
            cp_mean[:3] = math_utils.project_point3d(cp_mean[:3], table_model)

            proj_cp = math_utils.project_point3d(cp_mean[:3], table_model)
            proj_orig = math_utils.project_point3d([0, 0, 0], table_model)
            dd = np.sqrt(np.sum((proj_cp - proj_orig) ** 2))
            if dd == 0:
                uv = 0
            else:
                uv = (proj_cp - proj_orig) / dd
            test_p = proj_cp + uv * 0.035
            cp_mean[:3] = math_utils.project_point3d(test_p, table_model)
            can_cloud.from_array(cp_mean.reshape(1, 4))

            self.pos_by_lidar = cp_mean[:3]

            self.pub_table.publish(pcl_utils.pcl_to_ros(table_cloud))
            self.pub_cans.publish(pcl_utils.pcl_to_ros(can_cloud))

    def init_retinanet(self):
        self.net = RetinaNet()
        ckpt = torch.load(
            conf.checkpoint,  map_location=lambda storage, loc: storage)
        self.net.load_state_dict(ckpt['net'])
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

        print('Loaded RetinaNet.')
        print(self.net)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.encoder = DataEncoder()

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
            label_dict[item_id] = item_name

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
        if self.detection_mode == 'lidar':
            return self.detect_objects_by_lidar()
        else:
            return self.detect_objects_by_image()

    def detect_objects_by_lidar(self):
        if self.pos_by_lidar is None:
            print('no detection by lidar')
            return list()

        if self.label_map is None:
            self.load_label_map()

        detections = list()

        t_class = 1  # 'can'
        t_class_name = self.label_map[t_class]

        ex = 0.0
        ey = 0.0
        ez = 120.0
        x, y, z, w = math_utils.angles_to_quaternion(ex, ey, ez)

        rvec = np.array([x, y, z, w])
        tvec = np.array(self.pos_by_lidar)

        rst_vecs = [rvec, tvec, t_class_name, t_class]
        detections.append(rst_vecs)
        return detections

    def detect_objects_by_image(self):
        if self.img_msg is None:
            print('no input rgb stream')
            return list()

        if conf.depth_image_topic is None or self.depth_img_msg is None:
            print('no input depth stream')
            return list()

        if self.net is None:
            self.init_retinanet()

        if self.label_map is None:
            self.load_label_map()

        img = PILImage.fromarray(self.img_msg.copy())
        w, h = img.size

        x = self.transform(img)
        x = x.unsqueeze(0)
        with torch.no_grad():
            if torch.cuda.is_available():
                x = x.cuda()
            loc_preds, cls_preds = self.net(x)

            boxes, labels, scores = self.encoder.decode(
                loc_preds.cpu().data.squeeze(),
                cls_preds.cpu().data.squeeze(),
                (w, h))

        draw = ImageDraw.Draw(img, 'RGBA')

        if self.depth_img_msg is not None:
            depth_img = self.depth_img_msg.copy()
            # depth_img = PILImage.fromarray(depth)

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

        z0 = (self.camera_to_table /
              (np.cos(np.radians(90 - conf.camera_tilt)) + 1e-10))
        # z0 = self.camera_to_table

        detections = list()

        for box_idx in range(len(boxes)):
            t_class = labels[box_idx].item()
            t_class_name = self.label_map[t_class]

            txmin, tymin, txmax, tymax = boxes[box_idx].numpy() - 5

            if conf.depth_image_topic is not None:
                cropped_depth = depth_img[tymin:tymax, txmin:txmax]

                z0 = self.calculate_depth(cropped_depth)
                if z0 < 0:
                    print("skipping " + t_class_name + ": invalid depth")
                    continue

            if self.use_spnet:
                cropped_img = img[tymin:tymax, txmin:txmax]
                pred_position, pred_angle = self.spnet(cropped_img)
            else:
                pred_position = np.array([0.5, 0.5])
                pred_angle = 1.65

            txoff = (txmax - txmin) * pred_position[0]
            tyoff = (tymax - tymin) * pred_position[1]
            pt = [txmin + txoff, tymin + tyoff]

            x, y, z, w = math_utils.angles_to_quaternion(
                pred_angle, 0., 0.)
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
            box = boxes[idx].numpy() - 5
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
        use_spnet=conf.use_spnet,
        detection_mode=conf.detection_mode)

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
                    # pose.ns = conf.marker_ns
                    # pose.text = json.dumps(obj_info)
                    pose.ns = '{}_{}'.format(item[2], item_dict[item[2]])
                    pose.text = 'can'
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
        print('load ada configurations')
        from robot_conf import ada as conf
    elif config_filename.startswith('herb'):
        print('load herb configurations')
        from robot_conf import herb as conf
    else:
        print_usage('Invalid arguments')
        exit(0)

    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    run_detection()
