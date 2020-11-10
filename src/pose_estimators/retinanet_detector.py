#!/usr/bin/env python

import numpy as np
import rospy
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import torch
from tf.transformations import quaternion_matrix, quaternion_from_euler

from pose_estimator import PoseEstimator
from detected_item import DetectedItem
from utils import CameraSubscriber, ImagePublisher
from utils.retina_utils import load_retinanet, load_label_map

def crop_image(txmin, txmax, tymin, tymax, xlim, ylim, my_img):
    """ Crop an image, robust to boundary overflow
    """
    txmin = int(max(txmin, 0))
    txmax = int(min(txmax, xlim))
    tymin = int(max(tymin, 0))
    tymax = int(min(tymax, ylim))
    return my_img[tymin:tymax,txmin:txmax]

class RetinaNetDetector(PoseEstimator, CameraSubscriber, ImagePublisher):
    def __init__(
            self,
            retinanet_checkpoint,
            label_map_file,
            destination_frame,
            db_key,
            use_cuda=torch.cuda.is_available(),
            node_name=rospy.get_name(),
            camera_tilt=1e-5,
            camera_to_table=0, # if depth fails, use this to calculate distance
            image_topic='/camera/color/image_raw/compressed',
            image_msg_type='compressed',
            depth_image_topic='/camera/aligned_depth_to_color/image_raw',
            camera_info_topic='/camera/color/camera_info',
            detection_frame='camera_color_optical_frame',
            timeout=1.0,
            bbox_offset=5):

        PoseEstimator.__init__(self)
        CameraSubscriber.__init__(
            self,
            image_topic=image_topic,
            image_msg_type=image_msg_type,
            depth_image_topic=depth_image_topic,
            pointcloud_topic=None,
            camera_info_topic=camera_info_topic)
        ImagePublisher.__init__(self, node_name)

        self.destination_frame = destination_frame
        self.db_key = db_key
        self.timeout = timeout
        self.bbox_offset = bbox_offset
        self.use_cuda = use_cuda
        self.camera_tilt = camera_tilt
        self.camera_to_table = camera_to_table

        self.retinanet, self.retinanet_transform, self.encoder = \
            load_retinanet(use_cuda, retinanet_checkpoint)
        self.label_map = load_label_map(label_map_file)
        self.label_names = self.label_map.values()

        # Keeps track of previously detected items'
        # center of bounding boxes, class names and associate each
        # box x class with a unique id
        # Used for tracking (temporarily)
        self.detected_item_boxes = dict()
        for item in self.label_names:
            self.detected_item_boxes[item] = dict()

    def create_detected_item(self, rvec, tvec, detected_class_name, box_id,
                             db_key):
        pose = quaternion_matrix(rvec)
        pose[:3, 3] = tvec
        return DetectedItem(
            frame_id=self.destination_frame,
            marker_namespace=detected_class_name,
            marker_id=box_id,
            db_key=db_key,
            pose=pose,
            detected_time=rospy.Time.now())

    def find_closest_box_and_update(self, x, y, class_name, tolerance=70):
        """
        Finds ths closest bounding box in the current list and
        updates it with the provided x, y
        @param x: Center x-position of a bounding box in 2D image
        @param y: Center y-position of a bounding box in 2D image
        @param class_name: Class name of the associated item
        @param tolerance: Pixel tolerance. If no box of same class is found
        within this tolerance, adds a new box with a new id
        @return Box id associated with the closest bounding box
        """
        min_distance = np.float('inf')
        matched_id = None
        largest_id = -1
        ids_to_delete = []
        for bid, (bx, by) in self.detected_item_boxes[class_name].iteritems():
            distance = np.linalg.norm(np.array([x, y]) - np.array([bx, by]))
            largest_id = max(largest_id, bid)
            if distance >= tolerance:
                continue
            if distance < min_distance:
                if matched_id:
                    # Pop this one, since we found a closer one
                    ids_to_delete.append(matched_id)
                min_distance = distance
                matched_id = bid

        if ids_to_delete:
            print("Delete ", ids_to_delete)
            for mid in ids_to_delete:
                self.detected_item_boxes[class_name].pop(mid)

        if matched_id is not None:
            self.detected_item_boxes[class_name][matched_id] = (x, y)
        else:
            self.detected_item_boxes[class_name][largest_id + 1] = (x, y)
            matched_id = largest_id + 1
            print("Adding a new box with id {} for {}".format(
                matched_id, class_name))

        return matched_id

    def get_index_of_class_name(self, class_name):
        for index, name in self.label_map.items():
            if name == class_name:
                return index
        return -1

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
            print('[retinanet_detector] no input RGB stream')
            return list()

        if self.depth_img is None:
            print('[retinanet_detector] no input depth stream, assume depth=1')
            self.depth_img = np.ones(self.img_msg.shape[:2])

        copied_img_msg = self.img_msg.copy()
        img = PILImage.fromarray(copied_img_msg.copy())
        depth_img = self.depth_img.copy()

        width, height = img.size

        x = self.retinanet_transform(img).unsqueeze(0)
        with torch.no_grad():
            if self.use_cuda:
                x=x.cuda()
            loc_preds, cls_preds = self.retinanet(x)

            boxes, class_ids, scores = self.encoder.decode(
                loc_preds.cpu().data.squeeze(),
                cls_preds.cpu().data.squeeze(),
                (width, height))

        if boxes is None or len(boxes) == 0:
            msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
            self.pub_img.publish(msg_img)
            return list()

        boxes = boxes.numpy() - self.bbox_offset

        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        # Projects 3D points in the camera coordinate frame to 2D pixel
        # coordinates using the focal lengths (fx, fy) and principal point
        # (cx, cy).
        camera_matrix = np.asarray(self.camera_info.K).reshape(3, 3)
        self.cam_fx = camera_matrix[0, 0]
        self.cam_fy = camera_matrix[1, 1]
        self.cam_cx = camera_matrix[0, 2]
        self.cam_cy = camera_matrix[1, 2]

        z0 = (self.camera_to_table /
              (np.cos(np.radians(90 - self.camera_tilt)) + 1e-10))

        detections = list()

        box_skipped = [False] * len(boxes)
        labels = []

        for box_idx in range(len(boxes)):
            class_id = class_ids[box_idx].item()
            detected_class_name = ('sample' if class_id == -1
                                            else self.label_map[class_id])

            txmin, tymin, txmax, tymax = boxes[box_idx]
            class_box_id = self.find_closest_box_and_update(
                    (txmin + txmax) / 2.0, (tymin + tymax) / 2.0,
                    detected_class_name)
            labels.append("{}_{}".format(detected_class_name, class_box_id))

            cropped_depth = crop_image(txmin, txmax, tymin, tymax, width, height, depth_img)
            cropped_img = crop_image(txmin, txmax, tymin, tymax, width, height, copied_img_msg)
            z0 = self.calculate_depth(cropped_depth)
            if z0 < 0:
                print("skipping " + detected_class_name + ": invalid depth")
                box_skipped[box_idx] = True
                continue

            if self.filter and self.filter(cropped_img, cropped_depth,
                                           detected_class_name, z0):
                print("skipping " + detected_class_name + ": ignored by filter")
                box_skipped[box_idx] = True
                continue

            # the pred_angle was default to be half a pi (90 degrees)
            # to rotate the TSR by 90 degree along X-axis
            # which assumes that the new Z-axis is pointing up from the camera
            x, y, z, w = quaternion_from_euler(np.pi/2, 0., 0.)
            rvec = np.array([x, y, z, w])

            # To compute the offset
            # Normalize the x y coordinate (px, py) by camera profile
            # => (px - cam_cx) / self.cam_fx;
            # Then multiple by the depth
            # For a detailed explanation:
            # http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
            px, py = self.get_point_of_interest(cropped_img, cropped_depth,
                                                detected_class_name)
            px, py = px + txmin, py + tymin

            tz = z0  # depth
            tx = (tz / self.cam_fx) * (px - self.cam_cx)
            ty = (tz / self.cam_fy) * (py - self.cam_cy)
            tvec = np.array([tx, ty, tz])

            detections.append(self.create_detected_item(
                rvec, tvec, detected_class_name, class_box_id, self.db_key))

        self.visualize_detections(img, boxes, scores, labels, box_skipped)
        return detections

    def visualize_detections(self, img, boxes, scores, labels, box_skipped=None):
        """draw detection bounding box on image and publish to ROS"""
        fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 11)
        draw = ImageDraw.Draw(img, 'RGBA')

        for idx in range(len(boxes)):
            box = boxes[idx]
            color = (255, 0, 0) if (box_skipped and box_skipped[idx]) \
                                     else (0, 255, 0)
            draw.rectangle(box, outline=color+(200,), width=3)

            item_tag = '{0}: {1:.2f}'.format(
                labels[idx],
                scores[idx])
            iw, ih = fnt.getsize(item_tag)
            ix, iy = box[:2]
            draw.rectangle((ix, iy, ix + iw, iy + ih), fill=color+(100,))
            draw.text(
                box[:2],
                item_tag,
                font=fnt, fill=(255, 255, 255, 255))

        msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
        self.pub_img.publish(msg_img)

    # Inherited class change this
    def get_point_of_interest(self, cropped_img, cropped_depth, class_name):
        """ Get the xy-coordinate of the POI given the detection results"""
        h, w, _ = cropped_img.shape
        return w * 0.5, h * 0.5

    # Inherited class change this
    def filter(self, cropped_img, cropped_depth, class_name, est_depth):
        """ Return true if the detection is invalid"""
        return False
