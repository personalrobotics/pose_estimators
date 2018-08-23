''' configurations for herb '''

gpus = '0'

image_topic = '/multisense/left/image_rect_color/compressed'
msg_type = 'compressed'
depth_image_topic = None
depth_msg_type = None

camera_tf = 'multisense/left_camera_optical_frame'
camera_info_topic = '/multisense/left/image_rect_color/camera_info'

camera_to_table = 0.575
camera_tilt = 30.0

num_classes = 4
checkpoint = 'src/deep_pose_estimators/external/pytorch-retinanet/checkpoint/can_ckpt.pth'
label_map = 'src/deep_pose_estimators/external/pytorch-retinanet/data/can_data/can_label_map.pkl'

spnet_checkpoint = 'src/bite_selection_package/checkpoints/spnet_ckpt.pth'

pred_position = [0.5, 0.0]

node_title = 'deep_pose'
marker_ns = 'can'
frequency = 10

