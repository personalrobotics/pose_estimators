import sys
import ctypes
import struct

import pcl
import sensor_msgs.point_cloud2 as pc2


def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message

        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
    """
    points_list = list()

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])

    pcl_data = pcl.PointCloud_PointXYZRGB()
    pcl_data.from_list(points_list)

    return pcl_data


def pcl_to_ros(pcl_array, conf):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

        Args:
            pcl_array (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud

        Returns:
            PointCloud2: A ROS point cloud
    """
    ros_msg = pc2.PointCloud2()

    ros_msg.header.stamp = rospy.Time.now()
    ros_msg.header.frame_id = conf.camera_tf

    ros_msg.height = 1
    ros_msg.width = pcl_array.size

    ros_msg.fields.append(pc2.PointField(
        name="x",
        offset=0,
        datatype=pc2.PointField.FLOAT32, count=1))
    ros_msg.fields.append(pc2.PointField(
        name="y",
        offset=4,
        datatype=pc2.PointField.FLOAT32, count=1))
    ros_msg.fields.append(pc2.PointField(
        name="z",
        offset=8,
        datatype=pc2.PointField.FLOAT32, count=1))
    ros_msg.fields.append(pc2.PointField(
        name="rgb",
        offset=16,
        datatype=pc2.PointField.FLOAT32, count=1))

    ros_msg.is_bigendian = False
    ros_msg.point_step = 32
    ros_msg.row_step = ros_msg.point_step * ros_msg.width * ros_msg.height
    ros_msg.is_dense = False
    buffer = list()

    for data in pcl_array:
        s = struct.pack('>f', data[3])
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value

        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        buffer.append(struct.pack(
            'ffffBBBBIII', data[0], data[1], data[2], 1.0,
            b, g, r, 0, 0, 0, 0))

    ros_msg.data = "".join(buffer)

    return ros_msg
