import numpy
import rospy

def get_transform_matrix(listener, source_frame, target_frame, timeout):
    """
    Returns the transformation matrix from source to target frame
    @param listener TF listener
    @param source_frame
    """
    listener.waitForTransform(
            source_frame,
            target_frame,
            rospy.Time(),
            rospy.Duration(timeout))
    frame_trans, frame_rot = listener.lookupTransform(
            source_frame,
            target_frame,
            rospy.Time(0))

    frame_offset = numpy.matrix(quaternion_matrix(frame_rot))
    frame_offset[0,3] = frame_trans[0]
    frame_offset[1,3] = frame_trans[1]
    frame_offset[2,3] = frame_trans[2]

    return frame_offset
