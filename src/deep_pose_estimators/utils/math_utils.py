from __future__ import division

import numpy as np


# Reference: [Wikipedia] Conversion between quaternions and Euler angles
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def angles_to_quaternion(yaw, pitch, roll):
   yaw_rad = np.radians(yaw * 0.5)
   roll_rad = np.radians(roll * 0.5)
   pitch_rad = np.radians(pitch * 0.5)

   cy = np.cos(yaw_rad)
   sy = np.sin(yaw_rad)
   cr = np.cos(roll_rad)
   sr = np.sin(roll_rad)
   cp = np.cos(pitch_rad)
   sp = np.sin(pitch_rad)

   x = (cy * sr * cp) - (sy * cr * sp)
   y = (cy * cr * sp) + (sy * sr * cp)
   z = (sy * cr * cp) - (cy * sr * sp)
   w = (cy * cr * cp) + (sy * sr * sp)

   return x, y, z, w


# Reference: A softmax function for numpy
# https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
