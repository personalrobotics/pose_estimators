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
