from distutils.core import setup
from Cython.Build import cythonize

from setuptools import Extension

import numpy as np


_NP_INCLUDE_DIRS = np.get_include()

ext_modules = [
    Extension(
        name='utils.cython_nms',
        sources=[
            'utils/cython_nms.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    )
]

setup(
    name='deep_pose_estimators',
    packages=['deep_pose_estimators'],
    ext_modules=cythonize(ext_modules)
)
