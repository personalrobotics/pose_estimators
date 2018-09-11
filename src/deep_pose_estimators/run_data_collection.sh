#!/bin/bash

source $(catkin locate)/devel/setup.bash

python $(rospack find deep_pose_estimators)/src/deep_pose_estimators/data_collection.py ada.json

