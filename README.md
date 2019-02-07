# deep_pose_estimators
Perception modules with CNN based approaches for HERB


## Running a detection script on a robot

### Setup catkin environment
```
cd CATKIN_WS/src/deep_pose_estimators/scripts
source $(catkin locate)/devel/setup.bash
```

### Run a detection script
```
python run_DEMO_NAME_perception_module.py
```

## Components of deep_pose_estimators

### PoseEstimator
`PoseEstimator` is an abstract class for detection in camera frame. It optionally publishes a marker topic which contains pose and any other auxiliary information (e.g. object class).

It supports either directly calling `detect_objects()` or publishing a ROS topic.

### DetectedItem
`DetectedItem` represents an item detected by a pose estimator. It stores sensor_frame_id, uid, pose, and additional information map of an object. It can also convert a ROS `Marker` into a `DetectedItem`.

### MarkerManager
`MarkerManager` converts `DetectedItem` to a ROS `Marker` with filling up basic information for marker in addition to what `DetectedItem` provides.

`PerceptionModule` may optionally use this marker manager as a convenient way to populate markers with particular type, scale, and color.

### PerceptionModule
`PerceptionModule` is a wrapper for ROS communication with aikido's `PoseEstimatorModule`. It mainly converts detected items' poses from the detectino frame to the destination frame.

Each perception module listens to a single detection frame's marker topic and transforms the detected items into the destination frame.

It is upto the high-level user to integrate multiple perception modules' detection results.    

### run_perception_module
`run_detection()` method in this script gets a list of markers from a perception module and publish it as a ROS MarkerArray topic.


## How to build a specific demo
Please check an example demo for detecting cans on HERB: https://github.com/personalrobotics/can_detector
