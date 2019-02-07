# deep_pose_estimators
Deep Pose Estimators is a ROS wrapper for python-based pose detectors to publish marker topic in world frame. It is mainly used to commuicate with [aikido](https://github.com/personalrobotics/aikido)'s perception module.

## Components of deep_pose_estimators
Deep Pose Estimators is composed of four modules and a few convenience methods.

### PoseEstimator
`PoseEstimator` is an abstract class for detection in camera frame. Each application is expected to inherit PoseEstimator and implement `detect_objects` method. For example, one can create a CNNPoseEstimator which uses a trained CNN-based estimator to  detect and classify objects from image streams. Typically a pose estimator subscribes to camera image/depth topics and perform the detection when `detect_objects` is called. `detect_objects` returns _all_ detected items in the image, as a list of `DetectedItem`. It can optionally publish a marker topic which contains pose and any other auxiliary information (e.g. object class).

### DetectedItem
`DetectedItem` represents an item detected by a pose estimator. It stores sensor_frame_id, uid, pose, and additional information of an object as a dictionary. It can also convert a ROS `Marker` into a `DetectedItem`, which may be useful for extracting information from pose estimators that communicate only via ROS.

### PerceptionModule
`PerceptionModule` is a wrapper for ROS communication with aikido's `PoseEstimatorModule`. It mainly converts detected items' poses from the detection frame to the destination frame. It creates ROS markers for the transformed items by using `MarkerManager`.

Many pose estimators currently in use are one-time detectors with no tracking. For these estimators, `PerceptionModule` supports `purge_all_markers_per_update` option. If true, per every call to `get_detected_objects_as_markers` it creates a marker with `Marker.DELETEALL` (which is an indicator that all previous markers are to be removed) and prepends it to the list of currently detected markers.

### MarkerManager
`MarkerManager` has two main roles. It converts `DetectedItem` to a ROS `Marker` and fills up additional information such as Marker's type, scale, etc. In addition, it optionally supports `count_item`, which is used to provide a unique id to the detected object.

`PerceptionModule` uses this marker manager to provide unique id to the objects and to populate markers with particular type, scale, color, or any other auxiliary information.


### run_perception_module
`run_detection()` method in this script that calls `PerceptionModule` at the given frequency to get a list of markers and publish it to a ROS MarkerArray topic.

## Building a custom Pose Estimator
Each application would need to implement a `PoseEstimator` and use/inherit `MarkerManager` and `PerceptionModule`.

A application-specific pose estimator should inherit `PoseEstimator` and implement `detect_objects`. You may have to inherit the `MarkerManager` for more customized markers, or use the current one if single (type, scale, color)-markers are sufficient. The marker manager and pose estimator should be passed to initialize a `PerceptionModule`, which is then passed to `run_detection`. See `scripts/run_sim_perception_module.py` for a simple example. Each application would need a similar script.


## Running a detection script on a robot

1. Setup catkin environment and start rosmaster.
```
cd CATKIN_WS/src/deep_pose_estimators/scripts
source $(catkin locate)/devel/setup.bash
roscore
```

2. Launch the robot, camera, tf transform, etc. 

For `run_sim_perception_module.py`, you need to publish a static transform, e.g.
```
rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 camera_color_optical_frame map 1000
```

3. Run a simulated detection script
```
python run_sim_perception_module.py
```
4. Check the topic, e.g.
```
rostopic echo /simulated_pose/marker_array
```

## How to build a specific demo
Please check an example demo for [food_detector](https://github.com/personalrobotics/food_detector) or [can_detector](https://github.com/personalrobotics/can_detector).

