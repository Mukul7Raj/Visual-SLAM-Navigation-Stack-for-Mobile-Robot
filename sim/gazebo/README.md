# Gazebo / ROS 2 bringup

These files are the **robot-in-the-loop** side of the stack. The Python SLAM
node does not need Gazebo to run (CI uses the textured-room renderer).

## Why a separate sim

ORB extraction, PnP, and bundle adjustment are easier to unit-test against a
renderer with known ground truth. Gazebo is where you show a wheeled platform,
wheel odometry, and camera extrinsics.

## Run (ROS 2 Humble + TurtleBot3)

```bash
export TURTLEBOT3_MODEL=waffle
source /opt/ros/humble/setup.bash
gazebo sim/gazebo/worlds/indoor_lab.world
# or
ros2 launch sim/gazebo/launch/visual_slam.launch.py
```

Subscribe the Python tracker to `/camera/image_raw` (BGR8). Camera YAML must
match `configs/default.yaml` after calibration (`fx, fy, cx, cy`).

## Evaluation on-robot

Record `/tf` or SLAM poses as TUM `tx ty tz qx qy qz qw` and compare with
mocap / Gazebo `ground_truth` using `vslam.evaluation.ate_rmse`.
