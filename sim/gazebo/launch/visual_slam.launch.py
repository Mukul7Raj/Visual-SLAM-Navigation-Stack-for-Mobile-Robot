"""ROS 2 Humble launch: Gazebo indoor world + TurtleBot3 with a forward camera.

Install (Ubuntu):
  sudo apt install ros-humble-turtlebot3* ros-humble-gazebo-ros-pkgs

Run:
  export TURTLEBOT3_MODEL=waffle
  ros2 launch visual_slam_bringup visual_slam.launch.py
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    world = str(Path(__file__).resolve().parents[1] / "worlds" / "indoor_lab.world")
    tb3 = get_package_share_directory("turtlebot3_gazebo")
    return LaunchDescription(
        [
            SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle"),
            DeclareLaunchArgument("world", default_value=world),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(Path(tb3) / "launch" / "turtlebot3_world.launch.py")
                ),
                launch_arguments={"world": LaunchConfiguration("world")}.items(),
            ),
            Node(
                package="image_transport",
                executable="republish",
                name="camera_relay",
                arguments=["raw"],
                remappings=[("in", "/camera/image_raw"), ("out", "/vslam/image")],
            ),
        ]
    )
