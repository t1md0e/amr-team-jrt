from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation/Gazebo clock')

    # SLAM: creates the map and publishes map -> odom
    slam_gmapping_cmd = Node(
        package='slam_gmapping',
        executable='slam_gmapping',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}])

    # Task 3: selects goals at the map fringe
    explorer_cmd = Node(
        package='final_project',
        executable='explorer',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}])

    # Task 1: path to the goal (A*) and motion between the waypoints (potential field)
    path_planner_cmd = Node(
        package='final_project',
        executable='path_planner',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}])

    potential_field_navigator_cmd = Node(
        package='final_project',
        executable='potential_field_navigator',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}])

    return LaunchDescription([
        declare_use_sim_time_cmd,
        slam_gmapping_cmd,
        explorer_cmd,
        path_planner_cmd,
        potential_field_navigator_cmd
    ])
