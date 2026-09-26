from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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

    # Optional exploration boundary (map frame) and stopping criteria (s), disabled by default
    explorer_params = ['boundary_x_min', 'boundary_x_max', 'boundary_y_min', 'boundary_y_max',
                       'max_duration', 'stagnation_timeout']
    explorer_defaults = ['-1.0e9', '1.0e9', '-1.0e9', '1.0e9', '0.0', '0.0']
    declare_explorer_cmds = [
        DeclareLaunchArgument(name, default_value=default, description='Explorer parameter, see README')
        for name, default in zip(explorer_params, explorer_defaults)]

    # Task 3: selects goals at the map fringe
    explorer_cmd = Node(
        package='final_project',
        executable='explorer',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {name: ParameterValue(LaunchConfiguration(name), value_type=float) for name in explorer_params}])

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

    return LaunchDescription(declare_explorer_cmds + [
        declare_use_sim_time_cmd,
        slam_gmapping_cmd,
        explorer_cmd,
        path_planner_cmd,
        potential_field_navigator_cmd
    ])
