from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_file = LaunchConfiguration('map')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation/Gazebo clock')

    declare_map_cmd = DeclareLaunchArgument(
        'map',
        description='Full path to the map yaml file used for localisation')

    # Publishes the given map on /map
    map_server_cmd = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_file, 'use_sim_time': use_sim_time}])

    # map_server is a lifecycle node and needs to be activated
    lifecycle_manager_cmd = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localisation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'autostart': True, 'node_names': ['map_server']}])

    particle_filter_cmd = Node(
        package='final_project',
        executable='particle_filter',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}])

    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_map_cmd,
        map_server_cmd,
        lifecycle_manager_cmd,
        particle_filter_cmd
    ])
