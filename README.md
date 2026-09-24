# AMR Project

## Project Objectives

The objective of this project is that you deploy some of the functionalities that were discussed during the course on a real robot platform. In particular, we want to have functionalities for path and motion planning, localisation, and environment exploration on the robot.

We will particularly use the Robile platform during the project; you are already familiar with this robot from the simulation you have been using throughout the semester as well as from the few practical lab sessions that we have had.

## Task Description

The project consists of three parts that are building on each other: (i) path and motion planning, (ii) localisation, and (iii) environment exploration.

### 1. Path and Motion Planning

You have already implemented a *potential field planner* in one of your assignments. In this first part of the project, you need to port your implementation to the real robot and ensure that it is working as well as it was in the simulated environment so that you can navigate towards global goals while avoiding obstacles. Then, integrate your potential field planner with a global path planner, namely first use a path planner (e.g. A*) to find a rough global trajectory of waypoints that the robot can follow to reach a goal and then use the potential field planner to navigate between the waypoints. This will make your potential field planner applicable to large environments, where it can navigate given an environment map.

### 2. Localisation

In one of the course lectures, we discussed Monte Carlo localisation as a practical solution to the robot localisation problem in an existing map. In this second part of the project, your objective is to implement your very own particle filter that you then integrate on the Robile. You should implement the simple version of the filter that we discussed in the lecture; however, if you have time and interest, you are free to additionally explore extensions / improvements to the algorithm, for example in the form of the adaptive Monte Carlo approach that we mentioned in the lecture.

### 3. Environment Exploration

The final objective of the project is to incorporate an environment exploration functionality to the robot. This will have to be combined with a SLAM component, namely you will need your exploration component to select poses to explore and a SLAM component that will take care of actually creating a map. The exploration algorithm should ideally select poses at the map fringe (i.e. poses that are at the boundary between the explored and unexplored region), but you are free to explore different pose selection strategies in your implementation.

## Setup

- Clone the repository
- Move the folder to your ROS2 workspace under `/src`
- Run `colcon build` in the root of the workspace

## Usage

The package has the following nodes and scripts:

- Task 1:
  - `path_planner`
  - `potential_field_navigator`
- Task 2:
  - `particle_filter`
- Task 3:
  - `slam_gmapping`
  - `explorer`
- Additionally:
  - `a_star.py` (supports `path_planner` node)

The following launch files are available:

- Task 2: `ros2 launch final_project localisation.launch.py map:=/path/to/map.yaml` (starts `map_server` and `particle_filter`)

The launch file accepts `use_sim_time:=true` for running in simulation.

### Node: `path_planner`

Subscribes to:
- `/odom` -> `nav_msgs/Odometry`
- `/map` -> `nav_msgs/OccupancyGrid`
- `/goal` -> `geometry_msgs/PoseStamped` (expects position in map frame)

Publishes:
- `/waypoint` -> `geometry_msgs/PoseStamped` (position in map frame)

This node is responsible for finding a path between the robot's current position and a given goal position. It uses A* search to find an optimal path between the robot position from `/odom` and the goal position from `/goal`. It can only find a path if it is also given an occupancy grid through `/map`.

After a path has been calculated, it samples waypoints from the path and publishes them to `/waypoint`. Only ever one waypoint at a time is published: At first, this is simply the first waypoint in the path. The node keeps track of the robot's position and publishes the next waypoint in the path once the current waypoint has been reached. Once the final waypoint (i.e. the goal) has been reached, the path is discarded and no new waypoint is published.

If a new goal is published even though the goal has not been reached yet by the robot, the node will abandon the current path and search for a path to the new goal. The waypoints will start from the beginning again.

If a new map is published even though the goal has not been reached yet by the robot, it will be saved and used for coordinate calculations, but no new path will be generated. It is therefore important that any grids published to `/map` maintain the same origin and resolution.

### Node: `potential_field_navigator`

Subscribes to:
- `/odom` -> `nav_msgs/Odometry`
- `/scan` -> `sensor_msgs/LaserScan`
- `/waypoint` -> `geometry_msgs/PoseStamped` (position in map frame)

Publishes:
- `/cmd_vel` -> `geometry_msgs/Twist`

This node is responsible for navigating the robot towards a given waypoint while avoiding obstacles. It uses potential-field based navigation move around obstacles detected by its Lidar sensor and toward a goal given by `/waypoint`.

From the data of the Lidar sensor, the distances to obstacles are calculated and used to determine their repulsive forces. These forces are combined with the attractive forces calculated from the given goal position. The node then uses this data to calculate linear and angular velocities that are published to `/cmd_vel`.

If the waypoint is reached, the robot rotates to the desired position and stops.

### Node: `particle_filter`

Subscribes to:
- `/odom` -> `nav_msgs/Odometry`
- `/scan` -> `sensor_msgs/LaserScan`
- `/map` -> `nav_msgs/OccupancyGrid` (transient local, e.g. from `map_server`)
- `/initialpose` -> `geometry_msgs/PoseWithCovarianceStamped` (optional, e.g. from the 2D Pose Estimate widget in RViz)

Publishes:
- `/estimated_pose` -> `geometry_msgs/PoseWithCovarianceStamped` (position in map frame)
- `/particles` -> `geometry_msgs/PoseArray` (for visualisation in RViz)
- `/tf` -> transform `map` -> `odom`

This node is responsible for localising the robot in a given map using Monte Carlo localisation, i.e. a particle filter. When the map is received, the particles are sampled uniformly over the free cells of the map (global localisation). If a pose is published to `/initialpose`, the particles are instead sampled from a Gaussian distribution around that pose.

The filter is only updated once the robot has moved or rotated far enough. Every update consists of three steps:
- Motion update: Every particle is moved by sampling from the odometry motion model (initial rotation, translation, final rotation, each under the influence of Gaussian noise).
- Measurement update: For a subset of the laser beams, the expected range is calculated for every particle by casting a ray through the occupancy grid. The particle weight is the likelihood of the measured ranges, modelled as a Gaussian around the expected range mixed with a small probability for random measurements.
- Resampling: The particles are sampled with replacement proportional to their weights. To recover from localisation failures (kidnapped robot problem), random particles are added if the short term average of the measurement likelihood drops below the long term average.

The estimated pose is the weighted mean of the particles, its variance is published as covariance. The node also publishes the transform `map` -> `odom`, so that other nodes (e.g. `potential_field_navigator`) can transform between both frames.

### Node: `slam_gmapping`

Subscribes to:
- `/tf` -> `tf/tfMessage`
- `/scan` -> `sensor_msgs/LaserScan`

Publishes:
- `/map_metadata` -> `nav_msgs/MapMetaData`
- `/map` -> `nav_msgs/OccupancyGrid` (Get the map data from this topic, which is latched, and updated periodically)
- `/~entropy` -> `std_msgs/Float64` (Estimate of the entropy of the distribution over the robot's pose (a higher value indicates greater uncertainty))

TODO: description and source


### Script: `a_star.py`

This script contains the class `OccupancyGridAStar`, which carries out A* search on an occupancy grid. It considers cells with an occupancy below 50 as free and considers both direct and diagonal neighbors of cells as successors. The used heuristic is Euclidean distance.
