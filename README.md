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
- Task 3: `ros2 launch final_project exploration.launch.py` (starts `slam_gmapping`, `explorer`, `path_planner` and `potential_field_navigator`)

Both launch files accept `use_sim_time:=true` for running in simulation.

### Node: `path_planner`

Subscribes to:
- `/odom` -> `nav_msgs/Odometry` (only used if the transform `map` -> `base_link` is not available)
- `/tf` -> transform `map` -> `base_link` (robot position in map frame, e.g. from `particle_filter` or `slam_gmapping`)
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

The speed limits can be set with the parameters `max_linear_speed` (default 0.3 m/s), `min_linear_speed` (default 0.1 m/s) and `max_angular_speed` (default 0.8 rad/s), e.g. `ros2 run final_project potential_field_navigator --ros-args -p max_linear_speed:=0.5` or as launch arguments of `exploration.launch.py`. Laser measurements outside of the range of the scanner (e.g. 0 on the real robot) are ignored, and the laser frame is taken from the scan messages (`base_laser_front_link` in simulation, `base_laser` on the real robot).

The waypoint is given in map frame and transformed to the odom frame in every control step, so that the goal follows corrections of the localisation (`map` -> `odom`).

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
- Measurement update: For a subset of the laser beams, the expected range is calculated for every particle by casting a ray through the occupancy grid. The particle weight is the likelihood of the measured ranges, modelled as a Gaussian around the expected range mixed with a small probability for random measurements. Beams whose ray ends in an unknown cell of the map give no information and get a uniform likelihood, so that the filter also works with partial maps.
- Resampling: The particles are sampled with replacement proportional to their weights. To recover from localisation failures (kidnapped robot problem), random particles are added if the short term average of the measurement likelihood drops below the long term average.

Rays are cast up to 10 m (the real laser scanner has a range of 60 m), and the laser frame is taken from the scan messages.

The estimated pose is the weighted mean of the particles, its variance is published as covariance. The node also publishes the transform `map` -> `odom`, so that other nodes (e.g. `potential_field_navigator`) can transform between both frames.

### Node: `slam_gmapping`

Subscribes to:
- `/tf` -> `tf/tfMessage`
- `/scan` -> `sensor_msgs/LaserScan`

Publishes:
- `/map_metadata` -> `nav_msgs/MapMetaData`
- `/map` -> `nav_msgs/OccupancyGrid` (Get the map data from this topic, which is latched, and updated periodically)
- `/~entropy` -> `std_msgs/Float64` (Estimate of the entropy of the distribution over the robot's pose (a higher value indicates greater uncertainty))

This node is responsible for creating a map of the environment while localising the robot in it (SLAM). It implements grid-based FastSLAM, i.e. a Rao-Blackwellised particle filter where every particle maintains its own occupancy grid map. It also publishes the transform `map` -> `odom`.

Source: ROS2 port of gmapping (https://github.com/Project-MANAS/slam_gmapping, branch `eloquent-devel`), which needs to be cloned into the `/src` folder of the workspace next to this package, as there is no binary package for ROS2 Humble.

### Node: `explorer`

Subscribes to:
- `/map` -> `nav_msgs/OccupancyGrid`
- `/tf` -> transform `map` -> `base_link` (from `slam_gmapping`)

Publishes:
- `/goal` -> `geometry_msgs/PoseStamped` (position in map frame, used by `path_planner`)
- `/waypoint` -> `geometry_msgs/PoseStamped` (only for the initial step, see below)

This node is responsible for exploring the environment by selecting goals at the map fringe, i.e. free cells that are next to unknown cells. Fringe cells are grouped into connected regions, and only regions with a minimum size are considered. To make sure that the robot fits there, occupied cells are grown by the robot radius (configuration space) and only fringe cells that are still free are used as goals.

The goal is the closest reachable fringe cell that is at least 1 m away from the robot, which is found using the wavefront algorithm (breadth-first search over the free cells, starting at the robot position). Closer fringe cells are only used if there is no other, as there is always fringe right next to the robot (the laser scanner only looks to the front). For the same reason, the goal orientation points towards the unknown cells around the goal, so that the robot looks into the unexplored region once it has reached the goal. The goal is published to `/goal`, so that `path_planner` and `potential_field_navigator` move the robot there.

A new goal is selected if the current goal is reached, if the region around the goal has already been explored while driving there, or if the robot makes no progress towards the goal. In the last case, the goal is added to a blacklist and is not selected again. Once no reachable fringe is left, the exploration is finished.

As the laser scanner is mounted at the front of the robot, the robot's own cell is still unknown at the start, so that A* can not find a path. In this case, the node first moves the robot forward by publishing a waypoint directly to `potential_field_navigator`. This is only done at the start, until the robot's cell has been seen once.


### Script: `a_star.py`

This script contains the class `OccupancyGridAStar`, which carries out A* search on an occupancy grid. It considers cells with an occupancy below 50 as free and considers both direct and diagonal neighbors of cells as successors. The used heuristic is Euclidean distance.
