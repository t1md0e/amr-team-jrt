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

This node is responsible for finding a path between the robot's current position and a given goal position. It uses A* search to find an optimal path between the robot position from `/odom` and the goal position from `/goal`. It can only find a path if it is also given an occupancy grid through `/map`. Occupied cells are grown by 0.35 m (half of the robot width plus a margin, configuration space), so that the path keeps a distance to obstacles while the robot still fits lengthwise through doors. If no path is found (e.g. in a passage narrower than 0.7 m, or if start or goal are close to an obstacle), the distance is reduced step by step down to none, so that the path stays in the middle of a narrow passage instead of running along one of its walls.

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

The navigation is a potential field in the configuration space q = (x, y, θ) of the robot (lecture chapter 5): U(q) = U_att(q) + U_rep(q), and the robot moves along F(q) = −∇U(q). As the Robile is omnidirectional, the force is used directly as velocity in x, y and θ, so there are no special cases for rotating in place, narrow passages or obstacles next to the robot:
- U_rep: every obstacle point (laser scan and remembered points, thinned out to one per 5 cm cell, so that a wall counts by its length and not by the number of laser points) has the repulsive potential ½·k·(1/ρ − 1/ρ₀)² for ρ < ρ₀ = 0.4 m, where ρ is the clearance between the point and the rectangular robot footprint at q. A rotation that brings a corner closer to a wall therefore increases U, and in a narrow passage the repulsions of both sides cancel each other in the middle.
- U_att: quadratic within 0.3 m of the waypoint and conic (constant force) beyond, so the robot slows down only close to the waypoint; for the orientation, an attractive potential towards the waypoint (the laser scanner looks in driving direction) and, once the position is reached, towards the orientation of the waypoint.
- The gradient is computed numerically (central differences). A step along −∇U is limited to the speed limits and only taken if U decreases (line search with shorter steps, then translation or rotation alone); otherwise the robot would overshoot and oscillate between the walls of a narrow passage. Moving backwards (out of the laser's field of view) is slowed down to half the speed.
- Local minima: if the robot gets less than 0.05 m closer to the waypoint within 5 s, it escapes with a random walk (lecture chapter 5): for 2 to 4 s, the attractive potential is replaced by one in a random direction. Within 0.3 m of the waypoint (e.g. a waypoint close to a wall), the position is taken as reached and the robot turns to the orientation of the waypoint.

The speed limits can be set with the parameters `max_linear_speed` (default 0.3 m/s) and `max_angular_speed` (default 0.8 rad/s), e.g. `ros2 run final_project potential_field_navigator --ros-args -p max_linear_speed:=0.5` or as launch arguments of `exploration.launch.py` (`min_linear_speed` is no longer used). The simulated robot hardly moves below 0.25 m/s and 0.5 rad/s, so the small steps of the potential field would not be executed; `linear_deadband` and `angular_deadband` (default 0) are added to every command that is not zero, in simulation use e.g. `-p max_linear_speed:=0.55 -p max_angular_speed:=1.0 -p linear_deadband:=0.25 -p angular_deadband:=0.5`. Laser measurements outside of the range of the scanner (e.g. 0 on the real robot) are ignored, and the laser frame is taken from the scan messages (`base_laser_front_link` in simulation, `base_laser` on the real robot).

The footprint is set with the parameters `robot_length` (default 0.76 m), `robot_width` (default 0.47 m) and `laser_to_front` (default 0.05 m, distance from the laser scanner to the front side, as the scanner is mounted in the middle of the front side); the position of the laser scanner in `base_link` is taken from the transforms. Laser points of the last 300 s within 2 m are remembered (odom frame, transformed with the pose at the time of the scan; if the odometry is published slightly later than the scan, the latest transform is used if it is at most 50 ms older), so that obstacles that have left the laser's field of view (next to or behind the robot, e.g. the walls of a passage) are still considered; a remembered point is forgotten once the laser scanner sees through it, i.e. all measurements within ±0.05 rad of its angle are at least 0.1 m farther away (e.g. a person who has walked on; a single beam is not enough, as it can pass just beside a corner).

Note that obstacles behind the robot are only known if the laser scanner has seen them before. In simulation, the robot model is longer (laser 0.45 m in front of `base_link`, rear side 0.35 m behind it), so use `robot_length:=0.85` there.

The waypoint is given in map frame and transformed to the odom frame in every control step, so that the goal follows corrections of the localisation (`map` -> `odom`).

If the waypoint is reached, the robot rotates to the desired orientation (as far as the obstacles allow) and stops.

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

This node is responsible for exploring the environment by selecting goals at the map fringe, i.e. free cells that are next to unknown cells. Fringe cells are grouped into connected regions, and only regions with at least 4 cells (0.2 m) are considered. To make sure that the robot fits there and can rotate, occupied cells are grown by 0.5 m (the corners of the robot are about 0.45 m away from its center, configuration space) and only cells that are still free are used as goals. Fringe cells that are closer to obstacles (e.g. next to a wall behind an obstacle) are explored from a view pose: the closest free cell of the configuration space within 1 m, connected to the fringe cell by free cells, with the goal orientation pointing towards the fringe cell. A small enclosed unknown region (at most 1 m²) at the start position of the robot is ignored, as it is the floor below the robot at the start, which the laser scanner at the front never sees.

The goal is the closest reachable fringe cell (or view pose) that is at least 1 m away from the robot, which is found using the wavefront algorithm (breadth-first search over the free cells, starting at the robot position). Closer fringe cells are only used if there is no other, as there is always fringe right next to the robot (the laser scanner only looks to the front). For the same reason, the goal orientation points towards the unknown cells around the goal, so that the robot looks into the unexplored region once it has reached the goal. The goal is published to `/goal`, so that `path_planner` and `potential_field_navigator` move the robot there.

A new goal is selected if the current goal is reached, if the region around the goal has already been explored while driving there, or if the robot makes no progress towards the goal. If the region around the goal has been explored on the way, the new goal is the fringe cell closest to the previous goal instead of the robot, so that the exploration continues in the driving direction (otherwise the robot would e.g. turn around inside a narrow passage as soon as it looks into the room behind it). In the last case, the goal is added to a blacklist and is not selected again; the same happens with a fringe cell that is still unknown after the robot has reached its view pose (e.g. hidden by an obstacle). Once no reachable fringe is left, the exploration is finished.

If the parameter `map_file` is set (e.g. `ros2 launch final_project exploration.launch.py map_file:=~/lab_map`), the map is saved in the format of `map_server` (`.pgm` and `.yaml`) at the end of the exploration, every 30 s and when the node is stopped, so that it can be used for the localisation of task 2.

As the laser scanner is mounted at the front of the robot, the robot's own cell is still unknown at the start, so that A* can not find a path. In this case, the node first moves the robot forward by publishing a waypoint directly to `potential_field_navigator`. This is only done at the start, until the robot's cell has been seen once.


### Script: `a_star.py`

This script contains the class `OccupancyGridAStar`, which carries out A* search on an occupancy grid. It considers cells with an occupancy below 50 as free and considers both direct and diagonal neighbors of cells as successors. The used heuristic is Euclidean distance.



## Disclaimer:  
Some code was written with the help of AI.
