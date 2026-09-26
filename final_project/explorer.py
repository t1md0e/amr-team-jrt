import math
import os
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler
import tf2_ros

FREE_THRESHOLD = 50           # same as in a_star.py: cells with an occupancy below 50 are free
ROBOT_RADIUS = 0.4            # obstacles are grown by this radius (configuration space)
MIN_FRONTIER_SIZE = 15        # minimal number of connected fringe cells (about robot width) to be considered as a goal
BLACKLIST_RADIUS = 0.5        # fringe cells around a failed goal are ignored
THRESHOLD_GOAL = 0.3          # distance at which a goal counts as reached ...
THRESHOLD_GOAL_ROTATION = 0.3 # ... together with this orientation error (robot has to look into the unknown region)
MIN_GOAL_DISTANCE = 1.0       # preferred minimal distance of a goal, closer fringe cells are only used if there are no others
PROGRESS_DISTANCE = 0.2       # robot has to get this much closer to the goal ...
PROGRESS_TIMEOUT = 30.0       # ... within this time (s), otherwise the goal is abandoned
UNKNOWN_DIRECTION_RADIUS = 1.0  # unknown cells within this radius around a goal determine the goal orientation
MAP_SAVE_INTERVAL = 30.0      # s, the map is also saved periodically, so that it is not lost if the node is stopped
MIN_AREA_GAIN = 1.0           # explored area (m^2) that counts as progress for the stagnation criterion
INITIAL_STEP = 0.6            # distance the robot moves forward if its own cell is still unknown
MAX_INITIAL_STEPS = 5         # safety limit, e.g. if SLAM does not update the map

# 8-connected neighbors
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

class Explorer(Node):

    def __init__(self):
        super().__init__('explorer')

        self.goal_pub = self.create_publisher(PoseStamped, '/goal', 10)
        self.waypoint_pub = self.create_publisher(PoseStamped, '/waypoint', 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.update_map, 10)

        # Get listener for robot pose in map frame (published by SLAM)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Current pose (map frame) to be updated
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Last received OccupancyGrid msg
        self.grid = None
        self.occupancy = None

        # Currently explored goal (map frame) and progress towards it
        self.goal = None
        self.goal_theta = None
        self.best_distance = None
        self.last_progress_time = None

        # Goals that could not be reached
        self.blacklist = []
        self.finished = False

        # Optional exploration boundary (map frame), only fringe cells inside are used as goals,
        # e.g. to keep the robot inside a building with an open door
        self.boundary_x_min = self.declare_parameter('boundary_x_min', -math.inf).value
        self.boundary_x_max = self.declare_parameter('boundary_x_max', math.inf).value
        self.boundary_y_min = self.declare_parameter('boundary_y_min', -math.inf).value
        self.boundary_y_max = self.declare_parameter('boundary_y_max', math.inf).value

        # Optional stopping criteria in s (0 = disabled): total exploration time, and time without new explored area
        self.max_duration = self.declare_parameter('max_duration', 0.0).value
        self.stagnation_timeout = self.declare_parameter('stagnation_timeout', 0.0).value
        self.start_time = None
        self.known_area = 0.0
        self.last_area_gain_time = None
        self.stopped = False

        # The initial step is only needed at the start, before the robot's own cell has been seen once
        self.initial_step_done = False
        self.initial_steps = 0

        # Optional file name (without extension) to save the map to, in the format of map_server (.pgm and .yaml),
        # e.g. for the localisation of task 2 (empty = disabled)
        self.map_file = os.path.expanduser(self.declare_parameter('map_file', '').value)
        if self.map_file:
            self.map_save_timer = self.create_timer(MAP_SAVE_INTERVAL, self.save_map)

        self.timer = self.create_timer(1.0, self.control_loop)

    def update_map(self, msg):
        self.grid = msg
        self.occupancy = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)

    def get_fringe(self, free):
        """ Get free cells that are next to unknown cells, i.e. the boundary between explored and unexplored region """
        unknown = self.occupancy < 0
        unknown_neighbor = np.zeros_like(unknown)
        unknown_neighbor[1:, :] |= unknown[:-1, :]
        unknown_neighbor[:-1, :] |= unknown[1:, :]
        unknown_neighbor[:, 1:] |= unknown[:, :-1]
        unknown_neighbor[:, :-1] |= unknown[:, 1:]
        return free & unknown_neighbor

    def get_configuration_space(self, free):
        """ Grow occupied cells by the robot radius, so that only cells the robot fits into remain free """
        occupied = self.occupancy >= FREE_THRESHOLD
        radius = int(math.ceil(ROBOT_RADIUS / self.grid.info.resolution))
        height, width = occupied.shape
        grown = np.zeros_like(occupied)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx ** 2 + dy ** 2 > radius ** 2:
                    continue
                # Shift occupied cells by (dx, dy)
                grown[max(0, dy):height + min(0, dy), max(0, dx):width + min(0, dx)] |= \
                    occupied[max(0, -dy):height + min(0, -dy), max(0, -dx):width + min(0, -dx)]

        return free & ~grown

    def get_fringe_sizes(self, fringe):
        """ Group fringe cells into connected regions and return the region size for every fringe cell """
        sizes = np.zeros(fringe.shape, dtype=int)
        visited = np.zeros_like(fringe)
        height, width = fringe.shape

        for start_y, start_x in zip(*np.nonzero(fringe)):
            if visited[start_y, start_x]:
                continue
            # Breadth-first search over connected fringe cells
            region = [(start_x, start_y)]
            visited[start_y, start_x] = True
            queue = deque(region)
            while queue:
                x, y = queue.popleft()
                for dx, dy in NEIGHBORS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and fringe[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        region.append((nx, ny))
                        queue.append((nx, ny))
            for x, y in region:
                sizes[y, x] = len(region)

        return sizes

    def get_boundary_mask(self):
        """ Get cells whose center is inside the exploration boundary """
        map_origin = self.grid.info.origin.position
        map_res = self.grid.info.resolution
        height, width = self.occupancy.shape

        map_x = (np.arange(width) + 0.5) * map_res + map_origin.x
        map_y = (np.arange(height) + 0.5) * map_res + map_origin.y
        inside_x = (map_x >= self.boundary_x_min) & (map_x <= self.boundary_x_max)
        inside_y = (map_y >= self.boundary_y_min) & (map_y <= self.boundary_y_max)

        return inside_y[:, None] & inside_x[None, :]

    def find_goal(self):
        """ Find the closest reachable fringe cell using the wavefront algorithm (breadth-first search) """
        free = (self.occupancy >= 0) & (self.occupancy < FREE_THRESHOLD)
        fringe = self.get_fringe(free)
        candidates = fringe & self.get_configuration_space(free)
        candidates &= self.get_fringe_sizes(fringe) >= MIN_FRONTIER_SIZE
        candidates &= self.get_boundary_mask()

        height, width = self.occupancy.shape
        start_x, start_y = self.map_to_cell_coords(self.x, self.y)

        # Wavefront starts at all free cells covered by the robot, as the cell of the robot itself
        # can still be unknown (the laser scanner is mounted at the front of the robot)
        radius = int(math.ceil(ROBOT_RADIUS / self.grid.info.resolution))
        visited = np.zeros_like(free)
        queue = deque()
        for y in range(max(0, start_y - radius), min(height, start_y + radius + 1)):
            for x in range(max(0, start_x - radius), min(width, start_x + radius + 1)):
                if free[y, x] and (x - start_x) ** 2 + (y - start_y) ** 2 <= radius ** 2:
                    visited[y, x] = True
                    queue.append((x, y))

        # Closest fringe cell that is not far enough away, only used if there is no other
        # (the laser scanner only looks to the front, so there is always fringe right next to the robot)
        close_goal = None

        # Wavefront expands over free cells
        while queue:
            x, y = queue.popleft()
            if candidates[y, x] and not self.is_blacklisted(x, y):
                map_x, map_y = self.cell_to_map_coords(x, y)
                distance = euclid_distance(self.x, self.y, map_x, map_y)
                if distance >= MIN_GOAL_DISTANCE:
                    return map_x, map_y
                # Goals closer than the goal threshold would immediately count as reached
                if close_goal is None and distance >= THRESHOLD_GOAL:
                    close_goal = map_x, map_y
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and free[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

        # No reachable fringe far enough away, use a close one (None if there is no reachable fringe at all)
        return close_goal

    def is_blacklisted(self, cell_x, cell_y):
        map_x, map_y = self.cell_to_map_coords(cell_x, cell_y)
        return any(euclid_distance(map_x, map_y, b_x, b_y) < BLACKLIST_RADIUS for b_x, b_y in self.blacklist)

    def is_fringe(self, map_x, map_y):
        """ Check whether there are still unknown cells around a goal position """
        cell_x, cell_y = self.map_to_cell_coords(map_x, map_y)
        height, width = self.occupancy.shape
        window = self.occupancy[max(0, cell_y - 1):min(height, cell_y + 2), max(0, cell_x - 1):min(width, cell_x + 2)]
        return bool(np.any(window < 0))

    def cell_to_map_coords(self, cell_x, cell_y):
        """ Convert occupancy grid cell coordinates to map coordinates """
        map_origin = self.grid.info.origin.position
        map_res = self.grid.info.resolution

        map_x = (cell_x + 0.5) * map_res + map_origin.x
        map_y = (cell_y + 0.5) * map_res + map_origin.y

        return map_x, map_y

    def map_to_cell_coords(self, map_x, map_y):
        """ Convert map coordinates to occupancy grid cell coordinates """
        map_origin = self.grid.info.origin.position
        map_res = self.grid.info.resolution
        return int((map_x - map_origin.x) / map_res), int((map_y - map_origin.y) / map_res)

    def get_unknown_direction(self, map_x, map_y):
        """ Get the direction from a goal towards the unknown cells around it, so that the robot looks into the
        unexplored region once it has reached the goal (the laser scanner only looks to the front) """
        cell_x, cell_y = self.map_to_cell_coords(map_x, map_y)
        radius = int(math.ceil(UNKNOWN_DIRECTION_RADIUS / self.grid.info.resolution))
        height, width = self.occupancy.shape
        y_min, y_max = max(0, cell_y - radius), min(height, cell_y + radius + 1)
        x_min, x_max = max(0, cell_x - radius), min(width, cell_x + radius + 1)

        unknown_y, unknown_x = np.nonzero(self.occupancy[y_min:y_max, x_min:x_max] < 0)
        if len(unknown_x) == 0:
            # Fall back to the driving direction
            return math.atan2(map_y - self.y, map_x - self.x)

        # Direction towards the center of the unknown cells
        return math.atan2(np.mean(unknown_y) + y_min - cell_y, np.mean(unknown_x) + x_min - cell_x)

    def publish_goal(self, goal_theta=None):
        goal_x, goal_y = self.goal

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        if goal_theta is None:
            goal_theta = self.get_unknown_direction(goal_x, goal_y)
        self.goal_theta = goal_theta
        q = quaternion_from_euler(0.0, 0.0, goal_theta)
        msg.pose.position = Point(x=goal_x, y=goal_y, z=0.0)
        msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        self.goal_pub.publish(msg)

        self.get_logger().info(f'\nNew exploration goal: ({goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f})')

    def publish_initial_step(self):
        """ Move the robot forward so that its own cell becomes known, as A* needs a free start cell """
        step_x = self.x + INITIAL_STEP * math.cos(self.theta)
        step_y = self.y + INITIAL_STEP * math.sin(self.theta)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        q = quaternion_from_euler(0.0, 0.0, self.theta)
        msg.pose.position = Point(x=step_x, y=step_y, z=0.0)
        msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Sent directly to the potential field navigator, no path is needed for this short step
        self.waypoint_pub.publish(msg)

        self.get_logger().info(f'\nRobot cell unknown, moving forward to ({step_x:.2f}, {step_y:.2f})')

    def save_map(self):
        """ Save the last received map in the format of map_server (trinary .pgm image and .yaml metadata) """
        if not self.map_file or self.grid is None:
            return

        # Free cells are white, occupied cells are black and unknown cells are grey (as written by map_saver);
        # the image starts with the top row, the occupancy grid with the bottom row
        image = np.where(self.occupancy < 0, 205, np.where(self.occupancy >= FREE_THRESHOLD, 0, 254))
        image = image.astype(np.uint8)[::-1]
        height, width = image.shape
        with open(self.map_file + '.pgm', 'wb') as f:
            f.write(b'P5\n%d %d\n255\n' % (width, height))
            f.write(image.tobytes())

        map_origin = self.grid.info.origin.position
        with open(self.map_file + '.yaml', 'w') as f:
            f.write(f'image: {os.path.basename(self.map_file)}.pgm\n'
                    f'mode: trinary\n'
                    f'resolution: {self.grid.info.resolution}\n'
                    f'origin: [{map_origin.x}, {map_origin.y}, 0]\n'
                    f'negate: 0\n'
                    f'occupied_thresh: 0.65\n'
                    f'free_thresh: 0.25\n')

        self.get_logger().info(f'\nMap saved to {self.map_file}.pgm/.yaml')

    def stop_exploration(self, reason):
        """ Stop the exploration and let the robot stop at its current position """
        self.get_logger().info(f'\n{reason}, exploration stopped')
        self.stopped = True
        self.goal = (self.x, self.y)
        self.publish_goal(self.theta)
        self.save_map()

    def check_stopping_criteria(self, now):
        """ Check the optional stopping criteria, returns True if the exploration has to be stopped """
        if self.start_time is None:
            self.start_time = now
            self.last_area_gain_time = now

        known_area = np.sum(self.occupancy >= 0) * self.grid.info.resolution ** 2
        if known_area > self.known_area + MIN_AREA_GAIN:
            self.known_area = known_area
            self.last_area_gain_time = now

        if self.max_duration > 0 and now - self.start_time > self.max_duration:
            self.stop_exploration(f'Maximum exploration time of {self.max_duration:.0f} s reached')
            return True
        if self.stagnation_timeout > 0 and now - self.last_area_gain_time > self.stagnation_timeout:
            self.stop_exploration(f'No new area explored for {self.stagnation_timeout:.0f} s')
            return True
        return False

    def control_loop(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )
        except tf2_ros.TransformException:
            return

        self.x = transform.transform.translation.x
        self.y = transform.transform.translation.y
        quaternion = transform.transform.rotation
        self.theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]

        if self.grid is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.stopped or self.check_stopping_criteria(now):
            return

        if self.goal is not None:
            distance = euclid_distance(self.x, self.y, self.goal[0], self.goal[1])

            if distance < self.best_distance - PROGRESS_DISTANCE:
                self.best_distance = distance
                self.last_progress_time = now

            delta_theta = math.atan2(math.sin(self.goal_theta - self.theta), math.cos(self.goal_theta - self.theta))
            if distance < THRESHOLD_GOAL and abs(delta_theta) < THRESHOLD_GOAL_ROTATION:
                self.get_logger().info(f'\nExploration goal reached')
                self.goal = None
            elif not self.is_fringe(self.goal[0], self.goal[1]):
                # Region around goal has already been explored while driving there
                self.get_logger().info(f'\nExploration goal is no longer at the fringe')
                self.goal = None
            elif now - self.last_progress_time > PROGRESS_TIMEOUT:
                self.get_logger().warning(f'\nNo progress towards exploration goal, goal is blacklisted')
                self.blacklist.append(self.goal)
                self.goal = None
            else:
                # Keep following the current goal
                return

        cell_x, cell_y = self.map_to_cell_coords(self.x, self.y)
        height, width = self.occupancy.shape
        if not (0 <= cell_x < width and 0 <= cell_y < height):
            # Robot is outside of the map (e.g. before the map has been extended), this does not mean that
            # the exploration is finished
            self.get_logger().warning(f'\nRobot is outside of the map, waiting for the map to be extended')
            return

        # The laser scanner is mounted at the front, so the robot's own cell is unknown at the start
        # (later on, unknown cells below the robot are ignored, otherwise the robot would keep stepping
        # forward whenever it drives over a region that the laser has not seen)
        if self.occupancy[cell_y, cell_x] < 0 and not self.initial_step_done:
            if self.initial_steps < MAX_INITIAL_STEPS:
                self.initial_steps += 1
                self.publish_initial_step()
            else:
                self.get_logger().warning(f'\nRobot cell is still unknown after {MAX_INITIAL_STEPS} steps, '
                                          f'is the map being updated?')
            return
        self.initial_step_done = True

        self.goal = self.find_goal()
        if self.goal is None:
            # Keep checking, as the map can still change
            if not self.finished:
                self.get_logger().info(f'\nNo reachable fringe left, exploration finished')
                self.finished = True
                self.save_map()
            return
        self.finished = False

        self.best_distance = euclid_distance(self.x, self.y, self.goal[0], self.goal[1])
        self.last_progress_time = now
        self.publish_goal()


def euclid_distance(a_x, a_y, b_x, b_y):
    return math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2)

def main(args=None):
    rclpy.init(args=args)

    explorer = Explorer()

    try:
        rclpy.spin(explorer)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Save the final map also if the node is stopped (e.g. with Ctrl+C)
        explorer.save_map()

    explorer.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
