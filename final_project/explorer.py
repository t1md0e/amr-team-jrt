import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node

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
THRESHOLD_GOAL = 0.3          # distance at which a goal counts as reached
MIN_GOAL_DISTANCE = 1.0       # preferred minimal distance of a goal, closer fringe cells are only used if there are no others
PROGRESS_DISTANCE = 0.2       # robot has to get this much closer to the goal ...
PROGRESS_TIMEOUT = 30.0       # ... within this time (s), otherwise the goal is abandoned
INITIAL_STEP = 0.6            # distance the robot moves forward if its own cell is still unknown

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
        self.best_distance = None
        self.last_progress_time = None

        # Goals that could not be reached
        self.blacklist = []
        self.finished = False

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

    def find_goal(self):
        """ Find the closest reachable fringe cell using the wavefront algorithm (breadth-first search) """
        free = (self.occupancy >= 0) & (self.occupancy < FREE_THRESHOLD)
        fringe = self.get_fringe(free)
        candidates = fringe & self.get_configuration_space(free)
        candidates &= self.get_fringe_sizes(fringe) >= MIN_FRONTIER_SIZE

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

    def publish_goal(self):
        goal_x, goal_y = self.goal

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # Face in driving direction, towards the unexplored region
        goal_theta = math.atan2(goal_y - self.y, goal_x - self.x)
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

        if self.goal is not None:
            distance = euclid_distance(self.x, self.y, self.goal[0], self.goal[1])

            if distance < self.best_distance - PROGRESS_DISTANCE:
                self.best_distance = distance
                self.last_progress_time = now

            if distance < THRESHOLD_GOAL:
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
        if self.occupancy[cell_y, cell_x] < 0:
            self.publish_initial_step()
            return

        self.goal = self.find_goal()
        if self.goal is None:
            # Keep checking, as the map can still change
            if not self.finished:
                self.get_logger().info(f'\nNo reachable fringe left, exploration finished')
                self.finished = True
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

    rclpy.spin(explorer)

    explorer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
