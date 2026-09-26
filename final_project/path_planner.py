import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler
import tf2_ros
from tf2_ros import TransformException

from final_project.a_star import OccupancyGridAStar

THRESHOLD_WAYPOINT = 0.1               # final waypoint (goal) needs to be reached exactly
THRESHOLD_INTERMEDIATE_WAYPOINT = 0.3  # intermediate waypoints only need to be passed
ROBOT_RADIUS = 0.4                     # obstacles are grown by this radius for path finding (configuration space)
START_SEARCH_RADIUS = 0.5              # if the robot's cell is not free, A* starts at a free cell within this radius
ZERO_REPLACEMENT = 1e-6

class PathPlanner(Node):

    def __init__(self):
        super().__init__('path_planner')

        self.waypoint_pub = self.create_publisher(PoseStamped, '/waypoint', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.update_pose, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.update_map, 10)
        # map_server publishes the map only once (transient local), SLAM publishes it periodically (volatile),
        # a transient local subscription is not compatible with volatile publishers, so both are needed
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.latched_map_sub = self.create_subscription(OccupancyGrid, '/map', self.update_map, map_qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal', self.update_goal, 10)

        # Current pose (odom frame) to be updated
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Goal pose (world frame)
        self.goal_x = None
        self.goal_y = None
        self.goal_theta = None

        # Last received OccupancyGrid msg
        self.grid = None

        # Current path and currently followed waypoint index
        self.path = []
        self.current_waypoint = None

        self.timer = self.create_timer(0.1, self.control_loop)

    def update_pose(self, msg):
        """ Get current position from /odom topic and update attractive velocity towards goal """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        quaternion = msg.pose.pose.orientation
        self.theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]

    def update_goal(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        quaternion = msg.pose.orientation
        self.goal_theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]
        if self.grid is not None:
            self.find_path()
        else:
            self.get_logger().warning(f'Goal received, but no occupancy grid available')

    def update_map(self, msg):
        self.grid = msg
        # Trigger path finding if a goal is known and no path is currently followed
        if self.goal_x is not None and self.goal_y is not None and self.goal_theta is not None and not self.path:
            self.find_path()

    def find_path(self):
        """ Find path to goal using A* algorithm """
        start = self.get_free_start_cell(*self.map_to_cell_coords(self.x, self.y))
        goal = self.map_to_cell_coords(self.goal_x, self.goal_y)
        inflation_cells = int(math.ceil(ROBOT_RADIUS / self.grid.info.resolution))
        astar = OccupancyGridAStar(self.grid, start, goal, inflation_cells)
        path = astar.search()
        if not path:
            # Start or goal can be too close to an obstacle, try again without keeping a distance
            path = OccupancyGridAStar(self.grid, start, goal).search()
        self.path = self.get_sampled_path_in_map_coords(path)
        self.current_waypoint = None
        if self.path:
            self.get_logger().info(f"Path found with {len(self.path)} waypoints")
        else:
            self.get_logger().warning(f'No path found to goal')

    def get_free_start_cell(self, cell_x, cell_y):
        """ Get the closest free cell to the robot's cell, as the robot's own cell can be unknown
        (e.g. the laser scanner only looks to the front) or be marked as occupied due to noise """
        radius = int(math.ceil(START_SEARCH_RADIUS / self.grid.info.resolution))
        width, height = self.grid.info.width, self.grid.info.height
        best_cell, best_dist = (cell_x, cell_y), float('inf')

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = cell_x + dx, cell_y + dy
                if not (0 <= x < width and 0 <= y < height):
                    continue
                val = self.grid.data[y * width + x]
                dist = dx ** 2 + dy ** 2
                if 0 <= val < 50 and dist < best_dist:
                    best_cell, best_dist = (x, y), dist

        return best_cell

    def get_sampled_path_in_map_coords(self, path, waypoint_spacing=0.3):
        """ Sample A* cell path into map-frame waypoints """
        if not path:
            return []

        waypoints = []

        # Always include the first path point
        last_waypoint = self.cell_to_map_coords(path[0][0], path[0][1])
        waypoints.append(last_waypoint)

        for cell_x, cell_y in path[1:]:
            map_x, map_y = self.cell_to_map_coords(cell_x, cell_y)

            dist = math.sqrt(
                (map_x - last_waypoint[0]) ** 2 +
                (map_y - last_waypoint[1]) ** 2
            )

            if dist >= waypoint_spacing:
                waypoint = (map_x, map_y)
                waypoints.append(waypoint)
                last_waypoint = waypoint

        # Always include the final goal cell
        final_waypoint = self.cell_to_map_coords(path[-1][0], path[-1][1])
        if waypoints[-1] != final_waypoint:
            waypoints.append(final_waypoint)

        return waypoints

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
        return int(round((map_x - map_origin.x) / map_res)), int(round((map_y - map_origin.y) / map_res))

    def publish_current_waypoint(self):
        point_x, point_y = self.path[self.current_waypoint]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose.position = Point(x=point_x, y=point_y, z=0.0)
        if self.current_waypoint == len(self.path) - 1:
            q = quaternion_from_euler(0.0, 0.0, self.goal_theta)
            msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        else:
            msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        self.waypoint_pub.publish(msg)

        self.get_logger().info(f'\nCurrent waypoint: {self.current_waypoint} ({self.path[self.current_waypoint]})')

    def control_loop(self):
        if self.current_waypoint is not None:
            point_x, point_y = self.path[self.current_waypoint]
            delta_x = point_x - self.x
            delta_y = point_y - self.y

            # Check whether current position is within radial threshold around waypoint position
            if self.current_waypoint < len(self.path) - 1:
                threshold = THRESHOLD_INTERMEDIATE_WAYPOINT
            else:
                threshold = THRESHOLD_WAYPOINT
            pos_reached = (delta_x ** 2 + delta_y ** 2) < threshold ** 2

            if pos_reached:
                if self.current_waypoint < len(self.path) - 1: # Current waypoint is not goal
                    self.current_waypoint += 1
                    self.publish_current_waypoint()
                else: # Current waypoint is goal
                    self.path = []
                    self.current_waypoint = None
                    self.goal_x = None
                    self.goal_y = None
                    self.goal_theta = None

        elif self.path: # There is a goal path, but no waypoint has been published yet
            self.current_waypoint = 0
            self.publish_current_waypoint()


def main(args=None):
    rclpy.init(args=args)

    planner = PathPlanner()

    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
