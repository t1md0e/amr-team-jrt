import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

from tf_transformations import euler_from_quaternion
import tf2_ros
from tf2_ros import TransformException

ATTRACTION_C = 1.0
REPULSION_C = 0.5  # tuned for the repulsion of the closest obstacle: attraction and repulsion are equal at about 0.5 m
RHO_0 = 0.8

THRESHOLD_ROTATION = 0.1
THRESHOLD_POSE = 0.1

ZERO_REPLACEMENT = 1e-6

class PotentialFieldNavigator(Node):

    def __init__(self):
        super().__init__('potential_field_navigator')

        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.update_pose, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.update_obstacles, 10)
        self.waypoint_sub = self.create_subscription(PoseStamped, '/waypoint', self.update_goal, 10)

        # Get listener for static transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.laser_base_transform = None

        # Frame of the laser scanner, taken from the scan messages
        # (base_laser_front_link in simulation, base_laser on the real robot)
        self.laser_frame = None

        # Speed limits, slow defaults for the real robot (the simulation was tested with 0.8 / 0.5 / 1.5)
        self.max_linear_speed = self.declare_parameter('max_linear_speed', 0.3).value
        self.min_linear_speed = self.declare_parameter('min_linear_speed', 0.1).value
        self.max_angular_speed = self.declare_parameter('max_angular_speed', 0.8).value
        self.odom_base_transform = None
        self.map_odom_transform = None

        # Current pose (odom frame) to be updated
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Goal pose (world frame)
        self.goal_x = self.x
        self.goal_y = self.y
        self.goal_theta = self.theta

        # Robot does not move before the first waypoint has been received
        self.has_goal = False

        # Last received waypoint (map frame), transformed to the odom frame in every control step, so that the
        # goal follows corrections of the localisation (map -> odom)
        self.waypoint = None

        # Current attractive and repulsive velocities (base link frame)
        self.attraction = 0.0, 0.0
        self.repulsion = 0.0, 0.0

        self.timer = self.create_timer(0.1, self.control_loop)

    def update_pose(self, msg):
        """ Get current position from /odom topic and update attractive velocity towards goal """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        quaternion = msg.pose.pose.orientation
        self.theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]

        if self.odom_base_transform:
            vel_x, vel_y = self.get_attraction(self.x, self.y)
            self.attraction = tuple(apply_rotation(np.array([[vel_x, vel_y]]), self.odom_base_transform)[0])

    def update_goal(self, msg):
        goal_quat = msg.pose.orientation
        goal_yaw = euler_from_quaternion([goal_quat.x, goal_quat.y, goal_quat.z, goal_quat.w])[2]
        self.waypoint = (msg.pose.position.x, msg.pose.position.y, goal_yaw)
        self.has_goal = True

    def transform_waypoint(self):
        """ Transform the waypoint from map frame to odom frame with the current map_odom transform """
        transform = self.map_odom_transform
        waypoint_x, waypoint_y, waypoint_yaw = self.waypoint
        transformed_coords = apply_transform(np.array([[waypoint_x, waypoint_y]]), transform)
        self.goal_x = transformed_coords[0][0]
        self.goal_y = transformed_coords[0][1]
        transform_quat = transform.transform.rotation
        transform_yaw = euler_from_quaternion([transform_quat.x, transform_quat.y, transform_quat.z, transform_quat.w])[2]
        self.goal_theta = math.atan2(math.sin(waypoint_yaw + transform_yaw), math.cos(waypoint_yaw + transform_yaw))

    def update_obstacles(self, msg):
        """ Get obstacle distances from /scan topic and update repulsive velocity from obstacles """
        self.laser_frame = msg.header.frame_id
        if not self.laser_base_transform:
            return

        # Invalid measurements (e.g. 0 on the real laser scanner) would create a huge repulsion
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        coords_polar = np.column_stack((ranges[valid], angles[valid]))
        coords_clean = np_polar2cart(coords_polar)
        coords_transformed = apply_transform(coords_clean, self.laser_base_transform)

        if len(coords_transformed) == 0:
            self.repulsion = 0.0, 0.0
            return

        # Repulsive field depends on the minimum distance to an obstacle (closest scan point), summing over
        # all scan points would count a wall many times and create local minima in front of every wall
        distances = np.hypot(coords_transformed[:, 0], coords_transformed[:, 1])
        closest = coords_transformed[np.argmin(distances)]
        self.repulsion = self.get_repulsion(0.0, 0.0, closest[0], closest[1])

    def get_attraction(self, x, y):
        """ Calculate attractive velocity for a given point in world frame with respect to the goal position """
        delta_x = x - self.goal_x
        delta_y = y - self.goal_y
        distance = max(math.sqrt(delta_x ** 2 + delta_y ** 2), ZERO_REPLACEMENT)
        return (- ATTRACTION_C * delta_x / distance), (- ATTRACTION_C * delta_y / distance)

    def get_repulsion(self, x, y, obstacle_x, obstacle_y):
        """ Calculate repulsive velocity for a given point in world frame with respect to an obstacle position """
        delta_x = x - obstacle_x
        delta_y = y - obstacle_y
        distance = max(math.sqrt(delta_x ** 2 + delta_y ** 2), ZERO_REPLACEMENT)

        if distance < RHO_0:
            direction_x = delta_x / distance
            direction_y = delta_y / distance
            force_factor = REPULSION_C * (1.0 / distance - 1.0 / RHO_0) * (1 / distance ** 2)
            return force_factor * direction_x, force_factor * direction_y

        return 0.0, 0.0

    def control_loop(self):
        if self.laser_frame is None:
            # No scan received yet
            return
        try:
            self.laser_base_transform = self.tf_buffer.lookup_transform(
                "base_link",
                self.laser_frame,
                rclpy.time.Time()
            )
            self.odom_base_transform = self.tf_buffer.lookup_transform(
                "base_link",
                "odom",
                rclpy.time.Time()
            )
            self.map_odom_transform = self.tf_buffer.lookup_transform(
                "odom",
                "map",
                rclpy.time.Time()
            )
        except tf2_ros.TransformException:
            return

        msg = Twist()

        if not self.has_goal:
            # Wait for the first waypoint
            self.vel_pub.publish(msg)
            return

        self.transform_waypoint()

        delta_x = self.goal_x - self.x
        delta_y = self.goal_y - self.y
        delta_theta = self.goal_theta - self.theta

        # Normalize angles
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))

        # Check whether current position is within radial threshold around goal position
        pos_reached = (delta_x ** 2 + delta_y ** 2) < THRESHOLD_POSE ** 2

        if not pos_reached:
            # Get total velocities (base_link frame)
            vel_x = self.attraction[0] + self.repulsion[0]
            vel_y = self.attraction[1] + self.repulsion[1]

            # Heading towards combined vector
            desired_theta = math.atan2(vel_y, vel_x)

            # Angular control (unchanged idea)
            k_omega = 1.2
            MAX_ANG_VEL = self.max_angular_speed
            omega = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, k_omega * desired_theta))
            msg.angular.z = omega

            # Near-constant forward speed with smooth slowdowns
            V_REF = self.max_linear_speed  # nominal forward speed (parameter max_linear_speed)
            TURN_SLOWDOWN_K = 0.8       # how much to slow when turning (tune)
            OBS_SLOWDOWN_K = 2.0        # how much to slow when repulsion is strong (tune)

            # 1) Slow down when turning sharply
            turn_slow = 1.0 / (1.0 + TURN_SLOWDOWN_K * abs(omega) / MAX_ANG_VEL)

            # 2) Slow down when repulsion is large (i.e., close to obstacles)
            rep_mag = math.hypot(self.repulsion[0], self.repulsion[1])
            obs_slow = 1.0 / (1.0 + OBS_SLOWDOWN_K * rep_mag)

            # 3) Slow down close to the waypoint, otherwise the robot circles around it
            GOAL_SLOWDOWN_DIST = 0.3    # distance at which slowing down starts (tune)
            goal_slow = min(1.0, math.hypot(delta_x, delta_y) / GOAL_SLOWDOWN_DIST)

            # Final forward speed
            v_cmd = V_REF * turn_slow * obs_slow
            msg.linear.x = float(max(self.min_linear_speed, min(self.max_linear_speed, v_cmd)) * goal_slow)

            # Rotate in place if the desired direction is to the side or behind, driving would lead to circles
            ROTATE_IN_PLACE_ANGLE = math.pi / 2
            if abs(desired_theta) > ROTATE_IN_PLACE_ANGLE:
                msg.linear.x = 0.0

            self.get_logger().info(f'\nforces: ({vel_x:.2f}, {vel_y:.2f})')

        elif abs(delta_theta) > THRESHOLD_ROTATION:
            msg.angular.z = math.copysign(min(1.0, self.max_angular_speed), delta_theta)

        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.vel_pub.publish(msg)
        self.get_logger().info(f'\nCurrent pose: ({self.x:.2f}, {self.y:.2f}, {self.theta:.2f}), '
                               f'\ngoal: ({self.goal_x}, {self.goal_y}, {self.goal_theta}), '
                               f'\nattraction: ({self.attraction[0]:.2f}, {self.attraction[1]:.2f}), '
                               f'\nrepulsion: ({self.repulsion[0]:.2f}, {self.repulsion[1]:.2f})')


## Helper function to convert scan data from polar to Cartesian coordinates
def np_polar2cart(np_polar: np.ndarray):
    r = np_polar[:, 0]
    theta = np_polar[:, 1]
    np_cart = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    return np_cart

def euclid_distance(a_x, a_y, b_x, b_y):
    return math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2)

## Helper function to apply a given transform to an array of points
def apply_transform(points, transform):
    translation = np.array([transform.transform.translation.x, transform.transform.translation.y])

    return apply_rotation(points, transform) + translation

## Helper function to apply the rotation from a given transform to an array of points
def apply_rotation(points, transform):
    quaternion = transform.transform.rotation
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    _, _, yaw = euler_from_quaternion([x, y, z, w])

    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    return points @ rotation_matrix.T

def main(args=None):
    rclpy.init(args=args)

    navigator = PotentialFieldNavigator()

    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
