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
REPULSION_C = 3.0
RHO_0 = 1.5

MAX_SPEED = 4
MIN_SPEED = 0.5
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
        transform = self.map_odom_transform
        if transform is None:
            self.get_logger().warn('No map_odom transform available, cannot update goal')
            return
        transformed_coords = apply_transform(np.array([[msg.pose.position.x, msg.pose.position.y]]), transform)
        self.goal_x = transformed_coords[0][0]
        self.goal_y = transformed_coords[0][1]
        goal_quat = msg.pose.orientation
        goal_yaw = euler_from_quaternion([goal_quat.x, goal_quat.y, goal_quat.z, goal_quat.w])[2]
        transform_quat = transform.transform.rotation
        transform_yaw = euler_from_quaternion([transform_quat.x, transform_quat.y, transform_quat.z, transform_quat.w])[2]
        self.goal_theta = math.atan2(math.sin(goal_yaw + transform_yaw), math.cos(goal_yaw + transform_yaw))

    def update_obstacles(self, msg):
        """ Get obstacle distances from /scan topic and update repulsive velocity from obstacles """
        if not self.laser_base_transform:
            return

        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        coords_polar = np.column_stack((msg.ranges, angles))
        coords_cart = np_polar2cart(coords_polar)

        coords_clean = coords_cart[np.isfinite(coords_cart).all(axis=1)]
        coords_transformed = apply_transform(coords_clean, self.laser_base_transform)

        velocities = [self.get_repulsion(0.0, 0.0, p[0], p[1]) for p in coords_transformed]
        self.repulsion = sum([v[0] for v in velocities]), sum([v[1] for v in velocities])

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
        try:
            self.laser_base_transform = self.tf_buffer.lookup_transform(
                "base_link",
                "base_laser_front_link",
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
            MAX_ANG_VEL = 1.5
            omega = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, k_omega * desired_theta))
            msg.angular.z = omega

            # Near-constant forward speed with smooth slowdowns
            V_REF = 0.8                 # nominal forward speed (tune)
            TURN_SLOWDOWN_K = 0.8       # how much to slow when turning (tune)
            OBS_SLOWDOWN_K = 2.0        # how much to slow when repulsion is strong (tune)

            # 1) Slow down when turning sharply
            turn_slow = 1.0 / (1.0 + TURN_SLOWDOWN_K * abs(omega) / MAX_ANG_VEL)

            # 2) Slow down when repulsion is large (i.e., close to obstacles)
            rep_mag = math.hypot(self.repulsion[0], self.repulsion[1])
            obs_slow = 1.0 / (1.0 + OBS_SLOWDOWN_K * rep_mag)

            # Final forward speed
            v_cmd = V_REF * turn_slow * obs_slow
            msg.linear.x = float(max(MIN_SPEED, min(MAX_SPEED, v_cmd)))

            self.get_logger().info(f'\nforces: ({vel_x:.2f}, {vel_y:.2f})')

        elif abs(delta_theta) > THRESHOLD_ROTATION:
            msg.angular.z = math.copysign(1.0, delta_theta)

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
