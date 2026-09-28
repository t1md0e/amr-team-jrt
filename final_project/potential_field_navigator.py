import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

from tf_transformations import euler_from_quaternion
import tf2_ros

# Potential field in the configuration space q = (x, y, theta) of the robot (lecture chapter 5):
#   U(q) = U_att(q) + U_rep(q),  F(q) = -grad U(q)
# The robot is omnidirectional, so F is used directly as velocity in x, y and theta. The repulsive potential is
# computed from the clearance between the rectangular robot footprint at q and every obstacle, so a rotation that
# brings a corner closer to a wall increases U and is avoided without special cases.

ATTRACTION_C = 1.0         # attractive potential: quadratic within ATTRACTION_D0 of the waypoint, conic (constant
ATTRACTION_D0 = 0.3        # force ATTRACTION_C) beyond it, so the robot slows down only close to the waypoint
HEADING_C = 1.0            # attractive potential of the orientation: towards the waypoint (the laser scanner looks
HEADING_E0 = 0.3           # in driving direction), at the waypoint towards its orientation; quadratic below HEADING_E0

REPULSION_C = 0.0005       # repulsive potential 1/2 k (1/rho - 1/rho_0)^2 of every obstacle point (one point per
RHO_0 = 0.4                # OBSTACLE_CELL, so a wall counts by its length and not by the number of laser points)
MIN_CLEARANCE = 0.01       # clearances are limited to this value, the potential would be infinite at 0
OBSTACLE_CELL = 0.05       # m, obstacle points are thinned out to one per grid cell

GRADIENT_STEP_XY = 0.01    # m, step for the numerical gradient in x and y
GRADIENT_STEP_THETA = 0.01 # rad, step for the numerical gradient in theta
LINE_SEARCH = (1.0, 0.5, 0.25, 0.125)   # a step along -grad U is only taken if U decreases (otherwise the robot
                                        # would overshoot and oscillate between the walls of a narrow passage)
BLIND_SPEED_FACTOR = 0.5   # translation away from the laser's field of view (backwards) is slowed down to this factor

STUCK_TIME = 5.0           # s, if the robot gets less than STUCK_PROGRESS closer to the waypoint within STUCK_TIME,
STUCK_PROGRESS = 0.05      # m  it is in a local minimum of the potential field ...
RANDOM_WALK_TIME = (2.0, 4.0)   # s, ... and escapes with a random walk: for a random time, the attractive potential
                                # is replaced by one in a random direction

OBSTACLE_MEMORY_TIME = 300.0  # s, laser points are remembered this long (obstacles next to / behind the robot
OBSTACLE_MEMORY_RANGE = 2.0   # m, within this distance, are outside of the laser's field of view, e.g. the walls of a
                              # passage the robot slowly drives through), unless the laser scanner sees through them
TF_TOLERANCE = 0.05           # s, the latest transform is used for a laser scan if it is at most this much older
MEMORY_CLEAR_MARGIN = 0.1     # m, a remembered point is seen through if all measurements within MEMORY_CLEAR_ANGLE
MEMORY_CLEAR_ANGLE = 0.05     # rad of its angle are this much farther (a wall seen at a grazing angle is hit farther away
                              # by the beam at the angle of a remembered wall point, but closer by a neighbouring beam)

THRESHOLD_ROTATION = 0.1
THRESHOLD_POSE = 0.1

CONTROL_PERIOD = 0.1
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

        # Speed limits, slow defaults for the real robot (the simulation was tested with 0.8 / 1.5)
        self.max_linear_speed = self.declare_parameter('max_linear_speed', 0.3).value
        self.max_angular_speed = self.declare_parameter('max_angular_speed', 0.8).value
        # Not needed any more (the speed follows from the potential field), declared so that old launch arguments work
        self.declare_parameter('min_linear_speed', 0.1)
        # Speed that the drive loses (e.g. by friction), added to every command that is not zero: the simulated robot
        # hardly moves below 0.25 m/s and 0.5 rad/s, then the robot would not execute the small steps of the potential
        # field (0 for the real robot)
        self.linear_deadband = self.declare_parameter('linear_deadband', 0.0).value
        self.angular_deadband = self.declare_parameter('angular_deadband', 0.0).value

        # Rectangular robot footprint, the laser scanner is mounted in the middle of the front side
        self.robot_length = self.declare_parameter('robot_length', 0.76).value
        self.robot_width = self.declare_parameter('robot_width', 0.47).value
        self.laser_to_front = self.declare_parameter('laser_to_front', 0.05).value
        # Footprint in base_link frame (front, rear, half width), set once the laser position is known
        self.footprint = None

        # Obstacle points outside of the footprint (base_link frame, one per OBSTACLE_CELL) and the clearance to the
        # closest one
        self.obstacle_points = np.zeros((0, 2))
        self.min_clearance = math.inf

        # Remembered laser points (odom frame) and the time they were measured
        self.memory_points = np.zeros((0, 2))
        self.memory_times = np.zeros(0)

        # Detection of local minima: time and distance to the waypoint of the last progress, and the random direction
        # (base_link frame at the start) and end time of a random walk
        self.progress = None
        self.random_walk = None

        # The robot has reached the position of the waypoint and only turns to its orientation (kept until the next
        # waypoint, otherwise the robot switches between turning towards the waypoint and to its orientation)
        self.position_reached = False

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

        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)

    def update_pose(self, msg):
        """ Get current position from /odom topic """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        quaternion = msg.pose.pose.orientation
        self.theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]

    def update_goal(self, msg):
        goal_quat = msg.pose.orientation
        goal_yaw = euler_from_quaternion([goal_quat.x, goal_quat.y, goal_quat.z, goal_quat.w])[2]
        self.waypoint = (msg.pose.position.x, msg.pose.position.y, goal_yaw)
        self.has_goal = True
        self.progress = None
        self.random_walk = None
        self.position_reached = False

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
        """ Get the obstacle points (base_link frame) from /scan topic and the obstacle memory """
        self.laser_frame = msg.header.frame_id
        if not self.laser_base_transform or self.footprint is None:
            return

        # Invalid measurements (e.g. 0 on the real laser scanner) would create a huge repulsion
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        coords_polar = np.column_stack((ranges[valid], angles[valid]))
        coords_clean = np_polar2cart(coords_polar)
        coords_transformed = self.update_obstacle_memory(apply_transform(coords_clean, self.laser_base_transform), msg)

        # One point per grid cell, so that every obstacle contributes to the potential according to its size
        if len(coords_transformed):
            _, first = np.unique(np.floor(coords_transformed / OBSTACLE_CELL), axis=0, return_index=True)
            coords_transformed = coords_transformed[first]

        # Points inside the footprint are reflections from the robot itself
        clearances = footprint_clearance(coords_transformed, self.footprint)
        outside = clearances > 0.01
        self.obstacle_points = coords_transformed[outside]
        self.min_clearance = np.min(clearances[outside]) if np.any(outside) else math.inf

    def update_obstacle_memory(self, points, msg):
        """ Add the current laser points (base_link frame) of the scan msg to the obstacle memory and return all
        remembered points in base_link frame, so that obstacles that have left the laser's field of view are still
        considered """
        now = self.get_clock().now().nanoseconds * 1e-9

        # base_link -> odom with the pose at the time of the scan (with the current pose, the points would be
        # smeared while the robot rotates and remain as ghost obstacles)
        try:
            transform = self.lookup_transform_at("odom", "base_link", rclpy.time.Time.from_msg(msg.header.stamp))
            points_odom = apply_transform(points, transform)
            self.forget_seen_through(transform, msg)
        except tf2_ros.TransformException:
            points_odom = np.zeros((0, 2))
        all_points = np.vstack((points_odom, self.memory_points))
        all_times = np.concatenate((np.full(len(points_odom), now), self.memory_times))

        # Forget old and far away points, keep the newest point per grid cell
        keep = (now - all_times < OBSTACLE_MEMORY_TIME) & \
            (np.hypot(all_points[:, 0] - self.x, all_points[:, 1] - self.y) < OBSTACLE_MEMORY_RANGE)
        all_points, all_times = all_points[keep], all_times[keep]
        _, first = np.unique(np.floor(all_points / OBSTACLE_CELL), axis=0, return_index=True)
        self.memory_points, self.memory_times = all_points[first], all_times[first]

        # odom -> base_link with the current pose; the current scan is used directly, so only older points are
        # taken from the memory
        old = self.memory_points[self.memory_times < now]
        cos, sin = math.cos(self.theta), math.sin(self.theta)
        delta_x, delta_y = old[:, 0] - self.x, old[:, 1] - self.y
        remembered = np.column_stack((cos * delta_x + sin * delta_y, -sin * delta_x + cos * delta_y))
        return np.vstack((points, remembered))

    def lookup_transform_at(self, target_frame, source_frame, time):
        """ Transform at the given time; the odometry can be published slightly later than the laser scan (e.g. 2 ms
        in simulation), then the latest transform is used if it is at most TF_TOLERANCE older """
        try:
            return self.tf_buffer.lookup_transform(target_frame, source_frame, time)
        except tf2_ros.ExtrapolationException:
            latest = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            if (time - rclpy.time.Time.from_msg(latest.header.stamp)).nanoseconds * 1e-9 > TF_TOLERANCE:
                raise
            return latest

    def forget_seen_through(self, transform, msg):
        """ Forget remembered points that the laser scanner sees through, i.e. the measurement at their angle is
        farther away (e.g. a person who has walked on), otherwise they would remain as ghost obstacles; transform is
        base_link -> odom at the time of the scan """
        if len(self.memory_points) == 0:
            return
        # odom -> base_link -> laser frame
        points = apply_inverse_transform(apply_inverse_transform(self.memory_points, transform), self.laser_base_transform)
        distances = np.hypot(points[:, 0], points[:, 1])
        index = np.round(np.mod(np.arctan2(points[:, 1], points[:, 0]) - msg.angle_min, 2 * math.pi)
                         / msg.angle_increment).astype(int)

        # No echo within the range of the scanner (inf) means free space, invalid measurements (e.g. 0 on the real
        # laser scanner) never see through a point
        ranges = np.array(msg.ranges, dtype=float)
        ranges[np.isposinf(ranges)] = msg.range_max
        ranges[~(np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max))] = -np.inf
        # Smallest measurement within MEMORY_CLEAR_ANGLE
        beams = int(math.ceil(MEMORY_CLEAR_ANGLE / msg.angle_increment))
        padded = np.pad(ranges, beams, constant_values=np.inf)
        window_min = np.min([padded[i:i + len(ranges)] for i in range(2 * beams + 1)], axis=0)

        in_view = index < len(ranges)
        measured = np.full(len(points), -np.inf)
        measured[in_view] = window_min[index[in_view]]
        keep = ~(distances < measured - MEMORY_CLEAR_MARGIN)
        self.memory_points, self.memory_times = self.memory_points[keep], self.memory_times[keep]

    def get_attraction_target(self):
        """ Target of the attractive potential in base_link frame: waypoint position, orientation at the waypoint
        (relative to the robot) and whether the robot is at the waypoint (then it turns to that orientation);
        during a random walk, a point far away in the random direction """
        delta_x = self.goal_x - self.x
        delta_y = self.goal_y - self.y
        cos, sin = math.cos(self.theta), math.sin(self.theta)
        goal_base = np.array([cos * delta_x + sin * delta_y, -sin * delta_x + cos * delta_y])
        self.position_reached = self.position_reached or np.hypot(*goal_base) < THRESHOLD_POSE
        at_waypoint = self.position_reached
        delta_theta = math.atan2(math.sin(self.goal_theta - self.theta), math.cos(self.goal_theta - self.theta))
        return goal_base, delta_theta, at_waypoint

    def update_random_walk(self, distance):
        """ Detect a local minimum (no progress towards the waypoint for STUCK_TIME) and start a random walk,
        return whether a random walk is active """
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.random_walk is not None:
            if now < self.random_walk[1]:
                return True
            self.random_walk = None
            self.progress = None

        if self.progress is None or distance < self.progress[1] - STUCK_PROGRESS:
            self.progress = (now, distance)
        elif now - self.progress[0] > STUCK_TIME and distance < ATTRACTION_D0:
            # Close to the waypoint, but obstacles keep the robot away from it (e.g. a waypoint close to a wall):
            # as close as possible, turn to the orientation of the waypoint
            self.position_reached = True
            self.get_logger().warning(f'\nWaypoint cannot be reached closer than {distance:.2f} m, turning to its '
                                      f'orientation')
        elif now - self.progress[0] > STUCK_TIME:
            angle = np.random.uniform(-math.pi, math.pi)
            # Direction in odom frame, so that it does not turn with the robot
            direction = (math.cos(self.theta + angle), math.sin(self.theta + angle))
            self.random_walk = (direction, now + np.random.uniform(*RANDOM_WALK_TIME))
            self.get_logger().warning(f'\nLocal minimum (no progress for {STUCK_TIME:.0f} s), random walk in '
                                      f'direction {math.degrees(angle):.0f} deg')
            return True
        return False

    def potential(self, deltas, goal_base, delta_theta, at_waypoint, random_direction):
        """ Potential U(q) for the robot moved by the given displacements (x, y, theta in base_link frame of the
        current pose, one row per displacement) """
        dx, dy, dtheta = deltas[:, 0:1], deltas[:, 1:2], deltas[:, 2:3]
        cos, sin = np.cos(dtheta), np.sin(dtheta)

        # Repulsive potential: obstacle points relative to the moved footprint
        px = self.obstacle_points[None, :, 0] - dx
        py = self.obstacle_points[None, :, 1] - dy
        points = np.stack((cos * px + sin * py, -sin * px + cos * py), axis=-1)
        clearances = np.maximum(footprint_clearance(points.reshape(-1, 2), self.footprint).reshape(points.shape[:2]),
                                MIN_CLEARANCE)
        repulsion = np.where(clearances < RHO_0, 0.5 * REPULSION_C * (1.0 / clearances - 1.0 / RHO_0) ** 2, 0.0)
        u_rep = np.sum(repulsion, axis=1)

        if random_direction is not None:
            # Random walk: constant force in the random direction, no preferred orientation
            return u_rep - ATTRACTION_C * (deltas[:, 0] * random_direction[0] + deltas[:, 1] * random_direction[1])

        # Attractive potential of the position: waypoint relative to the moved robot
        gx, gy = goal_base[0] - dx[:, 0], goal_base[1] - dy[:, 0]
        distance = np.hypot(gx, gy)
        u_att = ATTRACTION_C * np.where(distance < ATTRACTION_D0, 0.5 * distance ** 2 / ATTRACTION_D0,
                                        distance - 0.5 * ATTRACTION_D0)

        # Attractive potential of the orientation: towards the waypoint while driving, towards the orientation of
        # the waypoint at the waypoint
        if at_waypoint:
            heading_error = delta_theta - dtheta[:, 0]
            weight = 1.0
        else:
            heading_error = np.arctan2(gy, gx) - dtheta[:, 0]
            weight = np.minimum(1.0, distance / ATTRACTION_D0)
        heading_error = np.abs(np.arctan2(np.sin(heading_error), np.cos(heading_error)))
        u_heading = HEADING_C * weight * np.where(heading_error < HEADING_E0, 0.5 * heading_error ** 2 / HEADING_E0,
                                                  heading_error - 0.5 * HEADING_E0)
        return u_rep + u_att + u_heading

    def get_step(self, goal_base, delta_theta, at_waypoint, random_direction):
        """ Step (x, y, theta in base_link frame) along the negative gradient of the potential for one control period,
        None if no step decreases the potential """
        args = (goal_base, delta_theta, at_waypoint, random_direction)

        # Numerical gradient (central differences)
        h = np.array([GRADIENT_STEP_XY, GRADIENT_STEP_XY, GRADIENT_STEP_THETA])
        deltas = np.vstack((np.zeros(3), np.diag(h), -np.diag(h)))
        u = self.potential(deltas, *args)
        gradient = (u[1:4] - u[4:7]) / (2 * h)

        # F = -grad U; the attractive forces are at most 1, so a force of 1 corresponds to the maximum speed
        max_xy = self.max_linear_speed * CONTROL_PERIOD
        max_theta = self.max_angular_speed * CONTROL_PERIOD
        step = -gradient * np.array([max_xy, max_xy, max_theta])
        length = np.hypot(step[0], step[1])
        if length > max_xy:
            step[:2] *= max_xy / length
        step[2] = np.clip(step[2], -max_theta, max_theta)

        # Moving backwards leaves the laser's field of view, only remembered obstacles are known there
        if length > ZERO_REPLACEMENT:
            step[:2] *= max(BLIND_SPEED_FACTOR, 0.5 * (1.0 + step[0] / length))

        # Line search: the full step, then shorter ones, then translation or rotation alone
        candidates = [factor * step for factor in LINE_SEARCH]
        candidates += [factor * step * np.array([1.0, 1.0, 0.0]) for factor in LINE_SEARCH]
        candidates += [factor * step * np.array([0.0, 0.0, 1.0]) for factor in LINE_SEARCH]
        u_candidates = self.potential(np.array(candidates), *args)
        for candidate, u_candidate in zip(candidates, u_candidates):
            if u_candidate < u[0] - ZERO_REPLACEMENT:
                return candidate
        return None

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
            if self.footprint is None:
                # Front side is just in front of the laser scanner, the rear side follows from the robot length
                front = self.laser_base_transform.transform.translation.x + self.laser_to_front
                self.footprint = front, self.robot_length - front, self.robot_width / 2
                self.get_logger().info(f'\nFootprint (base_link frame): front {front:.2f} m, '
                                       f'rear {self.robot_length - front:.2f} m, half width {self.robot_width / 2:.2f} m')
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
        goal_base, delta_theta, at_waypoint = self.get_attraction_target()

        if at_waypoint and abs(delta_theta) < THRESHOLD_ROTATION:
            # Waypoint reached with the desired orientation
            self.progress = None
            self.vel_pub.publish(msg)
            return

        # Random walk to escape a local minimum (not at the waypoint, there only the orientation is missing)
        random_direction = None
        if not at_waypoint and self.update_random_walk(np.hypot(*goal_base)):
            direction = self.random_walk[0]
            cos, sin = math.cos(self.theta), math.sin(self.theta)
            random_direction = (cos * direction[0] + sin * direction[1], -sin * direction[0] + cos * direction[1])

        step = self.get_step(goal_base, delta_theta, at_waypoint, random_direction)
        if step is None:
            self.get_logger().warning(f'\nNo step decreases the potential, waiting', throttle_duration_sec=2.0)
        else:
            velocity = step / CONTROL_PERIOD
            speed = np.hypot(velocity[0], velocity[1])
            if speed > ZERO_REPLACEMENT:
                velocity[:2] *= (speed + self.linear_deadband) / speed
            if abs(velocity[2]) > ZERO_REPLACEMENT:
                velocity[2] += math.copysign(self.angular_deadband, velocity[2])
            msg.linear.x, msg.linear.y, msg.angular.z = (float(value) for value in velocity)

        self.vel_pub.publish(msg)
        self.get_logger().info(f'\nCurrent pose: ({self.x:.2f}, {self.y:.2f}, {self.theta:.2f}), '
                               f'\ngoal: ({self.goal_x}, {self.goal_y}, {self.goal_theta}), '
                               f'\nvelocity: ({msg.linear.x:.2f}, {msg.linear.y:.2f}, {msg.angular.z:.2f}), '
                               f'\nclosest obstacle: {self.min_clearance:.2f} m')


## Helper function to get the distance of points (base_link frame) to the rectangular footprint
## (front, rear, half width), 0 for points inside the footprint
def footprint_clearance(points, footprint):
    front, rear, half_width = footprint
    closest_x = np.clip(points[:, 0], -rear, front)
    closest_y = np.clip(points[:, 1], -half_width, half_width)
    return np.hypot(points[:, 0] - closest_x, points[:, 1] - closest_y)

## Helper function to convert scan data from polar to Cartesian coordinates
def np_polar2cart(np_polar: np.ndarray):
    r = np_polar[:, 0]
    theta = np_polar[:, 1]
    np_cart = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    return np_cart

## Helper function to apply a given transform to an array of points
def apply_transform(points, transform):
    translation = np.array([transform.transform.translation.x, transform.transform.translation.y])

    return apply_rotation(points, transform) + translation

## Helper function to apply the inverse of a given transform to an array of points
def apply_inverse_transform(points, transform):
    translation = np.array([transform.transform.translation.x, transform.transform.translation.y])

    return apply_rotation(points - translation, transform, inverse=True)

## Helper function to apply the rotation from a given transform to an array of points
def apply_rotation(points, transform, inverse=False):
    quaternion = transform.transform.rotation
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    _, _, yaw = euler_from_quaternion([x, y, z, w])

    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    return points @ rotation_matrix if inverse else points @ rotation_matrix.T

def main(args=None):
    rclpy.init(args=args)

    navigator = PotentialFieldNavigator()

    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
