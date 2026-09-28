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
REPULSION_C = 0.1   # attraction and repulsion are equal at a clearance of about 0.35 m
RHO_0 = 0.6         # repulsion is active below this clearance between robot footprint and obstacle

SAFETY_CLEARANCE = 0.1   # below this clearance, the robot only moves away from the obstacle
ROTATION_MARGIN = 0.05   # additional clearance needed to rotate in place
ROTATION_STEP = 0.1      # angle step (rad) for checking whether a rotation in place hits an obstacle
ESCAPE_SPEED = 0.1       # speed for moving away from an obstacle
ESCAPE_TIMEOUT = 3.0     # s, if rotating is still not possible after moving away this long, the robot waits
TRAIL_SPACING = 0.05     # m, the driven path is recorded with this spacing ...
TRAIL_LENGTH = 5.0       # m, ... up to this length
BACKUP_STEP = 0.2        # m, if rotating is blocked in both directions, the robot moves back along the driven path
MAX_BACKUP = 3.0         # m, in steps of BACKUP_STEP, up to MAX_BACKUP, and tries to rotate again after every step
                         # (enough to move back out of a narrow passage)
BACKUP_SPEED = 0.1       # m/s
BACKUP_TOLERANCE = 0.03  # m
OBSTACLE_MEMORY_TIME = 300.0  # s, laser points are remembered this long (obstacles next to / behind the robot
OBSTACLE_MEMORY_RANGE = 2.0   # m, within this distance, are outside of the laser's field of view, e.g. the walls of a
                              # passage the robot slowly drives through), unless the laser scanner sees through them
TF_TOLERANCE = 0.05           # s, the latest transform is used for a laser scan if it is at most this much older
MEMORY_CLEAR_MARGIN = 0.1     # m, a remembered point is seen through if all measurements within MEMORY_CLEAR_ANGLE
MEMORY_CLEAR_ANGLE = 0.05     # rad of its angle are this much farther (a wall seen at a grazing angle is hit farther away
                              # by the beam at the angle of a remembered wall point, but closer by a neighbouring beam)
OBSTACLE_MEMORY_CELL = 0.05   # m, remembered points are thinned out to one per grid cell
STALL_TIME = 2.0         # s, a rotation in place is measured over this time ...
STALL_RATIO = 0.1        # ... and counts as blocked if the robot turns less than this ratio of the commanded rotation
                         # (it can wiggle or creep along an obstacle, so the net rotation in the commanded direction
                         # is compared with the commanded one)
STALL_BLOCK_TIME = 5.0   # s, a blocked rotation direction is not used for this time, the other one is tried first
PASSAGE_MARGIN = 0.05    # m, if obstacles are close (e.g. in a narrow passage), the robot moves straight to the waypoint
PASSAGE_STEP = 0.05      # without rotating as long as the footprint keeps this clearance on the way (checked in steps
PASSAGE_ANGLE = math.pi / 3   # of PASSAGE_STEP) and the waypoint is within PASSAGE_ANGLE of the heading (in the
PASSAGE_SPEED = 0.15     # m/s  field of view of the laser scanner); if the clearance is already smaller, the robot
MIN_PASSAGE_CLEARANCE = 0.02  # must not get closer than now (e.g. to move towards the middle of the passage), and not
PASSAGE_TOLERANCE = 0.005     # below MIN_PASSAGE_CLEARANCE (PASSAGE_TOLERANCE for moving parallel to a wall)
GOAL_SLOWDOWN_DIST = 0.3 # m, distance to the waypoint at which slowing down starts, otherwise the robot circles around it

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

        # Rectangular robot footprint, the laser scanner is mounted in the middle of the front side
        self.robot_length = self.declare_parameter('robot_length', 0.76).value
        self.robot_width = self.declare_parameter('robot_width', 0.47).value
        self.laser_to_front = self.declare_parameter('laser_to_front', 0.05).value
        # Footprint in base_link frame (front, rear, half width), set once the laser position is known
        self.footprint = None

        # Clearance to the closest obstacle and unit vector from the footprint towards it (base_link frame),
        # and all obstacle points outside of the footprint (base_link frame) for checking rotations in place
        self.min_clearance = math.inf
        self.obstacle_direction = 0.0, 0.0
        self.obstacle_points = np.zeros((0, 2))

        # Remembered laser points (odom frame) and the time they were measured
        self.memory_points = np.zeros((0, 2))
        self.memory_times = np.zeros(0)

        # Start time of moving away from an obstacle because a rotation in place is not possible
        self.escape_start = None

        # Driven path (odom frame, newest last), current target when moving back along it (and its index in the
        # driven path), and the distance already moved back since the last successful rotation
        self.trail = []
        self.backup_target = None
        self.backup_index = None
        self.backed_up = 0.0

        # Detection of rotations that are physically blocked (e.g. by an obstacle outside of the laser's field of
        # view): start time and orientation of the current rotation in place
        self.rotation_start = None
        self.stalled_until = {1.0: 0.0, -1.0: 0.0}   # per rotation direction

        # Direction of the ongoing rotation in place, kept until the rotation is finished; otherwise the direction
        # flips between +180 and -180 degrees if the desired direction is right behind the robot. It is only kept
        # within one phase (turning towards the driving direction / to the orientation at the waypoint).
        self.rotation_direction = None
        self.rotation_phase = None
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

        # Record the driven path (not while moving back along it)
        if self.backup_target is None and \
                (not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > TRAIL_SPACING):
            self.trail.append((self.x, self.y))
            if len(self.trail) > TRAIL_LENGTH / TRAIL_SPACING:
                self.trail.pop(0)

        if self.odom_base_transform:
            vel_x, vel_y = self.get_attraction(self.x, self.y)
            self.attraction = tuple(apply_rotation(np.array([[vel_x, vel_y]]), self.odom_base_transform)[0])

    def update_goal(self, msg):
        goal_quat = msg.pose.orientation
        goal_yaw = euler_from_quaternion([goal_quat.x, goal_quat.y, goal_quat.z, goal_quat.w])[2]
        self.waypoint = (msg.pose.position.x, msg.pose.position.y, goal_yaw)
        self.has_goal = True
        self.escape_start = None
        self.rotation_direction = None
        self.backup_target = None
        self.backed_up = 0.0

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
        if not self.laser_base_transform or self.footprint is None:
            return

        # Invalid measurements (e.g. 0 on the real laser scanner) would create a huge repulsion
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        coords_polar = np.column_stack((ranges[valid], angles[valid]))
        coords_clean = np_polar2cart(coords_polar)
        coords_transformed = self.update_obstacle_memory(apply_transform(coords_clean, self.laser_base_transform), msg)

        # Points inside the footprint are reflections from the robot itself
        clearances, closest_points = self.get_footprint_clearance(coords_transformed)
        outside = clearances > 0.01
        coords_transformed, clearances, closest_points = \
            coords_transformed[outside], clearances[outside], closest_points[outside]

        self.obstacle_points = coords_transformed
        if len(coords_transformed) == 0:
            self.repulsion = 0.0, 0.0
            self.min_clearance = math.inf
            return

        # Repulsive field depends on the minimum distance between the robot footprint and an obstacle
        # (configuration space), summing over all scan points would count a wall many times
        closest = np.argmin(clearances)
        self.min_clearance = clearances[closest]
        delta = coords_transformed[closest] - closest_points[closest]
        self.obstacle_direction = tuple(delta / max(np.hypot(delta[0], delta[1]), ZERO_REPLACEMENT))
        self.repulsion = self.get_repulsion(self.min_clearance, *self.obstacle_direction)


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
        _, first = np.unique(np.floor(all_points / OBSTACLE_MEMORY_CELL), axis=0, return_index=True)
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

    def get_footprint_clearance(self, points):
        """ Get the distance of every point (base_link frame) to the rectangular footprint and the closest
        point of the footprint (distance 0 for points inside the footprint) """
        front, rear, half_width = self.footprint
        closest_points = np.column_stack((np.clip(points[:, 0], -rear, front),
                                          np.clip(points[:, 1], -half_width, half_width)))
        clearances = np.hypot(points[:, 0] - closest_points[:, 0], points[:, 1] - closest_points[:, 1])
        return clearances, closest_points

    def get_attraction(self, x, y):
        """ Calculate attractive velocity for a given point in world frame with respect to the goal position """
        delta_x = x - self.goal_x
        delta_y = y - self.goal_y
        distance = max(math.sqrt(delta_x ** 2 + delta_y ** 2), ZERO_REPLACEMENT)
        return (- ATTRACTION_C * delta_x / distance), (- ATTRACTION_C * delta_y / distance)

    def get_repulsion(self, clearance, obstacle_direction_x, obstacle_direction_y):
        """ Calculate repulsive velocity (base_link frame) from the clearance to an obstacle and the direction
        towards it, i.e. the distance to the obstacle in the configuration space """
        distance = max(clearance, ZERO_REPLACEMENT)

        if distance < RHO_0:
            force_factor = REPULSION_C * (1.0 / distance - 1.0 / RHO_0) * (1 / distance ** 2)
            return -force_factor * obstacle_direction_x, -force_factor * obstacle_direction_y

        return 0.0, 0.0

    def escape(self, msg):
        """ Move away from the closest obstacle without rotating (the robot is omnidirectional) """
        msg.linear.x = -ESCAPE_SPEED * self.obstacle_direction[0]
        msg.linear.y = -ESCAPE_SPEED * self.obstacle_direction[1]
        msg.angular.z = 0.0

    def rotation_free(self, angle):
        """ Check whether the footprint hits an obstacle when rotating in place by the given angle: rotating the
        robot by an angle corresponds to rotating the obstacle points by the negative angle in base_link """
        if len(self.obstacle_points) == 0:
            return True
        steps = np.arange(ROTATION_STEP, abs(angle) + ROTATION_STEP, ROTATION_STEP)
        thetas = -math.copysign(1.0, angle) * np.minimum(steps, abs(angle))
        cos, sin = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
        x, y = self.obstacle_points[:, 0][None, :], self.obstacle_points[:, 1][None, :]
        rotated = np.stack((cos * x - sin * y, sin * x + cos * y), axis=-1).reshape(-1, 2)
        clearances, _ = self.get_footprint_clearance(rotated)
        return bool(np.all(clearances >= ROTATION_MARGIN))

    def straight_motion_free(self, goal_x, goal_y):
        """ Check whether the footprint can move in a straight line without rotating to the given point (base_link
        frame) while keeping PASSAGE_MARGIN to all obstacles, or at least the current clearance if it is smaller """
        distance = math.hypot(goal_x, goal_y)
        if len(self.obstacle_points) == 0 or distance < ZERO_REPLACEMENT:
            return True
        required = min(PASSAGE_MARGIN, self.min_clearance - PASSAGE_TOLERANCE)
        steps = np.minimum(np.arange(PASSAGE_STEP, distance + PASSAGE_STEP, PASSAGE_STEP), distance)
        offsets = np.column_stack((steps * goal_x / distance, steps * goal_y / distance))
        shifted = (self.obstacle_points[None, :, :] - offsets[:, None, :]).reshape(-1, 2)
        clearances, _ = self.get_footprint_clearance(shifted)
        return bool(np.all(clearances >= required))

    def rotation_stalled(self, direction, speed):
        """ Detect a rotation in place (direction +1 / -1, commanded angular speed) that does not turn the robot in
        that direction, i.e. that is physically blocked """
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.rotation_start is None or self.rotation_start[2] != direction:
            self.rotation_start = (now, self.theta, direction)
            return False
        start_time, start_theta, _ = self.rotation_start
        if now - start_time < STALL_TIME:
            return False
        turned = direction * math.atan2(math.sin(self.theta - start_theta), math.cos(self.theta - start_theta))
        if turned > STALL_RATIO * abs(speed) * (now - start_time):
            # Rotation makes progress, start a new measuring window
            self.rotation_start = (now, self.theta, direction)
            return False
        self.get_logger().warning(f'\nRotation is blocked (robot does not turn), trying the other direction')
        self.rotation_start = None
        self.stalled_until[direction] = now + STALL_BLOCK_TIME
        return True

    def get_rotation_direction(self, angle):
        """ Direction (+1 / -1) in which the robot can rotate by the given angle, the other way round if the short
        way is blocked (by the footprint check or because it was just physically blocked); 0 if both are blocked """
        now = self.get_clock().now().nanoseconds * 1e-9
        for way in (angle, angle - math.copysign(2 * math.pi, angle)):
            direction = math.copysign(1.0, way)
            if now >= self.stalled_until[direction] and self.rotation_free(way):
                return direction
        return 0.0

    def get_backup_target(self):
        """ Point on the driven path about BACKUP_STEP behind the robot and its index, if the footprint is free there
        (with the current orientation, the robot moves back without rotating); the driven path is only shortened
        once the point is reached """
        distance = 0.0
        last = (self.x, self.y)
        index = len(self.trail)
        while index > 0 and distance < BACKUP_STEP:
            index -= 1
            point = self.trail[index]
            distance += math.hypot(point[0] - last[0], point[1] - last[1])
            last = point
        if distance < BACKUP_TOLERANCE:
            return None, None

        # Obstacle points relative to the footprint at the target (base_link frame of the current pose)
        cos, sin = math.cos(self.theta), math.sin(self.theta)
        offset_x = cos * (last[0] - self.x) + sin * (last[1] - self.y)
        offset_y = -sin * (last[0] - self.x) + cos * (last[1] - self.y)
        if len(self.obstacle_points):
            clearances, _ = self.get_footprint_clearance(self.obstacle_points - np.array([offset_x, offset_y]))
            if np.any(clearances < ROTATION_MARGIN):
                return None, None
        return last, index

    def back_up_or_escape(self, msg):
        """ Rotating in place is blocked in both directions: move back along the driven path (known to be free) in
        small steps and try to rotate again after every step; if that is not possible, move away / wait """
        if self.backup_target is None and self.backed_up < MAX_BACKUP:
            self.backup_target, self.backup_index = self.get_backup_target()
            if self.backup_target is not None:
                self.get_logger().warning(f'\nNot enough space to rotate, moving back along the driven path '
                                          f'({self.backed_up + BACKUP_STEP:.1f} m)')

        if self.backup_target is None:
            self.escape_or_wait(msg)
            return

        delta_x, delta_y = self.backup_target[0] - self.x, self.backup_target[1] - self.y
        distance = math.hypot(delta_x, delta_y)
        if distance < BACKUP_TOLERANCE:
            # Step done, the rotation is checked again in the next control step
            del self.trail[self.backup_index:]
            self.backup_target = None
            self.backed_up += BACKUP_STEP
            msg.linear.x = msg.linear.y = msg.angular.z = 0.0
            return

        # Move towards the target without rotating (velocity in base_link frame, the robot is omnidirectional)
        cos, sin = math.cos(self.theta), math.sin(self.theta)
        speed = min(BACKUP_SPEED, distance)
        msg.linear.x = speed * (cos * delta_x + sin * delta_y) / distance
        msg.linear.y = speed * (-sin * delta_x + cos * delta_y) / distance
        msg.angular.z = 0.0

    def get_committed_direction(self, angle, phase):
        """ Direction of a rotation in place by the given angle: the direction of the ongoing rotation in the same phase
        as long as it is possible, otherwise a new one (see get_rotation_direction) """
        now = self.get_clock().now().nanoseconds * 1e-9
        if phase != self.rotation_phase:
            # E.g. arrived at the waypoint: the direction used for turning towards it is not the short way to the
            # orientation at the waypoint
            self.rotation_direction = None
            self.rotation_phase = phase
        direction = self.rotation_direction
        if direction is not None:
            way = angle if math.copysign(1.0, angle) == direction else angle - math.copysign(2 * math.pi, angle)
            if now >= self.stalled_until[direction] and self.rotation_free(way):
                return direction
        direction = self.get_rotation_direction(angle)
        self.rotation_direction = direction if direction != 0.0 else None
        return direction

    def escape_or_wait(self, msg):
        """ Rotating in place is not possible: move away from the closest obstacle for a limited time, then wait
        (e.g. until the explorer selects another goal), instead of alternating between rotating and moving away """
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.escape_start is None:
            self.escape_start = now
            self.get_logger().warning(f'\nNot enough space to rotate, moving away from the closest obstacle')
        if now - self.escape_start < ESCAPE_TIMEOUT:
            self.escape(msg)
        else:
            msg.linear.x = msg.linear.y = msg.angular.z = 0.0
            self.get_logger().warning(f'\nStill not enough space to rotate, waiting for another waypoint',
                                      throttle_duration_sec=2.0)

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

        delta_x = self.goal_x - self.x
        delta_y = self.goal_y - self.y
        delta_theta = self.goal_theta - self.theta

        # Normalize angles
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))

        # Check whether current position is within radial threshold around goal position
        pos_reached = (delta_x ** 2 + delta_y ** 2) < THRESHOLD_POSE ** 2

        # Waypoint in base_link frame
        cos, sin = math.cos(self.theta), math.sin(self.theta)
        goal_base_x = cos * delta_x + sin * delta_y
        goal_base_y = -sin * delta_x + cos * delta_y

        if not pos_reached and MIN_PASSAGE_CLEARANCE <= self.min_clearance < RHO_0 and \
                abs(math.atan2(goal_base_y, goal_base_x)) < PASSAGE_ANGLE and \
                self.straight_motion_free(goal_base_x, goal_base_y):
            # Obstacles are close (e.g. the sides of a narrow passage), but the footprint can move straight to the
            # waypoint: move there without rotating (the robot is omnidirectional), otherwise the repulsion of the
            # sides turns the robot away from the passage and it never passes through
            distance = math.hypot(goal_base_x, goal_base_y)
            speed = min(PASSAGE_SPEED, self.max_linear_speed) * min(1.0, distance / GOAL_SLOWDOWN_DIST)
            msg.linear.x = speed * goal_base_x / distance
            msg.linear.y = speed * goal_base_y / distance
            self.escape_start = None
            self.backup_target = None
            self.backed_up = 0.0
            self.get_logger().info(f'\nObstacle {self.min_clearance:.2f} m from the robot, but the way is free, '
                                   f'moving straight to the waypoint', throttle_duration_sec=1.0)

        elif not pos_reached:
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
            goal_slow = min(1.0, math.hypot(delta_x, delta_y) / GOAL_SLOWDOWN_DIST)

            # Final forward speed
            v_cmd = V_REF * turn_slow * obs_slow
            msg.linear.x = float(max(self.min_linear_speed, min(self.max_linear_speed, v_cmd)) * goal_slow)

            # Rotate in place if the desired direction is to the side or behind, driving would lead to circles
            ROTATE_IN_PLACE_ANGLE = math.pi / 2
            safety_clearance = SAFETY_CLEARANCE
            if abs(desired_theta) > ROTATE_IN_PLACE_ANGLE:
                msg.linear.x = 0.0
                direction = self.get_committed_direction(desired_theta, 'driving')
                if direction != 0.0 and not self.rotation_stalled(direction, omega):
                    msg.angular.z = direction * abs(omega)
                    # The rotation was checked to keep ROTATION_MARGIN, moving away before would interrupt it
                    safety_clearance = ROTATION_MARGIN
                    if self.min_clearance < SAFETY_CLEARANCE:
                        self.get_logger().info(f'\nRotating close to an obstacle ({self.min_clearance:.2f} m)',
                                               throttle_duration_sec=1.0)
                    self.escape_start = None
                    self.backup_target = None
                    self.backed_up = 0.0
                else:
                    # A corner would hit an obstacle in both directions
                    self.back_up_or_escape(msg)

            # Too close to an obstacle: only move away from it
            if self.min_clearance < safety_clearance:
                self.escape(msg)
                self.get_logger().warning(f'\nObstacle {self.min_clearance:.2f} m from the robot, moving away')

            self.get_logger().info(f'\nforces: ({vel_x:.2f}, {vel_y:.2f})')

        elif abs(delta_theta) > THRESHOLD_ROTATION:
            direction = self.get_committed_direction(delta_theta, 'waypoint')
            if direction == 0.0 or self.rotation_stalled(direction, min(1.0, self.max_angular_speed)):
                # Waypoint is too close to an obstacle to rotate to the desired orientation, stay
                self.get_logger().warning(f'\nNot enough space to rotate at the waypoint, keeping the orientation',
                                          throttle_duration_sec=2.0)
            else:
                msg.angular.z = direction * min(1.0, self.max_angular_speed)

        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        # Stall detection and the kept rotation direction only apply to consecutive rotations in place
        if msg.angular.z == 0.0 or msg.linear.x != 0.0 or msg.linear.y != 0.0:
            self.rotation_start = None
            self.rotation_direction = None

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
