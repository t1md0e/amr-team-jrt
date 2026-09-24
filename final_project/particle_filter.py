import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy

from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan

from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler
import tf2_ros
from tf2_ros import TransformBroadcaster

NUM_PARTICLES = 2000

# Noise parameters of the odometry motion model
ALPHA_1 = 0.1   # rotation noise caused by rotation
ALPHA_2 = 0.05  # rotation noise caused by translation
ALPHA_3 = 0.1   # translation noise caused by translation
ALPHA_4 = 0.05  # translation noise caused by rotation

# Measurement model
NUM_BEAMS = 30    # number of laser beams used for weighting a particle
SIGMA_HIT = 0.4   # standard deviation of a measured range around the expected range
Z_RANDOM = 0.05   # probability of a random measurement (e.g. dynamic obstacles)

# Filter is only updated if the robot has moved far enough
MIN_TRAVEL_DISTANCE = 0.05
MIN_TRAVEL_ROTATION = 0.05

# Short and long term average of the measurement likelihood (for adding random particles)
ALPHA_SLOW = 0.01
ALPHA_FAST = 0.2

# Spread of the particles around a pose given on /initialpose
INITIAL_POS_STD = 0.3
INITIAL_ROT_STD = 0.2

FREE_THRESHOLD = 50
ZERO_REPLACEMENT = 1e-6

class ParticleFilter(Node):

    def __init__(self):
        super().__init__('particle_filter')

        # Map from map_server is only published once, so the subscription needs to be transient local
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/estimated_pose', 10)
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.update_pose, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.update_scan, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.update_map, map_qos)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                                         self.update_initial_pose, 10)

        # Get listener for static transforms and broadcaster for map -> odom transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.laser_base_transform = None

        # Current odometry pose (odom frame) to be updated
        self.odom_pose = None

        # Odometry pose at the time of the last filter update
        self.prev_odom_pose = None

        # Last received LaserScan and OccupancyGrid msgs
        self.scan = None
        self.grid = None
        self.occupancy = None
        self.free_cells = None

        # Particles (x, y, theta in map frame) and their weights
        self.particles = None
        self.weights = None

        # Estimated pose (map frame) and its covariance
        self.estimate = None
        self.covariance = np.zeros(3)

        # Short and long term averages of the measurement likelihood
        self.w_slow = 0.0
        self.w_fast = 0.0

        self.timer = self.create_timer(0.1, self.control_loop)

    def update_pose(self, msg):
        """ Get current position from /odom topic """
        quaternion = msg.pose.pose.orientation
        theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]
        self.odom_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, theta])

    def update_scan(self, msg):
        self.scan = msg

    def update_map(self, msg):
        """ Save occupancy grid and initialise particles uniformly over the free cells (global localisation) """
        self.grid = msg
        self.occupancy = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        free_y, free_x = np.nonzero((self.occupancy >= 0) & (self.occupancy < FREE_THRESHOLD))
        self.free_cells = np.column_stack((free_x, free_y))

        if self.particles is None:
            self.particles = self.sample_uniform_particles(NUM_PARTICLES)
            self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
            self.get_logger().info(f'Map received, {NUM_PARTICLES} particles initialised uniformly')

    def update_initial_pose(self, msg):
        """ Reinitialise particles around a pose given e.g. by the 2D Pose Estimate widget in RViz """
        quaternion = msg.pose.pose.orientation
        theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]

        self.particles = np.column_stack((
            np.random.normal(msg.pose.pose.position.x, INITIAL_POS_STD, NUM_PARTICLES),
            np.random.normal(msg.pose.pose.position.y, INITIAL_POS_STD, NUM_PARTICLES),
            np.random.normal(theta, INITIAL_ROT_STD, NUM_PARTICLES),
        ))
        self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
        self.w_slow = 0.0
        self.w_fast = 0.0
        self.get_logger().info(f'\nParticles reinitialised around '
                               f'({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f}, {theta:.2f})')

    def sample_uniform_particles(self, count):
        """ Sample particles uniformly from the free cells of the map with a random orientation """
        cells = self.free_cells[np.random.randint(len(self.free_cells), size=count)]
        map_res = self.grid.info.resolution
        map_origin = self.grid.info.origin.position

        # Uniformly distributed position inside the sampled cell
        x = (cells[:, 0] + np.random.uniform(size=count)) * map_res + map_origin.x
        y = (cells[:, 1] + np.random.uniform(size=count)) * map_res + map_origin.y
        theta = np.random.uniform(-math.pi, math.pi, count)

        return np.column_stack((x, y, theta))

    def motion_update(self, odom_pose):
        """ Move all particles by sampling from the odometry motion model """
        delta_x = odom_pose[0] - self.prev_odom_pose[0]
        delta_y = odom_pose[1] - self.prev_odom_pose[1]

        # Decompose motion into initial rotation, translation and final rotation
        delta_trans = math.sqrt(delta_x ** 2 + delta_y ** 2)
        if delta_trans < 0.01:
            # Pure rotation, the direction of a tiny translation is meaningless
            delta_rot1 = 0.0
        else:
            delta_rot1 = normalize_angle(math.atan2(delta_y, delta_x) - self.prev_odom_pose[2])
        delta_rot2 = normalize_angle(odom_pose[2] - self.prev_odom_pose[2] - delta_rot1)

        # Components under the influence of noise, sampled for every particle
        rot1_hat = delta_rot1 - sample_normal(ALPHA_1 * delta_rot1 ** 2 + ALPHA_2 * delta_trans ** 2)
        trans_hat = delta_trans - sample_normal(ALPHA_3 * delta_trans ** 2 +
                                                ALPHA_4 * delta_rot1 ** 2 + ALPHA_4 * delta_rot2 ** 2)
        rot2_hat = delta_rot2 - sample_normal(ALPHA_1 * delta_rot2 ** 2 + ALPHA_2 * delta_trans ** 2)

        theta = self.particles[:, 2]
        self.particles[:, 0] += trans_hat * np.cos(theta + rot1_hat)
        self.particles[:, 1] += trans_hat * np.sin(theta + rot1_hat)
        self.particles[:, 2] = normalize_angle(theta + rot1_hat + rot2_hat)

    def measurement_update(self):
        """ Weight particles by comparing measured ranges with the ranges expected in the map """
        scan = self.scan
        beam_indices = np.linspace(0, len(scan.ranges) - 1, NUM_BEAMS).astype(int)
        beam_angles = scan.angle_min + beam_indices * scan.angle_increment

        # Measurements without a hit (inf) are treated as maximum range
        measured = np.array(scan.ranges)[beam_indices]
        measured = np.where(np.isfinite(measured), measured, scan.range_max)
        measured = np.clip(measured, scan.range_min, scan.range_max)

        expected = self.get_expected_ranges(beam_angles, scan.range_max)

        # Gaussian around expected range mixed with a uniform random measurement
        p_hit = np.exp(-0.5 * ((measured - expected) / SIGMA_HIT) ** 2) / (SIGMA_HIT * math.sqrt(2 * math.pi))
        p = (1.0 - Z_RANDOM) * p_hit + Z_RANDOM / scan.range_max

        # Sum log likelihoods of all beams to avoid numerical underflow
        log_likelihood = np.sum(np.log(p + ZERO_REPLACEMENT), axis=1)
        self.weights = np.exp(log_likelihood - np.max(log_likelihood))
        self.weights /= np.sum(self.weights)

        # Average likelihood per beam, used to decide on adding random particles
        w_avg = np.mean(np.exp(log_likelihood / NUM_BEAMS))
        if self.w_slow == 0.0:
            self.w_slow = w_avg
            self.w_fast = w_avg
        self.w_slow += ALPHA_SLOW * (w_avg - self.w_slow)
        self.w_fast += ALPHA_FAST * (w_avg - self.w_fast)

    def get_expected_ranges(self, beam_angles, range_max):
        """ Cast rays from the laser pose of every particle and return the distance to the first occupied cell """
        map_res = self.grid.info.resolution
        map_origin = self.grid.info.origin.position
        height, width = self.occupancy.shape

        # Laser pose of every particle (laser is mounted with an offset to base_link)
        laser_x, laser_y, laser_yaw = self.laser_base_transform
        theta = self.particles[:, 2]
        origin_x = self.particles[:, 0] + np.cos(theta) * laser_x - np.sin(theta) * laser_y
        origin_y = self.particles[:, 1] + np.sin(theta) * laser_x + np.cos(theta) * laser_y

        # Points along every ray, shape: (particles, beams, steps)
        steps = np.arange(0.0, range_max, map_res)
        angles = theta[:, None] + laser_yaw + beam_angles[None, :]
        points_x = origin_x[:, None, None] + np.cos(angles)[:, :, None] * steps[None, None, :]
        points_y = origin_y[:, None, None] + np.sin(angles)[:, :, None] * steps[None, None, :]

        cells_x = np.floor((points_x - map_origin.x) / map_res).astype(int)
        cells_y = np.floor((points_y - map_origin.y) / map_res).astype(int)
        inside = (cells_x >= 0) & (cells_x < width) & (cells_y >= 0) & (cells_y < height)

        # Rays leaving the map are treated as hitting an obstacle
        occupied = np.ones(cells_x.shape, dtype=bool)
        occupied[inside] = self.occupancy[cells_y[inside], cells_x[inside]] >= FREE_THRESHOLD

        # Distance of the first occupied cell along each ray, maximum range if there is none
        first_hit = np.argmax(occupied, axis=2)
        return np.where(occupied.any(axis=2), steps[first_hit], range_max)

    def resample(self):
        """ Sample particles with replacement proportional to their weights and add random particles """
        # More random particles are added if the short term likelihood drops below the long term likelihood
        random_ratio = max(0.0, 1.0 - self.w_fast / max(self.w_slow, ZERO_REPLACEMENT))
        num_random = int(random_ratio * NUM_PARTICLES)

        indices = np.random.choice(NUM_PARTICLES, size=NUM_PARTICLES - num_random, p=self.weights)
        particles = self.particles[indices]
        if num_random > 0:
            particles = np.vstack((particles, self.sample_uniform_particles(num_random)))
            self.get_logger().info(f'\nAdded {num_random} random particles')

        self.particles = particles
        self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)

    def update_estimate(self):
        """ Represent the state by the weighted mean of the particles """
        mean_x = np.sum(self.weights * self.particles[:, 0])
        mean_y = np.sum(self.weights * self.particles[:, 1])
        mean_theta = math.atan2(np.sum(self.weights * np.sin(self.particles[:, 2])),
                                np.sum(self.weights * np.cos(self.particles[:, 2])))
        self.estimate = np.array([mean_x, mean_y, mean_theta])

        # Variance of the particles tells how certain the estimate is
        var_x = np.sum(self.weights * (self.particles[:, 0] - mean_x) ** 2)
        var_y = np.sum(self.weights * (self.particles[:, 1] - mean_y) ** 2)
        var_theta = np.sum(self.weights * normalize_angle(self.particles[:, 2] - mean_theta) ** 2)
        self.covariance = np.array([var_x, var_y, var_theta])

    def publish_estimate(self):
        stamp = self.get_clock().now().to_msg()

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.pose.pose = pose_from_array(self.estimate)
        msg.pose.covariance[0] = self.covariance[0]
        msg.pose.covariance[7] = self.covariance[1]
        msg.pose.covariance[35] = self.covariance[2]
        self.pose_pub.publish(msg)

        particles_msg = PoseArray()
        particles_msg.header.stamp = stamp
        particles_msg.header.frame_id = 'map'
        particles_msg.poses = [pose_from_array(p) for p in self.particles]
        self.particles_pub.publish(particles_msg)

    def publish_map_odom_transform(self):
        """ Publish map -> odom so that map pose = estimate for the odometry pose used in the last update """
        odom_x, odom_y, odom_theta = self.prev_odom_pose
        est_x, est_y, est_theta = self.estimate

        # map->odom = map->base_link * (odom->base_link)^-1
        theta = normalize_angle(est_theta - odom_theta)
        x = est_x - (math.cos(theta) * odom_x - math.sin(theta) * odom_y)
        y = est_y - (math.sin(theta) * odom_x + math.cos(theta) * odom_y)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = x
        t.transform.translation.y = y
        q = quaternion_from_euler(0.0, 0.0, theta)
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.tf_broadcaster.sendTransform(t)

    def control_loop(self):
        try:
            # Get transform of the laser scanner with respect to the base link
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                "base_laser_front_link",
                rclpy.time.Time()
            )
            quaternion = transform.transform.rotation
            laser_yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]
            self.laser_base_transform = (transform.transform.translation.x,
                                         transform.transform.translation.y,
                                         laser_yaw)
        except tf2_ros.TransformException:
            return

        if self.particles is None or self.grid is None or self.scan is None or self.odom_pose is None:
            return

        odom_pose = self.odom_pose.copy()
        if self.prev_odom_pose is None:
            # First iteration: no resampling before the robot has moved, otherwise the particles converge
            # to the best hypotheses of a single scan (which are often wrong in symmetric environments)
            self.prev_odom_pose = odom_pose
            self.update_estimate()
        else:
            delta_trans = math.sqrt((odom_pose[0] - self.prev_odom_pose[0]) ** 2 +
                                    (odom_pose[1] - self.prev_odom_pose[1]) ** 2)
            delta_rot = abs(normalize_angle(odom_pose[2] - self.prev_odom_pose[2]))

            # Particle filter: motion update, measurement update, resampling
            if delta_trans > MIN_TRAVEL_DISTANCE or delta_rot > MIN_TRAVEL_ROTATION:
                self.motion_update(odom_pose)
                self.prev_odom_pose = odom_pose
                self.measurement_update()
                self.update_estimate()
                self.resample()

                self.get_logger().info(f'\nEstimated pose: ({self.estimate[0]:.2f}, {self.estimate[1]:.2f}, '
                                       f'{self.estimate[2]:.2f}), '
                                       f'\nvariance: ({self.covariance[0]:.3f}, {self.covariance[1]:.3f}, '
                                       f'{self.covariance[2]:.3f})')

        self.publish_estimate()
        self.publish_map_odom_transform()


## Helper function to sample from a zero-mean normal distribution with the given variance (one sample per particle)
def sample_normal(variance):
    return np.random.normal(0.0, math.sqrt(variance), NUM_PARTICLES)

## Helper function to normalize angles to [-pi, pi]
def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

## Helper function to convert an (x, y, theta) array into a Pose msg
def pose_from_array(pose):
    q = quaternion_from_euler(0.0, 0.0, float(pose[2]))
    return Pose(position=Point(x=float(pose[0]), y=float(pose[1]), z=0.0),
                orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))

def main(args=None):
    rclpy.init(args=args)

    particle_filter = ParticleFilter()

    rclpy.spin(particle_filter)

    particle_filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
