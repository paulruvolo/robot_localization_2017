#!/usr/bin/env python

""" This is the starter code for the robot localization project """

import rospy

from dynamic_reconfigure.server import Server
from my_localizer.cfg import PfWeek11Config

from std_msgs.msg import Header, String
from sensor_msgs.msg import LaserScan, PointCloud
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion
from nav_msgs.srv import GetMap
from copy import deepcopy

import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
from random import gauss

import math
import time

import numpy as np
from numpy.random import random_sample
from sklearn.neighbors import NearestNeighbors
from occupancy_field import OccupancyField

from helper_functions import (convert_pose_inverse_transform,
                              convert_translation_rotation_to_pose,
                              convert_pose_to_xy_and_theta,
                              angle_diff)

class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self,x=0.0,y=0.0,theta=0.0,w=1.0):
        """ Construct a new Particle
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized """ 
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        orientation_tuple = tf.transformations.quaternion_from_euler(0,0,self.theta)
        return Pose(position=Point(x=self.x,y=self.y,z=0), orientation=Quaternion(x=orientation_tuple[0], y=orientation_tuple[1], z=orientation_tuple[2], w=orientation_tuple[3]))

    # TODO: define additional helper functions if needed

class ParticleFilter:
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            initialized: a Boolean flag to communicate to other class methods that initializaiton is complete
            base_frame: the name of the robot base coordinate frame (should be "base_link" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            laser_max_distance: the maximum distance to an obstacle we should use in a likelihood calculation
            pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            laser_subscriber: listens for new scan data on topic self.scan_topic
            tf_listener: listener for coordinate transforms
            tf_broadcaster: broadcaster for coordinate transforms
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            map: the map we will be localizing ourselves in.  The map should be of type nav_msgs/OccupancyGrid
    """
    def __init__(self):
        self.initialized = False        # make sure we don't perform updates before everything is setup
        rospy.init_node('pf')           # tell roscore that we are creating a new node named "pf"

        self.base_frame = "base_link"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from 
        # the number of particles to use
        self.n_particles = rospy.get_param('~n_particles', 300)
        self.d_thresh = 0.2             # the amount of linear movement before performing an update
        self.a_thresh = math.pi/6       # the amount of angular movement before performing an update

        self.laser_max_distance = 2.0   # maximum penalty to assess in the likelihood field model

        # TODO: define additional constants if needed
        self.alpha1 = 0.2
        self.alpha2 = 0.2
        self.alpha3 = 0.2
        self.alpha4 = 0.2
        self.z_hit = rospy.get_param('~z_hit', 0.5)
        self.z_rand = 1 - self.z_hit
        self.sigma_hit = 0.1
        # Setup pubs and subs

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        rospy.Subscriber("initialpose", PoseWithCovarianceStamped, self.update_initial_pose)

        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = rospy.Publisher("particlecloud", PoseArray, queue_size=10)

        # laser_subscriber listens for data from the lidar
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_received)

        # enable listening for and broadcasting coordinate transforms
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

        srv = Server(PfWeek11Config, self.config_callback)

        self.particle_cloud = []

        self.last_pose_guess = None

        # change use_projected_stable_scan to True to use point clouds instead of laser scans
        self.use_projected_stable_scan = False
        self.last_projected_stable_scan = None
        if self.use_projected_stable_scan:
            # subscriber to the odom point cloud
            rospy.Subscriber("projected_stable_scan", PointCloud, self.projected_scan_received)

        self.current_odom_xy_theta = []
        self.new_odom_xy_theta = []

        rospy.wait_for_service("static_map")
        static_map = rospy.ServiceProxy("static_map", GetMap)
        try:
            map = static_map().map
        except:
            print "error receiving map"

        self.occupancy_field = OccupancyField(map)
        print "Occupancy field initialized!"
        self.initialized = True


    def config_callback(self, config, level):
        print "config.n_particles", config.n_particles
        print "config.z_hit", config.z_hit 
        self.z_hit = config.z_hit
        self.z_rand = 1 - self.z_hit
        self.n_particles = config.n_particles
        return config

    def update_robot_pose(self):
        """ Update the estimate of the robot's pose given the updated particles.
            There are two logical methods for this:
                (1): compute the mean pose
                (2): compute the most likely pose (i.e. the mode of the distribution)
        """
        # first make sure that the particle weights are normalized
        self.normalize_particles()
        mean_x = 0.0
        mean_y = 0.0
        mean_theta = 0.0
        theta_list = []
        weighted_orientation_vec = np.zeros((2,1))
        for p in self.particle_cloud:
            mean_x += p.x*p.w
            mean_y += p.y*p.w
            weighted_orientation_vec[0] += p.w*math.cos(p.theta)
            weighted_orientation_vec[1] += p.w*math.sin(p.theta)
        mean_theta = math.atan2(weighted_orientation_vec[1],weighted_orientation_vec[0])
        self.robot_pose = Particle(x=mean_x,y=mean_y,theta=mean_theta).as_pose()


    def projected_scan_received(self, msg):
        self.last_projected_stable_scan = msg

    def reinitialize(self):
        if self.last_pose_guess:
            # clear out the tf_listener as time has gone backwards
            self.tf_listener.clear()
            # set the stamp to use the latest transform when fixing map to odom
            self.last_pose_guess.header.stamp = rospy.Time(0)
            print "reinitializing with last message"
            self.update_initial_pose(self.last_pose_guess)

    def update_particles_with_odom(self, msg):
        """ Implement a simple version of this (Level 1) or a more complex one (Level 2) """
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0], new_odom_xy_theta[1] - self.current_odom_xy_theta[1], new_odom_xy_theta[2] - self.current_odom_xy_theta[2])
            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return
        # Implement sample_motion_odometry (Prob Rob p 136)
        # Avoid computing a bearing from two poses that are extremely near each
        # other (happens on in-place rotation).
        delta_trans = math.sqrt(delta[0]*delta[0] + delta[1]*delta[1])
        if delta_trans < 0.01:
            delta_rot1 = 0.0
        else:
            delta_rot1 = angle_diff(math.atan2(delta[1], delta[0]), old_odom_xy_theta[2])

        delta_rot2 = angle_diff(delta[2], delta_rot1)
        # We want to treat backward and forward motion symmetrically for the
        # noise model to be applied below.  The standard model seems to assume
        # forward motion.
        delta_rot1_noise = min(math.fabs(angle_diff(delta_rot1, 0.0)), math.fabs(angle_diff(delta_rot1, math.pi)));
        delta_rot2_noise = min(math.fabs(angle_diff(delta_rot2, 0.0)), math.fabs(angle_diff(delta_rot2, math.pi)));

        for sample in self.particle_cloud:
            # Sample pose differences
            delta_rot1_hat = angle_diff(delta_rot1, gauss(0, self.alpha1*delta_rot1_noise*delta_rot1_noise + self.alpha2*delta_trans*delta_trans))
            delta_trans_hat = delta_trans - gauss(0, self.alpha3*delta_trans*delta_trans + self.alpha4*delta_rot1_noise*delta_rot1_noise + self.alpha4*delta_rot2_noise*delta_rot2_noise)
            delta_rot2_hat = angle_diff(delta_rot2, gauss(0, self.alpha1*delta_rot2_noise*delta_rot2_noise + self.alpha2*delta_trans*delta_trans))

            # Apply sampled update to particle pose
            sample.x += delta_trans_hat * math.cos(sample.theta + delta_rot1_hat)
            sample.y += delta_trans_hat * math.sin(sample.theta + delta_rot1_hat)
            sample.theta += delta_rot1_hat + delta_rot2_hat

    def map_calc_range(self,x,y,theta):
        """ Difficulty Level 3: implement a ray tracing likelihood model... Let me know if you are interested """
        # TODO: nothing unless you want to try this alternate likelihood model
        pass

    def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample.
        """
        # make sure the distribution is normalized
        self.normalize_particles()
        self.particle_cloud = ParticleFilter.draw_random_sample(self.particle_cloud,
                                                                [p.w for p in self.particle_cloud],
                                                                self.n_particles)

    def update_particles_with_laser(self, msg):
        """ Updates the particle weights in response to the scan contained in the msg """
        laser_xy_theta = convert_pose_to_xy_and_theta(self.laser_pose.pose)
        for p in self.particle_cloud:
            # Pre-compute a couple of things
            z_hit_denom = 2*self.sigma_hit**2
            z_rand_mult = 1.0/msg.range_max

            # This assumes quite a bit about the weights beforehand (TODO: could base this on p.w)
            new_prob = 1.0  # more agressive DEBUG, was 1.0
            for i in range(0,len(msg.ranges),6):
                pz = 1.0

                obs_range = msg.ranges[i]
                obs_bearing = i*msg.angle_increment+msg.angle_min + math.pi # need to add pi since the laser frame is flipped

                if math.isnan(obs_range):
                    continue

                if obs_range >= msg.range_max:
                    continue

                # compute the endpoint of the laser
                end_x = p.x + obs_range*math.cos(p.theta+obs_bearing)
                end_y = p.y + obs_range*math.sin(p.theta+obs_bearing)
                z = self.occupancy_field.get_closest_obstacle_distance(end_x,end_y)
                if math.isnan(z):
                    z = self.laser_max_distance

                pz += self.z_hit * math.exp(-(z * z) / z_hit_denom) / (math.sqrt(2*math.pi)*self.sigma_hit)
                pz += self.z_rand * z_rand_mult

                new_prob += pz**3
            p.w = new_prob

    @staticmethod
    def draw_random_sample(choices, probabilities, n):
        """ Return a random sample of n elements from the set choices with the specified probabilities
            choices: the values to sample from represented as a list
            probabilities: the probability of selecting each element in choices represented as a list
            n: the number of samples
        """
        values = np.array(range(len(choices)))
        probs = np.array(probabilities)
        bins = np.add.accumulate(probs)
        inds = values[np.digitize(random_sample(n), bins)]
        samples = []
        for i in inds:
            samples.append(deepcopy(choices[int(i)]))
        return samples

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        self.last_pose_guess = msg
        xy_theta = convert_pose_to_xy_and_theta(msg.pose.pose)
        print "Initializing with", xy_theta
        self.initialize_particle_cloud(xy_theta)
        self.fix_map_to_odom_transform(msg)

    def initialize_particle_cloud(self, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is ommitted, the odometry will be used """
        if xy_theta == None:
            xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)
        print "INITIALIZING the CLOUD!!", xy_theta
        self.particle_cloud = []
        for i in range(self.n_particles):
            self.particle_cloud.append(Particle(x=xy_theta[0]+gauss(0,.25),y=xy_theta[1]+gauss(0,.25),theta=xy_theta[2]+gauss(0,.25)))
        self.normalize_particles()
        self.update_robot_pose()

    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """
        z = 0.0
        for p in self.particle_cloud:
            z += p.w
        for i in range(len(self.particle_cloud)):
            self.particle_cloud[i].w /= z

    def publish_particles(self, msg):
        particles_conv = []
        for p in self.particle_cloud:
            particles_conv.append(p.as_pose())
        # actually send the message so that we can view it in rviz
        self.particle_pub.publish(PoseArray(header=Header(stamp=rospy.Time.now(),
                                            frame_id=self.map_frame),
                                  poses=particles_conv))

    def scan_received(self, msg):
        """ This is the default logic for what to do when processing scan data.
            Feel free to modify this, however, I hope it will provide a good
            guide.  The input msg is an object of type sensor_msgs/LaserScan """
        if not(self.initialized):
            # wait for initialization to complete
            return

        if not(self.tf_listener.canTransform(self.base_frame,msg.header.frame_id,msg.header.stamp)):
            # need to know how to transform the laser to the base frame
            # this will be given by either Gazebo or neato_node
            return

        if not(self.tf_listener.canTransform(self.base_frame,self.odom_frame,msg.header.stamp)):
            # need to know how to transform between base and odometric frames
            # this will eventually be published by either Gazebo or neato_node
            return

        # calculate pose of laser relative ot the robot base
        p = PoseStamped(header=Header(stamp=rospy.Time(0),
                                      frame_id=msg.header.frame_id))
        self.laser_pose = self.tf_listener.transformPose(self.base_frame,p)

        # find out where the robot thinks it is based on its odometry
        p = PoseStamped(header=Header(stamp=msg.header.stamp,
                                      frame_id=self.base_frame),
                        pose=Pose())
        self.odom_pose = self.tf_listener.transformPose(self.odom_frame, p)
        # store the the odometry pose in a more convenient format (x,y,theta)
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)

        if not(self.particle_cloud):
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud()
            # cache the last odometric pose so we can only update our particle filter if we move more than self.d_thresh or self.a_thresh
            self.current_odom_xy_theta = new_odom_xy_theta
            # update our map to odom transform now that the particles are initialized
            self.fix_map_to_odom_transform(msg)
        elif (len(self.current_odom_xy_theta) < 3 or 
              math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or
              math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or
              math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh):
            # we have moved far enough to do an update!
            self.update_particles_with_odom(msg)    # update based on odometry
            if self.last_projected_stable_scan:
                last_projected_scan_timeshift = deepcopy(self.last_projected_stable_scan)
                last_projected_scan_timeshift.header.stamp = msg.header.stamp
                self.scan_in_base_link = self.tf_listener.transformPointCloud("base_link", last_projected_scan_timeshift)

            self.update_particles_with_laser(msg)   # update based on laser scan
            self.update_robot_pose()                # update robot's pose
            self.resample_particles()               # resample particles to focus on areas of high density
            self.fix_map_to_odom_transform(msg)     # update map to odom transform now that we have new particles
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg)

    def fix_map_to_odom_transform(self, msg):
        """ This method constantly updates the offset of the map and 
            odometry coordinate systems based on the latest results from
            the localizer
            TODO: if you want to learn a lot about tf, reimplement this... I can provide
                  you with some hints as to what is going on here. """
        (translation, rotation) = convert_pose_inverse_transform(self.robot_pose)
        p = PoseStamped(pose=convert_translation_rotation_to_pose(translation,rotation),
                        header=Header(stamp=msg.header.stamp,frame_id=self.base_frame))
        self.tf_listener.waitForTransform(self.base_frame, self.odom_frame, msg.header.stamp, rospy.Duration(1.0))
        self.odom_to_map = self.tf_listener.transformPose(self.odom_frame, p)
        (self.translation, self.rotation) = convert_pose_inverse_transform(self.odom_to_map.pose)

    def broadcast_last_transform(self):
        """ Make sure that we are always broadcasting the last map
            to odom transformation.  This is necessary so things like
            move_base can work properly. """
        if not(hasattr(self,'translation') and hasattr(self,'rotation')):
            return
        self.tf_broadcaster.sendTransform(self.translation,
                                          self.rotation,
                                          rospy.get_rostime(),
                                          self.odom_frame,
                                          self.map_frame)

if __name__ == '__main__':
    n = ParticleFilter()
    r = rospy.Rate(5)

    while not(rospy.is_shutdown()):
        # in the main loop all we do is continuously broadcast the latest map to odom transform
        n.broadcast_last_transform()
        try:
            r.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException as inst:
            print "Error:", inst
            n.reinitialize()