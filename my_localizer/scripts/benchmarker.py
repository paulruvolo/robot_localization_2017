#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
from helper_functions import angle_diff, convert_pose_to_xy_and_theta
from std_msgs.msg import Header
import tf

class Benchmarker(object):
    def __init__(self):
        rospy.init_node('benchmarker')
        rospy.Subscriber('/map_pose', PoseStamped, self.process_pose)
        self.tf_listener = tf.TransformListener()

    def process_pose(self, msg):
        x, y, theta = convert_pose_to_xy_and_theta(msg.pose)
        self.tf_listener.waitForTransform('base_link', 'map', msg.header.stamp, rospy.Duration(0.1))
        base_link_pose = PoseStamped(header=Header(stamp=msg.header.stamp, frame_id='base_link'))
        map_pose = self.tf_listener.transformPose('map', base_link_pose)
        x_pred, y_pred, theta_pred = convert_pose_to_xy_and_theta(map_pose.pose)
        print "x %f,y %f,theta %f" % (x, y, theta)
        print "x_pred %f, y_pred %f,theta_pred %f" % (x_pred, y_pred, theta_pred)
        print "x_error %f, y_error %f, theta_error %f" % (abs(x - x_pred), abs(y - y_pred), abs(angle_diff(theta, theta_pred)))

    def run(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            r.sleep()

if __name__ == '__main__':
    node = Benchmarker()
    node.run()