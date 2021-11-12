#!/usr/bin/env python
import rospy
import tf2_ros
import numpy as np
import tf2_geometry_msgs
import tf.transformations
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped


# Global variables
listener = None
tf_buffer = None
link_to_optical_numpy = None


def odom_callback(msg):
    global listener
    global tf_buffer
    global link_to_optical_numpy

    br = tf.TransformBroadcaster()

    # Create 4x4 numpy optical to world transformation
    optical_to_world_numpy_rot = tf.transformations.quaternion_matrix([msg.pose.pose.orientation.x,
                                                                       msg.pose.pose.orientation.y,
                                                                       msg.pose.pose.orientation.z,
                                                                       msg.pose.pose.orientation.w])
    optical_to_world_numpy_trans = tf.transformations.translation_matrix([msg.pose.pose.position.x,
                                                                          msg.pose.pose.position.y,
                                                                          msg.pose.pose.position.z])
    optical_to_world_numpy = np.dot(optical_to_world_numpy_trans, optical_to_world_numpy_rot)

    # Compute link to world
    link_to_world = np.dot(optical_to_world_numpy, link_to_optical_numpy)

    # Publish transform
    br.sendTransform(tf.transformations.translation_from_matrix(link_to_world),
                     tf.transformations.quaternion_from_matrix(link_to_world),
                     rospy.Time.now(),
                     "d455_link",
                     "world")


def main():
    global listener
    global tf_buffer
    global link_to_optical_numpy

    # Create ROS node
    rospy.init_node('tf_publisher')

    # Transform listener
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            # Optical to link frame transformation
            link_to_optical = tf_buffer.lookup_transform("d455_color_optical_frame", "d455_link", rospy.Time(1))
            
            # Create 4x4 numpy optical to link transformation
            link_to_optical_numpy_rot = tf.transformations.quaternion_matrix([link_to_optical.transform.rotation.x,
                                                                              link_to_optical.transform.rotation.y,
                                                                              link_to_optical.transform.rotation.z,
                                                                              link_to_optical.transform.rotation.w])
            link_to_optical_numpy_trans = tf.transformations.translation_matrix([link_to_optical.transform.translation.x,
                                                                                 link_to_optical.transform.translation.y,
                                                                                 link_to_optical.transform.translation.z])
            link_to_optical_numpy = np.dot(link_to_optical_numpy_trans, link_to_optical_numpy_rot)

            print(link_to_optical_numpy)
            print(link_to_optical_numpy_rot)
            print(link_to_optical_numpy_trans)
            break

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

    # Subscribe to VIO odom topic
    rospy.Subscriber('/qxc_robot/system/pose', PoseWithCovarianceStamped, odom_callback)

    # Spin it
    rospy.spin()


if __name__ == "__main__":
    main()