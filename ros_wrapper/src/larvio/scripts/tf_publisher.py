#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped


# Global variables
listener = None
tf_buffer = None
d455_optical_frame_to_d455_link = None


def odom_callback(msg):
    global listener
    global tf_buffer
    global d455_optical_frame_to_d455_link

    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()

    # Odom pose in color optical frame
    odom_optical_frame = tf2_geometry_msgs.PoseStamped()
    odom_optical_frame.header.stamp = rospy.Time.now()
    odom_optical_frame.header.frame_id = "d455_color_optical_frame"
    odom_optical_frame.pose.position.x = msg.pose.pose.position.x
    odom_optical_frame.pose.position.y = msg.pose.pose.position.y
    odom_optical_frame.pose.position.z = msg.pose.pose.position.z
    odom_optical_frame.pose.orientation.x = msg.pose.pose.orientation.x
    odom_optical_frame.pose.orientation.y = msg.pose.pose.orientation.y
    odom_optical_frame.pose.orientation.z = msg.pose.pose.orientation.z
    odom_optical_frame.pose.orientation.w = msg.pose.pose.orientation.w

    odom_d455_link = None
    try:
        odom_d455_link = tf_buffer.transform(odom_optical_frame, "d455_link", rospy.Duration(0))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Could not convert odom from optical frame to d455 link")

    # t.header.stamp = rospy.Time.now()
    # t.header.frame_id = "world"
    # t.child_frame_id = "d455_link"
    # t.transform.translation.x = odom_d455_link.pose.position.x
    # t.transform.translation.y = odom_d455_link.pose.position.y
    # t.transform.translation.z = odom_d455_link.pose.position.z
    # t.transform.rotation.x = odom_d455_link.pose.orientation.x
    # t.transform.rotation.y = odom_d455_link.pose.orientation.y
    # t.transform.rotation.z = odom_d455_link.pose.orientation.z
    # t.transform.rotation.w = odom_d455_link.pose.orientation.w

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = "d455_link"
    t.transform.translation.x = msg.pose.pose.position.x
    t.transform.translation.y = msg.pose.pose.position.y
    t.transform.translation.z = msg.pose.pose.position.z
    t.transform.rotation.x = msg.pose.pose.orientation.x
    t.transform.rotation.y = msg.pose.pose.orientation.y
    t.transform.rotation.z = msg.pose.pose.orientation.z
    t.transform.rotation.w = msg.pose.pose.orientation.w

    br.sendTransform(t)


def main():
    global listener
    global tf_buffer
    global d455_optical_frame_to_d455_link

    # Create ROS node
    rospy.init_node('tf_publisher')

    # Transform listener
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            d455_optical_frame_to_d455_link = tf_buffer.lookup_transform("d455_color_optical_frame", "d455_link", rospy.Time(1))
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
