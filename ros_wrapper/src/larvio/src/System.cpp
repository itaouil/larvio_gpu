//
// Created by xiaochen at 19-8-21.
// Managing the image processer and the estimator.
//

#include <chrono>

#include <sensor_msgs/PointCloud2.h>

#include <System.h>

#include <iostream>

#include <cv_bridge/cv_bridge.h>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace larvio {


System::System(ros::NodeHandle& n) : nh(n), publish(false) {}


System::~System() {
    // Clear buffer
    imu_msg_buffer.clear();
    img_msg_buffer.clear();
}


// Load parameters from launch file
bool System::loadParameters() {
    // Configuration file path.
    nh.getParam("config_file", config_file);

    // Imu and img synchronized threshold.
    double imu_rate;
    nh.param<double>("imu_rate", imu_rate, 200.0);
    imu_img_timeTh = 1/(2*imu_rate);

    return true;
}


// Subscribe image and imu msgs.
bool System::createRosIO() {
    // Subscribe imu msg.
    imu_sub = nh.subscribe("imu", 5000, &System::imuCallback, this);

    // Subscribe image msg.
    img_sub = nh.subscribe("cam0_image", 50, &System::imageCallback, this);

    // Advertise processed image msg.
     image_transport::ImageTransport it(nh);
     vis_img_pub = it.advertise("visualization_image", 1);

    // Advertise odometry msg.
    odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);

    // Advertise pose with covariance message
    pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose", 10);

    // Advertise point cloud msg.
     stable_feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
             "stable_feature_point_cloud", 1);
     active_feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
             "active_feature_point_cloud", 1);

    // Advertise path msg.
    path_pub = nh.advertise<nav_msgs::Path>("path", 10);

    nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
    nh.param<string>("child_frame_id", child_frame_id, "robot");

     stable_feature_msg_ptr.reset(
         new pcl::PointCloud<pcl::PointXYZ>());
     stable_feature_msg_ptr->header.frame_id = fixed_frame_id;
     stable_feature_msg_ptr->height = 1;

    return true;
}


// Initializing the system.
bool System::initialize() {
    // Load necessary parameters
    if (!loadParameters())
        return false;
    ROS_INFO("System: Finish loading ROS parameters...");

    // Set pointers of image processor and estimator.
    ImgProcesser.reset(new ImageProcessor(config_file));
    Estimator.reset(new LarVio(config_file));

    // Initialize image processor and estimator.
    if (!ImgProcesser->initialize()) {
        ROS_WARN("Image processor initialization failed!");
        return false;
    }
    if (!Estimator->initialize()) {
        ROS_WARN("Estimator initialization failed!");
        return false;
    }

    // Try subscribing msgs
    if (!createRosIO())
        return false;
    ROS_INFO("System Manager: Finish creating ROS IO...");

    return true;
}


// Push imu msg into the buffer.
void System::imuCallback(const sensor_msgs::ImuConstPtr& msg) {
    imu_msg_buffer.push_back(ImuData(msg->header.stamp.toSec(),
            msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z,
            msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z));
}


// Process the image and trigger the estimator.
void System::imageCallback(const sensor_msgs::ImageConstPtr& msg) {

    // Do nothing if no imu msg is received.
    if (imu_msg_buffer.empty())
        return;

    // test
    cv_bridge::CvImageConstPtr cvCPtr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    larvio::ImageDataPtr msgPtr(new ImgData);
    msgPtr->timeStampToSec = cvCPtr->header.stamp.toSec();
    msgPtr->image = cvCPtr->image.clone();
    std_msgs::Header header = cvCPtr->header;

    // Decide if use img msg in buffer.
    bool bUseBuff = false;
    if (!imu_msg_buffer.empty() ||
        (imu_msg_buffer.end()-1)->timeStampToSec-msgPtr->timeStampToSec<-imu_img_timeTh) {
        img_msg_buffer.push_back(msgPtr);
        header_buffer.push_back(header);
    }
    if (!img_msg_buffer.empty()) {
        if ((imu_msg_buffer.end()-1)->timeStampToSec-(*(img_msg_buffer.begin()))->timeStampToSec<-imu_img_timeTh)
            return;
        bUseBuff = true;
    }

    if (!bUseBuff) {
        MonoCameraMeasurementPtr features = new MonoCameraMeasurement;

        // Process image to get feature measurement.
        auto t1 = std::chrono::high_resolution_clock::now();
        bool bProcess = ImgProcesser->processImage(msgPtr, imu_msg_buffer, features);
        auto t2 = std::chrono::high_resolution_clock::now();

        ROS_INFO_STREAM("Image processing: " << ( t2 - t1 ).count()/1e9);

        // Filtering if get processed feature.
        bool bPubOdo = false;
        if (bProcess) {
            auto t3 = std::chrono::high_resolution_clock::now();
            bPubOdo = Estimator->processFeatures(features, imu_msg_buffer);
            auto t4 = std::chrono::high_resolution_clock::now();

            ROS_INFO_STREAM("Estimator processing: " << ( t4 - t3 ).count()/1e9);
        }

        // Publish msgs if necessary
        if (bProcess) {
            cv_bridge::CvImage _image(header, "bgr8", ImgProcesser->getVisualImg());
            vis_img_pub.publish(_image.toImageMsg());
        }
        if (bPubOdo) {
            publishVIO(header.stamp);
        }

        delete features;

        return;
    } else {
        // Loop for using all the img in the buffer that satisfy the condition.
        int counter = 0;
        for (int i = 0; i < img_msg_buffer.size(); ++i) {
            // Break the loop if imu data is not enough
            if ((imu_msg_buffer.end()-1)->timeStampToSec-img_msg_buffer[i]->timeStampToSec<-imu_img_timeTh)
                break;

            MonoCameraMeasurementPtr features = new MonoCameraMeasurement;

            // Process image to get feature measurement.
            auto t1 = std::chrono::high_resolution_clock::now();
            bool bProcess = ImgProcesser->processImage(img_msg_buffer[i], imu_msg_buffer, features);
            auto t2 = std::chrono::high_resolution_clock::now();

            ROS_INFO_STREAM("Image processing: " << ( t2 - t1 ).count()/1e9);

            // Filtering if get processed feature.
            bool bPubOdo = false;
            if (bProcess) {
                auto t3 = std::chrono::high_resolution_clock::now();
                bPubOdo = Estimator->processFeatures(features, imu_msg_buffer);
                auto t4 = std::chrono::high_resolution_clock::now();

                ROS_INFO_STREAM("Estimator processing: " << ( t4 - t3 ).count()/1e9);
            }

            // Publish msgs if necessary
            if (bProcess) {
                cv_bridge::CvImage _image(header_buffer[i], "bgr8", ImgProcesser->getVisualImg());
                vis_img_pub.publish(_image.toImageMsg());
            }
            if (bPubOdo) {
                publishVIO(header_buffer[i].stamp);
            }

            delete features;

            counter++;
        }
        img_msg_buffer.erase(img_msg_buffer.begin(), img_msg_buffer.begin()+counter);
        header_buffer.erase(header_buffer.begin(), header_buffer.begin()+counter);
    }
}


// Publish information of VIO, including odometry, path, points cloud and whatever needed.
void System::publishVIO(const ros::Time& time) {
    // construct odometry msg
    odom_msg.header.stamp = time;
    odom_msg.header.frame_id = fixed_frame_id;
    odom_msg.child_frame_id = child_frame_id;

    // Construct pose msg
    pose_msg.header.stamp = time;
    pose_msg.header.frame_id = fixed_frame_id;

    // Retrieve info
    Matrix3d P_body_vel = Estimator->getPvel();
    Eigen::Isometry3d T_b_w = Estimator->getTbw();
    Eigen::Vector3d body_velocity = Estimator->getVel();
    Matrix<double, 6, 6> P_body_pose = Estimator->getPpose();

    // Copy pose and velocity to odom message
    tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
    tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);

    // Copy pose to pose message
    tf::poseEigenToMsg(T_b_w, pose_msg.pose.pose);

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
        {
            odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);
            pose_msg.pose.covariance[6*i+j] = P_body_pose(i, j);
        }

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);

    // construct path msg
    path_msg.header.stamp = time;
    path_msg.header.frame_id = fixed_frame_id;
    geometry_msgs::PoseStamped curr_path;
    curr_path.header.stamp = time;
    curr_path.header.frame_id = fixed_frame_id;
    tf::poseEigenToMsg(T_b_w, curr_path.pose);
    path_msg.poses.push_back(curr_path);

    // construct point cloud msg
    // Publish the 3D positions of the features.
    // Including stable and active ones.
    // --Stable features
     std::map<larvio::FeatureIDType,Eigen::Vector3d> StableMapPoints;
     Estimator->getStableMapPointPositions(StableMapPoints);
     for (const auto& item : StableMapPoints) {
         const auto& feature_position = item.second;
         stable_feature_msg_ptr->points.push_back(pcl::PointXYZ(
                 feature_position(0), feature_position(1), feature_position(2)));
     }
     stable_feature_msg_ptr->width = stable_feature_msg_ptr->points.size();

    // --Active features
     active_feature_msg_ptr.reset(
         new pcl::PointCloud<pcl::PointXYZ>());
     active_feature_msg_ptr->header.frame_id = fixed_frame_id;
     active_feature_msg_ptr->height = 1;
     std::map<larvio::FeatureIDType,Eigen::Vector3d> ActiveMapPoints;
     Estimator->getActiveeMapPointPositions(ActiveMapPoints);
     for (const auto& item : ActiveMapPoints) {
         const auto& feature_position = item.second;
         active_feature_msg_ptr->points.push_back(pcl::PointXYZ(
                 feature_position(0), feature_position(1), feature_position(2)));
     }
     active_feature_msg_ptr->width = active_feature_msg_ptr->points.size();

    odom_pub.publish(odom_msg);
    pose_pub.publish(pose_msg);
    path_pub.publish(path_msg);
    stable_feature_pub.publish(stable_feature_msg_ptr);
    active_feature_pub.publish(active_feature_msg_ptr);
}

} // end namespace larvio
