/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

// The original file belongs to MSCKF_VIO (https://github.com/KumarRobotics/msckf_vio/)
// Tremendous changes have been made to use it in LARVIO


#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

// LARVIO
#include "sensors/ImuData.hpp"
#include <ORB/ORBDescriptor.h>
#include <larvio/feature_msg.h>
#include "sensors/ImageData.hpp"

// C++
#include <map>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <boost/shared_ptr.hpp>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

// Vilib
#include "vilib/config.h"
#include "vilib/statistics.h"
#include "vilib/cuda_common.h"
#include "vilib/storage/pyramid_pool.h"
#include "vilib/feature_detection/fast/fast_gpu.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/feature_tracker/feature_tracker_gpu.h"
#include "vilib/feature_detection/harris/harris_gpu.h"
#include "vilib/feature_tracker/feature_tracker_base.h"
#include "vilib/feature_tracker/feature_tracker_options.h"

namespace larvio {

/*
 * @brief ImageProcessor Detects and tracks features
 *    in image sequences.
 */
class ImageProcessor {
public:
  // Constructor
  ImageProcessor(std::string& config_file_);
  // Disable copy and assign constructors.
  ImageProcessor(const ImageProcessor&) = delete;
  ImageProcessor operator=(const ImageProcessor&) = delete;

  // Destructor
  ~ImageProcessor();

  // Initialize the object.
  bool initialize();

  /*
   * @brief processImage
   *    Processing function for the monocular images.
   * @param image msg.
   * @param imu msg buffer.
   * @return true if have feature msg.
   */
  bool processImage(
        const ImageDataPtr& msg,
        const std::vector<ImuData>& imu_msg_buffer,
        MonoCameraMeasurementPtr features);

  // Get publish image.
  cv::Mat getVisualImg() {
      return visual_img;
  }

  typedef boost::shared_ptr<ImageProcessor> Ptr;
  typedef boost::shared_ptr<const ImageProcessor> ConstPtr;

private:

  /*
   * @brief ProcessorConfig Configuration parameters for
   *    feature detection and tracking.
   */
  struct ProcessorConfig {
    int pyramid_levels;
    int fast_threshold;

    int max_distance;
    bool flag_equalize;
    int max_features_num;
    bool publish_features;

    int img_rate;
    int pub_frequency;
  };

  /*
   * @brief FeatureIDType An alias for unsigned long long int.
   */
  typedef unsigned long long int FeatureIDType;

  /*
   * @brief MapType An alias for map<FeatureIDType, cv::Point2f>
   */
  typedef std::map<FeatureIDType, cv::Point2f> MapType;

  /*
   * @brief loadParameters
   *    Load parameters from the parameter server.
   */
  bool loadParameters();

  /*
   * @brief initializeVilib
   *    Initialize the feature detector and tracker
   *    using the vilib libray to process the detection
   *    and tracking on the GPU.
   */
  bool initializeVilib();

  /*
   * @brief clearDeadFeatures
   *    Clear dead features and relative information,
   *    where a dead feature is one that was not tracked
   *    in the next frame.
   */
  void clearDeadFeatures();

    /*
   * @brief applyClahe
   *    Apply histogram equalization
   */
  void applyClahe();

    /*
   * @brief trackImage
   *    Perform tracking on GPU using a detection
   *    algorithm also no the GPU and the current image
   * @param imu msg buffer
   */
    void trackImage(
            const cv::Mat &img,
            std::map<FeatureIDType, cv::Point2f> &points);

  /*
   * @brief initializeFirstFrame
   *    Initialize the image processing sequence, which is
   *    bascially detect new features on the first image.
   */
  bool initializeFirstFrame();

  /*
   * @brief initializeFirstFeatures
   *    Initialize the image processing sequence, which is
   *    bascially detect new features on the first set of
   *    stereo images.
   * @param imu msg buffer
   */
  bool trackFirstFeatures(
          const std::vector<ImuData>& imu_msg_buffer);

  /*
   * @brief trackFeatures
   *    Tracker features on the newly received image.
   */
  void trackFeatures();

  /*
   * @brief publish
   *    Publish the features on the current image including
   *    both the tracked and newly detected ones.
   */
  void publish();

  /*
   * @brief undistortPoints Undistort points based on camera
   *    calibration intrinsics and distort model.
   * @param pts_in: input distorted points.
   * @param intrinsics: intrinsics of the camera.
   * @param distortion_model: distortion model of the camera.
   * @param distortion_coeffs: distortion coefficients.
   * @param rectification_matrix: matrix to rectify undistorted points.
   * @param new_intrinsics: intrinsics on new camera for
   *    undistorted points to projected into.
   * @return pts_out: undistorted points.
   */
  void undistortPoints(
      const std::vector<cv::Point2f>& pts_in,
      const cv::Vec4d& intrinsics,
      const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs,
      std::vector<cv::Point2f>& pts_out,
      const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
      const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));

  /*
   * @brief removeUnmarkedElements Remove the unmarked elements
   *    within a vector.
   * @param raw_vec: vector with outliers.
   * @param markers: 0 will represent a outlier, 1 will be an inlier.
   * @return refined_vec: a vector without outliers.
   *
   * Note that the order of the inliers in the raw_vec is perserved
   * in the refined_vec.
   */
  template <typename T>
  void removeUnmarkedElements(
      const std::vector<T>& raw_vec,
      const std::vector<unsigned char>& markers,
      std::vector<T>& refined_vec) {
    if (raw_vec.size() != markers.size()) {
      for (int i = 0; i < raw_vec.size(); ++i)
        refined_vec.push_back(raw_vec[i]);
      return;
    }
    for (int i = 0; i < markers.size(); ++i) {
      if (markers[i] == 0) continue;
      refined_vec.push_back(raw_vec[i]);
    }
  }

  /*
   * @brief rescalePoints Rescale image coordinate of pixels to gain
   *    numerical stability.
   * @param pts1: an array of image coordinate of some pixels,
   *    they will be rescaled.
   * @param pts2: corresponding image coordinate of some pixels
   *    in another image as pts2, they will be rescaled.
   * @return scaling_factor: scaling factor of the rescaling process.
   */
  void rescalePoints(
      std::vector<cv::Point2f>& pts1,
      std::vector<cv::Point2f>& pts2,
      float& scaling_factor);

  /*
   * @brief getFeatureMsg Get processed feature msg.
   * @return pointer for processed features
   */
  void getFeatureMsg(MonoCameraMeasurementPtr features);

  // Enum type for image state.
  enum eImageState {
      FIRST_IMAGE = 1,
      SECOND_IMAGE = 2,
      OTHER_IMAGES = 3
  };

  // Indicate if this is the first or second image message.
  eImageState image_state;

  // ID for the next new feature.
  FeatureIDType next_feature_id;

  // Feature detector
  ProcessorConfig processor_config;

  // Camera calibration parameters
  cv::Vec2i cam_resolution;
  cv::Vec4d cam_intrinsics;
  cv::Vec4d cam_distortion_coeffs;
  std::string cam_distortion_model;

  // Take a vector from cam frame to the IMU frame.
  cv::Vec3d t_cam_imu;
  cv::Matx33d R_cam_imu;

  // Take a vector from prev cam frame to curr cam frame
  cv::Matx33f R_Prev2Curr;

  // Previous and current images
  ImageDataPtr prev_img_ptr;
  ImageDataPtr curr_img_ptr;

  // Number of features after each outlier removal step.
  int after_ransac;
  int after_tracking;
  int before_tracking;

  // Config file path
  std::string config_file;

  // Image for visualization
  cv::Mat visual_img;

  // Points for tracking, added by QXC
  std::vector<int> pts_lifetime_;
  std::vector<cv::Point2f> prev_pts_;
  std::vector<cv::Point2f> curr_pts_;
  std::vector<cv::Point2f> init_pts_;
  std::vector<FeatureIDType> pts_ids_;

  // Time of last published image
  double last_pub_time;
  double curr_img_time;
  double prev_img_time;

  // Publish counter
  long pub_counter;

  // debug log:
  std::string output_dir;

  // ORB descriptor pointer, added by QXC
  std::vector<cv::Mat> vOrbDescriptors;
  boost::shared_ptr<ORBdescriptor> prevORBDescriptor_ptr;
  boost::shared_ptr<ORBdescriptor> currORBDescriptor_ptr;

  // flag for first useful image msg
  bool bFirstImg;

  // GPU detector and tracker
  std::shared_ptr<vilib::DetectorBaseGPU> detector_gpu;
  std::shared_ptr<vilib::FeatureTrackerBase> tracker_gpu;

  // Tracked points on GPU
  std::map<FeatureIDType, cv::Point2f> current_tracked_points_map;
  std::map<FeatureIDType, cv::Point2f> previous_tracked_points_map;

  // Lifetime, initial point, and descriptors for tracked points
  std::map<FeatureIDType, int> tracked_points_lifetime_map;
  std::map<FeatureIDType, cv::Mat> tracked_points_descriptor_map;
  std::map<FeatureIDType, cv::Point2f> tracked_points_initial_map;
};

typedef ImageProcessor::Ptr ImageProcessorPtr;
typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;

} // end namespace larvio

#endif
