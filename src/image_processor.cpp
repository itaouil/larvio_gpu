/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

// The original file belongs to MSCKF_VIO (https://github.com/KumarRobotics/msckf_vio/)
// Tremendous changes have been done to use it in LARVIO

// C++
#include <set>
#include <iostream>
#include <algorithm>

// Eigen
#include <Eigen/Dense>

// LARVIO
#include <larvio/math_utils.hpp>
#include <larvio/image_processor.h>

// OpenCV
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace std;
using namespace vilib;
using namespace Eigen;

namespace larvio {

ImageProcessor::ImageProcessor(std::string& config_file_) :
        config_file(config_file_) {
    image_state = FIRST_IMAGE;
    next_feature_id = 0;
}


ImageProcessor::~ImageProcessor() {
    destroyAllWindows();
}


bool ImageProcessor::loadParameters() {

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cout << "config_file error: cannot open " << config_file << endl;
        return false;
    }

    // Read config parameters
    processor_config.img_rate = fsSettings["img_rate"];
    processor_config.patch_size = fsSettings["patch_size"];
    processor_config.min_distance = fsSettings["min_distance"];
    processor_config.pub_frequency = fsSettings["pub_frequency"];
    processor_config.max_iteration = fsSettings["max_iteration"];
    processor_config.fast_threshold = fsSettings["fast_threshold"];
    processor_config.pyramid_levels = fsSettings["pyramid_levels"];
    processor_config.track_precision = fsSettings["track_precision"];
    processor_config.ransac_threshold = fsSettings["ransac_threshold"];
    processor_config.max_features_num = fsSettings["max_features_num"];
    processor_config.flag_equalize = static_cast<int>(fsSettings["flag_equalize"]) != 0;

    // Output files directory
    fsSettings["output_dir"] >> output_dir;

    /*
     * Camera calibration parameters
     */

    // Distortion model
    fsSettings["distortion_model"] >> cam_distortion_model;

    // Resolution of camera
    cam_resolution[0] = fsSettings["resolution_width"];
    cam_resolution[1] = fsSettings["resolution_height"];

    // Camera calibration instrinsics
    cv::FileNode n_instrin = fsSettings["intrinsics"];
    cam_intrinsics[0] = static_cast<double>(n_instrin["fx"]);
    cam_intrinsics[1] = static_cast<double>(n_instrin["fy"]);
    cam_intrinsics[2] = static_cast<double>(n_instrin["cx"]);
    cam_intrinsics[3] = static_cast<double>(n_instrin["cy"]);

    // Distortion coefficient
    cv::FileNode n_distort = fsSettings["distortion_coeffs"];
    cam_distortion_coeffs[0] = static_cast<double>(n_distort["k1"]);
    cam_distortion_coeffs[1] = static_cast<double>(n_distort["k2"]);
    cam_distortion_coeffs[2] = static_cast<double>(n_distort["p1"]);
    cam_distortion_coeffs[3] = static_cast<double>(n_distort["p2"]);
    
    // Extrinsic between camera and IMU
    cv::Mat T_imu_cam;
    fsSettings["T_cam_imu"] >> T_imu_cam;
    cv::Matx33d R_imu_cam(T_imu_cam(cv::Rect(0,0,3,3)));      
    cv::Vec3d t_imu_cam = T_imu_cam(cv::Rect(3,0,1,3));
    R_cam_imu = R_imu_cam.t();
    t_cam_imu = -R_imu_cam.t() * t_imu_cam;

    return true;
}


bool ImageProcessor::initialize() {
    if (!loadParameters()) return false;

    // Initialize frontend
    initializeVilib();

    // Initialize publish counter
    pub_counter = 0;

    // Initialize flag for first useful img msg
    bFirstImg = false;

    return true;
}


// Process current img msg and return the feature msg.
bool ImageProcessor::processImage(const ImageDataPtr& msg,
        const std::vector<ImuData>& imu_msg_buffer, MonoCameraMeasurementPtr features) {

    // images are not utilized until receiving imu msgs ahead
    if (!bFirstImg) {
        if ((imu_msg_buffer.begin() != imu_msg_buffer.end()) && 
            (imu_msg_buffer.begin()->timeStampToSec-msg->timeStampToSec <= 0.0)) {
            bFirstImg = true;
            printf("Images from now on will be utilized...\n");
        }
        else
            return false;
    }

    curr_img_ptr = msg;

    // Build the image pyramids once since they're used at multiple places
    clahe();

    // Initialize ORBDescriptor pointer
    currORBDescriptor_ptr.reset(new ORBdescriptor(curr_img_ptr->image, 2, processor_config.pyramid_levels));

    // Get current image time
    curr_img_time = curr_img_ptr->timeStampToSec;

    // Flag to return;
    bool haveFeatures = false;

    // Detect features in the first frame.
    if ( FIRST_IMAGE==image_state ) {
        if (initializeFirstFrame())
            image_state = SECOND_IMAGE;
    } else if ( SECOND_IMAGE==image_state ) {
        if ( !trackFirstFeatures(imu_msg_buffer) ) {
            image_state = FIRST_IMAGE;
        } else {
            // frequency control
            if ( curr_img_time-last_pub_time >= 0.9*(1.0/processor_config.pub_frequency) ) {
                // Det processed feature
                getFeatureMsg(features);

                // Publishing msgs
                publish();

                haveFeatures = true;
            }

            image_state = OTHER_IMAGES;
        }
    } else if ( OTHER_IMAGES==image_state ) {
        // Integrate gyro data to get a guess of rotation between current and previous image
//        integrateImuData(R_Prev2Curr, imu_msg_buffer);

        // Tracking features
        trackFeatures();

        // frequency control
//        printf("Time %f\n", curr_img_time-last_pub_time);
//        printf("RHS %f\n", 0.9*(1.0/processor_config.pub_frequency));

        if (curr_img_time-last_pub_time >= 0.9*(1.0/processor_config.pub_frequency)) {
            // Det processed feature
            getFeatureMsg(features);

            // Publishing msgs
            publish();

            haveFeatures = true;
        }
    }

    // Update the previous image and previous features.
    prev_img_ptr = curr_img_ptr;
    prevORBDescriptor_ptr = currORBDescriptor_ptr;

    // Initialize the current features to empty vectors.
//    swap(prev_pts_,curr_pts_);
//    vector<Point2f>().swap(curr_pts_);

    prev_img_time = curr_img_time;

    return haveFeatures;
}


void ImageProcessor::integrateImuData(Matx33f& cam_R_p2c,
        const std::vector<ImuData>& imu_msg_buffer) {
    // Find the start and the end limit within the imu msg buffer.
    auto begin_iter = imu_msg_buffer.begin();
    while (begin_iter != imu_msg_buffer.end()) {
    if (begin_iter->timeStampToSec-
            prev_img_ptr->timeStampToSec < -0.0049)
        ++begin_iter;
    else
        break;
    }

    auto end_iter = begin_iter;
    while (end_iter != imu_msg_buffer.end()) {
    if (end_iter->timeStampToSec-
            curr_img_ptr->timeStampToSec < 0.0049)
        ++end_iter;
    else
        break;
    }

    // Compute the mean angular velocity in the IMU frame.
    Vec3f mean_ang_vel(0.0, 0.0, 0.0);
    for (auto iter = begin_iter; iter < end_iter; ++iter)
        mean_ang_vel += Vec3f(iter->angular_velocity[0],iter->angular_velocity[1], iter->angular_velocity[2]);

    if ((end_iter-begin_iter) > 0)
        mean_ang_vel *= 1.0f / (end_iter-begin_iter);

    // Transform the mean angular velocity from the IMU
    // frame to the cam0 and cam1 frames.
    Vec3f cam_mean_ang_vel = R_cam_imu.t() * mean_ang_vel;

    // Compute the relative rotation.
    double dtime = curr_img_ptr->timeStampToSec-
        prev_img_ptr->timeStampToSec;
    Rodrigues(cam_mean_ang_vel*dtime, cam_R_p2c);
    cam_R_p2c = cam_R_p2c.t();
}


void ImageProcessor::predictFeatureTracking(
    const vector<cv::Point2f>& input_pts,
    const cv::Matx33f& R_p_c,
    const cv::Vec4d& intrinsics,
    vector<cv::Point2f>& compensated_pts) {
    // Return directly if there are no input features.
    if (input_pts.empty()) {
        compensated_pts.clear();
        return;
    }
    compensated_pts.resize(input_pts.size());

    // Intrinsic matrix.
    cv::Matx33f K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);
    cv::Matx33f H = K * R_p_c * K.inv();  

    for (int i = 0; i < input_pts.size(); ++i) {
        cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
        cv::Vec3f p2 = H * p1;
        compensated_pts[i].x = p2[0] / p2[2];
        compensated_pts[i].y = p2[1] / p2[2];
    }
}


void ImageProcessor::rescalePoints(
    vector<Point2f>& pts1, vector<Point2f>& pts2,
    float& scaling_factor) {
    scaling_factor = 0.0f;

    for (int i = 0; i < pts1.size(); ++i) {
        scaling_factor += sqrt(pts1[i].dot(pts1[i]));
        scaling_factor += sqrt(pts2[i].dot(pts2[i]));
    }

    scaling_factor = (pts1.size()+pts2.size()) /
        scaling_factor * sqrt(2.0f);

    for (int i = 0; i < pts1.size(); ++i) {
        pts1[i] *= scaling_factor;
        pts2[i] *= scaling_factor;
    }
}


void ImageProcessor::clahe() {
    const Mat& curr_img = curr_img_ptr->image;

    // CLAHE
    cv::Mat img_;
    if (processor_config.flag_equalize) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(curr_img, img_);
    }
    else
        img_ = curr_img;
}


bool ImageProcessor::initializeVilib() {
    FeatureTrackerOptions l_feature_tracker_options;
    l_feature_tracker_options.affine_est_gain = false;
    l_feature_tracker_options.affine_est_offset = false;
    l_feature_tracker_options.reset_before_detection = false;
    l_feature_tracker_options.use_best_n_features = processor_config.max_features_num;
    l_feature_tracker_options.min_tracks_to_detect_new_features = l_feature_tracker_options.use_best_n_features;

    // Create feature detector for the GPU
    if (FEATURE_DETECTOR_USED == FEATURE_DETECTOR_FAST)
    {
        detector_gpu.reset(new FASTGPU(cam_resolution[0],
                                       cam_resolution[1],
                                       FEATURE_DETECTOR_CELL_SIZE_WIDTH,
                                       FEATURE_DETECTOR_CELL_SIZE_HEIGHT,
                                       FEATURE_DETECTOR_MIN_LEVEL,
                                       FEATURE_DETECTOR_MAX_LEVEL,
                                       FEATURE_DETECTOR_HORIZONTAL_BORDER,
                                       FEATURE_DETECTOR_VERTICAL_BORDER,
                                       FEATURE_DETECTOR_FAST_EPISLON,
                                       FEATURE_DETECTOR_FAST_ARC_LENGTH,
                                       FEATURE_DETECTOR_FAST_SCORE));
    }
    else if (FEATURE_DETECTOR_USED == FEATURE_DETECTOR_HARRIS || FEATURE_DETECTOR_USED == FEATURE_DETECTOR_SHI_TOMASI)
    {
        detector_gpu.reset(new HarrisGPU(cam_resolution[0],
                                         cam_resolution[1],
                                         FEATURE_DETECTOR_CELL_SIZE_WIDTH,
                                         FEATURE_DETECTOR_CELL_SIZE_HEIGHT,
                                         FEATURE_DETECTOR_MIN_LEVEL,
                                         FEATURE_DETECTOR_MAX_LEVEL,
                                         FEATURE_DETECTOR_HORIZONTAL_BORDER,
                                         FEATURE_DETECTOR_VERTICAL_BORDER,
                                         FEATURE_DETECTOR_HARRIS_BORDER_TYPE,
                                         (FEATURE_DETECTOR_USED == FEATURE_DETECTOR_HARRIS),
                                         FEATURE_DETECTOR_HARRIS_K,
                                         FEATURE_DETECTOR_HARRIS_QUALITY_LEVEL));
    }

    tracker_gpu.reset(new FeatureTrackerGPU(l_feature_tracker_options, 1));
    tracker_gpu->setDetectorGPU(detector_gpu, 0);

    PyramidPool::init(1,
                      cam_resolution[0],
                      cam_resolution[1],
                      1, // grayscale
                      FRAME_IMAGE_PYRAMID_LEVELS,
                      IMAGE_PYRAMID_MEMORY_TYPE);
}


void ImageProcessor::trackImage(
        const cv::Mat &img,
        std::map<FeatureIDType, cv::Point2f> &points) {
    // Tracking variables
    std::size_t l_total_tracked_ftr_cnt, l_total_detected_ftr_cnt;

    // Create image frame and push it to frames collection (mono)
    std::shared_ptr<Frame> l_frame = std::make_shared<Frame>(img,
                                                             0,
                                                             FRAME_IMAGE_PYRAMID_LEVELS);

    // Create collection of frames
    std::vector<std::shared_ptr<Frame>> l_framelist;
    l_framelist.push_back(l_frame);

    // Create FrameBundle for tracking
    std::shared_ptr<FrameBundle> l_framebundle(new FrameBundle(l_framelist));

    // Track
    tracker_gpu->track(l_framebundle,
                       l_total_tracked_ftr_cnt,
                       l_total_detected_ftr_cnt);

    // Populate hashmap where the key is the feature id
    // and the value the (x,y) coordinate point of the feature
    for (std::size_t i = 0; i < l_frame->num_features_; ++i)
    {
        //printf("Track ID: %d\n", l_frame->track_id_vec_[i]);
        points[l_frame->track_id_vec_[i]] = cv::Point2f((int)l_frame->px_vec_.col(i)[0], (int)l_frame->px_vec_.col(i)[1]);
    }
}


void ImageProcessor::clearDeadFeatures() {
    // IDs of inactive features
    std::vector<FeatureIDType> inactive_feature_ids;

    // Check which features are dead
    for (const auto& point: previous_tracked_points_map)
    {
        if (current_tracked_points_map.find(point.first) == current_tracked_points_map.end())
        {
            inactive_feature_ids.push_back(point.first);
        }
    }

    // Clear dead features info
    for (auto id: inactive_feature_ids)
    {
        tracked_points_initial_map.erase(id);
        tracked_points_lifetime_map.erase(id);
        previous_tracked_points_map.erase(id);
    }
}


bool ImageProcessor::initializeFirstFrame() {
    printf("initializeFirstFrame called...\n");

    // Get current image
    const Mat& img = curr_img_ptr->image;

    // The tracking call for this first stage only
    // consists of detecting new features from the
    // very first image received
    trackImage(img, current_tracked_points_map);

//    printf("Newly detected points size %zu\n", current_tracked_points_map.size());

    // Initialize last publish time
    last_pub_time = curr_img_ptr->timeStampToSec;

    if (current_tracked_points_map.size()>20)
        return true;
    else
        return false;
}


bool ImageProcessor::trackFirstFeatures(
        const std::vector<ImuData>& imu_msg_buffer) {

    // Integrate gyro data to get a guess of rotation between current and previous image
//    integrateImuData(R_Prev2Curr, imu_msg_buffer);

    //TODO: IMU pre-integration

    // IDs of active features tracked
    std::vector<int> active_ids_inlier;

    // Current tracked features till now
    // become the previous tracked features
    previous_tracked_points_map.clear();
    current_tracked_points_map.swap(previous_tracked_points_map);

    // Track previous points on new image
    const Mat& img = curr_img_ptr->image;
    trackImage(img, current_tracked_points_map);

    // Previous and current points which have been
    // tracked through two consecutive images
    std::vector<cv::Point2f> current_tracked_points_inlier;
    std::vector<cv::Point2f> previous_tracked_points_inlier;

    // Filter currently tracked points between
    // inliers and newly detected points and initialize
    // their lifetime and initial point
    for (auto it : current_tracked_points_map)
    {
        // Tracked point
        if (previous_tracked_points_map.find(it.first) != previous_tracked_points_map.end())
        {
            // Make sure tracked point is within image region
            if (it.second.y < 0 || it.second.y > curr_img_ptr->image.rows-1 ||
                it.second.x < 0 || it.second.x > curr_img_ptr->image.cols-1)
                continue;

            // Mark point as active
            active_ids_inlier.push_back(it.first);

            // Tracked point initial lifetime
            tracked_points_lifetime_map[it.first] = 2;

            // Tracked point in previous and current frame
            cv::Point2f current_point = it.second;
            cv::Point2f previous_point = previous_tracked_points_map[it.first];

            // Populate inlier vectors
            current_tracked_points_inlier.push_back(current_point);
            previous_tracked_points_inlier.push_back(previous_point);
        }
        // Newly detected point
        else
        {
            // Newly detected point lifetime
            tracked_points_lifetime_map[it.first] = 1;
        }

        // Initial points (required by the estimator)
        tracked_points_initial_map[it.first] = cv::Point2f(-1, -1);
    }

    //TODO: refine points using ORB descriptor distance

    // Undistorted inlier tracked points
    vector<Point2f> curr_unpts_inlier(current_tracked_points_inlier.size());
    vector<Point2f> prev_unpts_inlier(previous_tracked_points_inlier.size());
    undistortPoints(
            previous_tracked_points_inlier, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, prev_unpts_inlier,
            cv::Matx33d::eye(), cam_intrinsics);
    undistortPoints(
            current_tracked_points_inlier, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, curr_unpts_inlier,
            cv::Matx33d::eye(), cam_intrinsics);

    // Compute RANSAC inliers
    vector<unsigned char> ransac_inliers;
    findFundamentalMat(prev_unpts_inlier, curr_unpts_inlier,cv::FM_RANSAC,1.0,0.99, ransac_inliers);

    // Refine points with RANSAC
    vector<int> active_ids_matched(0);
    vector<Point2f> prev_pts_matched(0);
    vector<Point2f> curr_pts_matched(0);
    removeUnmarkedElements(active_ids_inlier, ransac_inliers, active_ids_matched);
    removeUnmarkedElements(previous_tracked_points_inlier, ransac_inliers,prev_pts_matched);
    removeUnmarkedElements(current_tracked_points_inlier, ransac_inliers, curr_pts_matched);

    // Features initialized failed if less than 20 inliers are tracked
    if (curr_pts_matched.size() < 20)
    {
        printf("Not enough features after RANSAC1...");
        return false;
    }

    // Clear vectors
    vector<Point2f>().swap(prev_pts_);
    vector<Point2f>().swap(curr_pts_);
    vector<int>().swap(pts_lifetime_);
    vector<Point2f>().swap(init_pts_);
    vector<FeatureIDType>().swap(pts_ids_);

//    printf("Points inserted: %zu\n", prev_pts_matched.size());
//    printf("Points inserted: %zu\n", curr_pts_matched.size());
//    printf("IDs inserted: %zu\n", active_ids_matched.size());

    // Fill current and previous undistorted points
    for (int i = 0; i < prev_pts_matched.size(); ++i) {
        prev_pts_.push_back(prev_pts_matched[i]);
        curr_pts_.push_back(curr_pts_matched[i]);
    }

    // Fill respective lifetime, initial pts and ids of tracked point
    for (auto id: active_ids_matched) {
        pts_ids_.push_back(id);
        init_pts_.emplace_back(tracked_points_initial_map[id]);
        pts_lifetime_.push_back(tracked_points_lifetime_map[id]);
    }

//    printf("Features after initial tracking: %zu\n", prev_pts_.size());

    return true;
}


void ImageProcessor::trackFeatures() {
    // Number of the features before tracking.
    before_tracking = prev_pts_.size();

    // Abort tracking if there is no
    // features in the previous frame.
    if (0 == before_tracking) {
        printf("No feature in prev img !\n");
        return;
    }

    //TODO: IMU pre-integration

    // IDs of active features tracked
    std::vector<int> active_ids_inlier;

    // Current tracked features till now
    // become the previous tracked features
    previous_tracked_points_map.clear();
    current_tracked_points_map.swap(previous_tracked_points_map);

    // Track previous points on new image
    const Mat& img = curr_img_ptr->image;
    trackImage(img, current_tracked_points_map);

    // Remove dead features
    clearDeadFeatures();

    // Previous and current points which have been
    // tracked through two consecutive images
    std::vector<cv::Point2f> current_tracked_points_inlier;
    std::vector<cv::Point2f> previous_tracked_points_inlier;

    // Filter currently tracked points between
    // inliers and newly detected points and initialize
    // their lifetime and initial point
    for (auto it : current_tracked_points_map)
    {
        // Tracked point
        if (previous_tracked_points_map.find(it.first) != previous_tracked_points_map.end())
        {
            // Make sure tracked point is within image region
            if (it.second.y < 0 || it.second.y > curr_img_ptr->image.rows-1 ||
                it.second.x < 0 || it.second.x > curr_img_ptr->image.cols-1)
                continue;

            active_ids_inlier.push_back(it.first);

            // Corresponding points in tracking process
            cv::Point2f current_point = it.second;
            cv::Point2f previous_point = previous_tracked_points_map[it.first];

            // Populate inliers
            current_tracked_points_inlier.push_back(current_point);
            previous_tracked_points_inlier.push_back(previous_point);

            // Tracked point lifetime
            tracked_points_lifetime_map[it.first] += 1;
        }
        // Newly detected point
        else
        {
            // Newly detected point lifetime
            tracked_points_lifetime_map[it.first] = 1;
        }

        // Initial points (required by the estimator)
        tracked_points_initial_map[it.first] = previous_tracked_points_map[it.first];
    }

    // Number of features left after tracking.
    after_tracking = current_tracked_points_inlier.size();

    // debug log
    if (0 == after_tracking) {
        printf("No feature is tracked !");
        vector<Point2f>().swap(prev_pts_);
        vector<Point2f>().swap(curr_pts_);
        vector<FeatureIDType>().swap(pts_ids_);
        vector<int>().swap(pts_lifetime_);
        vector<Point2f>().swap(init_pts_);
        vector<Mat>().swap(vOrbDescriptors);
        return;
    }

    // Undistorted points
    vector<Point2f> curr_unpts_inlier(current_tracked_points_inlier.size());
    vector<Point2f> prev_unpts_inlier(previous_tracked_points_inlier.size());
    undistortPoints(
            previous_tracked_points_inlier, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, prev_unpts_inlier,
            cv::Matx33d::eye(), cam_intrinsics);
    undistortPoints(
            current_tracked_points_inlier, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, curr_unpts_inlier,
            cv::Matx33d::eye(), cam_intrinsics);

    // Compute RANSAC inliers
    vector<unsigned char> ransac_inliers;
    findFundamentalMat(prev_unpts_inlier, curr_unpts_inlier, cv::FM_RANSAC, 1.0, 0.99, ransac_inliers);

    vector<int> active_ids_matched(0);
    vector<Point2f> prev_pts_matched(0);
    vector<Point2f> curr_pts_matched(0);
    removeUnmarkedElements(active_ids_inlier, ransac_inliers, active_ids_matched);
    removeUnmarkedElements(previous_tracked_points_inlier, ransac_inliers,prev_pts_matched);
    removeUnmarkedElements(current_tracked_points_inlier, ransac_inliers, curr_pts_matched);

    // Features initialized failed if less than 20 inliers are tracked
    if (curr_pts_matched.size() < 20)
    {
        printf("No feature survive after RANSAC2: %zu\n", curr_pts_matched.size());
    }

    // Clear vectors
    vector<Point2f>().swap(prev_pts_);
    vector<Point2f>().swap(curr_pts_);
    vector<int>().swap(pts_lifetime_);
    vector<Point2f>().swap(init_pts_);
    vector<FeatureIDType>().swap(pts_ids_);

    // Fill current and previous undistorted points
    for (int i = 0; i < prev_unpts_inlier.size(); ++i) {
        prev_pts_.push_back(prev_pts_matched[i]);
        curr_pts_.push_back(curr_pts_matched[i]);
    }

    // Fill respective lifetime, initial pts and ids of tracked point
    for (auto id: active_ids_matched) {
        pts_ids_.push_back(id);
        init_pts_.emplace_back(tracked_points_initial_map[id]);
        pts_lifetime_.push_back(tracked_points_lifetime_map[id]);
    }

//    printf("Features after tracking: %zu\n", prev_pts_.size());
}


void ImageProcessor::undistortPoints(
    const vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    vector<cv::Point2f>& pts_out,
    const cv::Matx33d &rectification_matrix,
    const cv::Vec4d &new_intrinsics) {
    if (pts_in.empty()) return;

    const cv::Matx33d K(
            intrinsics[0], 0.0, intrinsics[2],
            0.0, intrinsics[1], intrinsics[3],
            0.0, 0.0, 1.0);

    const cv::Matx33d K_new(
            new_intrinsics[0], 0.0, new_intrinsics[2],
            0.0, new_intrinsics[1], new_intrinsics[3],
            0.0, 0.0, 1.0);

    if (distortion_model == "radtan") {
        cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                            rectification_matrix, K_new);       
    } else if (distortion_model == "equidistant") {
        cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                                     rectification_matrix, K_new);
    } else {
        printf("The model %s is unrecognized, use radtan instead...\n",
                      distortion_model.c_str());
        cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                            rectification_matrix, K_new);
    }
}


// Get processed feature msg.
void ImageProcessor::getFeatureMsg(MonoCameraMeasurementPtr feature_msg_ptr) {
    feature_msg_ptr->timeStampToSec = curr_img_ptr->timeStampToSec;

    vector<Point2f> curr_points_undistorted(0);
    vector<Point2f> init_points_undistorted(0);     
    vector<Point2f> prev_points_undistorted(0);

    undistortPoints(
            curr_pts_, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, curr_points_undistorted);
    undistortPoints(
            init_pts_, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, init_points_undistorted);
    undistortPoints(
            prev_pts_, cam_intrinsics, cam_distortion_model,
            cam_distortion_coeffs, prev_points_undistorted);

    // time interval between current and previous image
    double dt_1 = curr_img_time-prev_img_time;
    bool prev_is_last = prev_img_time==last_pub_time;
    double dt_2 = (prev_is_last ? dt_1 : prev_img_time-last_pub_time);

//    printf("Points id size feature message: %zu\n", pts_ids_.size());

    for (int i = 0; i < pts_ids_.size(); ++i) {
        feature_msg_ptr->features.push_back(MonoFeatureMeasurement());
        feature_msg_ptr->features[i].id = pts_ids_[i];
        feature_msg_ptr->features[i].u = curr_points_undistorted[i].x;
        feature_msg_ptr->features[i].v = curr_points_undistorted[i].y;
        feature_msg_ptr->features[i].u_vel =
                (curr_points_undistorted[i].x-prev_points_undistorted[i].x)/dt_1;
        feature_msg_ptr->features[i].v_vel =
                (curr_points_undistorted[i].y-prev_points_undistorted[i].y)/dt_1;
        if (init_pts_[i].x==-1 && init_pts_[i].y==-1) {    
            feature_msg_ptr->features[i].u_init = -1;
            feature_msg_ptr->features[i].v_init = -1;
        } else {
            feature_msg_ptr->features[i].u_init = init_points_undistorted[i].x;
            feature_msg_ptr->features[i].v_init = init_points_undistorted[i].y;
            init_pts_[i].x = -1;    
            init_pts_[i].y = -1;
            if (prev_is_last) {
                feature_msg_ptr->features[i].u_init_vel =
                        (curr_points_undistorted[i].x-init_points_undistorted[i].x)/dt_2;
                feature_msg_ptr->features[i].v_init_vel =
                        (curr_points_undistorted[i].y-init_points_undistorted[i].y)/dt_2;
            } else {
                feature_msg_ptr->features[i].u_init_vel =
                        (prev_points_undistorted[i].x-init_points_undistorted[i].x)/dt_2;
                feature_msg_ptr->features[i].v_init_vel =
                        (prev_points_undistorted[i].y-init_points_undistorted[i].y)/dt_2;
            }
        }

//        printf("Feature message id: %llu\n", feature_msg_ptr->features[i].id);
//        printf("Feature message u: %f\n", feature_msg_ptr->features[i].u);
//        printf("Feature message v: %f\n", feature_msg_ptr->features[i].v);
//        printf("Feature message u_vel: %f\n", feature_msg_ptr->features[i].u_vel);
//        printf("Feature message v_vel: %f\n", feature_msg_ptr->features[i].v_vel);
//        printf("Feature message u_init: %f\n", feature_msg_ptr->features[i].u_init);
//        printf("Feature message v_init: %f\n", feature_msg_ptr->features[i].v_init);
//        printf("Feature message u_init_vel: %f\n", feature_msg_ptr->features[i].u_init_vel);
//        printf("Feature message v_init_vel: %f\n", feature_msg_ptr->features[i].v_init_vel);
    }

//    printf("Feature message size: %zu\n", feature_msg_ptr->features.size());
}


void ImageProcessor::publish() {
    // Colors for different features.
    Scalar tracked(255, 0, 0);
    Scalar new_feature(0, 255, 0);

    // Create an output image.
    int img_height = curr_img_ptr->image.rows;
    int img_width = curr_img_ptr->image.cols;
    Mat out_img(img_height, img_width, CV_8UC3);
    cvtColor(curr_img_ptr->image, out_img, COLOR_GRAY2RGB);

    // Collect feature points in the previous frame, 
    // and feature points in the current frame, 
    // and lifetime of tracked features
    map<FeatureIDType, Point2f> prev_points;
    map<FeatureIDType, Point2f> curr_points;
    map<FeatureIDType, int> points_lifetime;

    for (int i = 0; i < pts_ids_.size(); i++) {
        prev_points[pts_ids_[i]] = prev_pts_[i];
        curr_points[pts_ids_[i]] = curr_pts_[i];
        points_lifetime[pts_ids_[i]] = pts_lifetime_[i];
    }

    // Draw tracked features.
    for (const auto& id : pts_ids_) {
        if (prev_points.find(id) != prev_points.end() &&
            curr_points.find(id) != curr_points.end()) {
            cv::Point2f prev_pt = prev_points[id];
            cv::Point2f curr_pt = curr_points[id];
            int life = points_lifetime[id];

            double len = std::min(1.0, 1.0 * life / 50);
            circle(out_img, curr_pt, 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 5);
            line(out_img, prev_pt, curr_pt, Scalar(0,128,0));

            prev_points.erase(id);
            curr_points.erase(id);
        }
    }

    visual_img = out_img;

    // Update last publish time and publish counter
    last_pub_time = curr_img_ptr->timeStampToSec;
    pub_counter++;
}

} // end namespace larvio
