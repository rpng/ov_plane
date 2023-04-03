/*
 * ov_plane: Monocular Visual-Inertial Odometry with Planar Regularities
 * Copyright (C) 2022-2023 Chuchu Chen
 * Copyright (C) 2022-2023 Patrick Geneva
 * Copyright (C) 2022-2023 Guoquan Huang
 *
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "ROS1Visualizer.h"

#include "core/VioManager.h"
#include "ros/ROSVisualizerHelper.h"
#include "sim/Simulator.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "track_plane/TrackPlane.h"
#include "utils/dataset_reader.h"
#include "utils/helper.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "CDT.h"
#include "render_model.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_plane;

ROS1Visualizer::ROS1Visualizer(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim)
    : _nh(nh), _app(app), _sim(sim), thread_update_running(false) {

  // Setup our transform broadcaster
  mTfBr = std::make_shared<tf::TransformBroadcaster>();

  // Create image transport
  image_transport::ImageTransport it(*_nh);

  // Setup pose and path publisher
  pub_poseimu = nh->advertise<geometry_msgs::PoseWithCovarianceStamped>("poseimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_poseimu.getTopic().c_str());
  pub_odomimu = nh->advertise<nav_msgs::Odometry>("odomimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_odomimu.getTopic().c_str());
  pub_pathimu = nh->advertise<nav_msgs::Path>("pathimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_pathimu.getTopic().c_str());

  // 3D points publishing
  pub_points_msckf = nh->advertise<sensor_msgs::PointCloud2>("points_msckf", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_msckf.getTopic().c_str());
  pub_points_slam = nh->advertise<sensor_msgs::PointCloud2>("points_slam", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_msckf.getTopic().c_str());
  pub_points_aruco = nh->advertise<sensor_msgs::PointCloud2>("points_aruco", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_aruco.getTopic().c_str());
  pub_points_sim = nh->advertise<sensor_msgs::PointCloud2>("points_sim", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_sim.getTopic().c_str());
  pub_points_tracker = nh->advertise<sensor_msgs::PointCloud2>("points_tracker", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_tracker.getTopic().c_str());

  // Planes :)
  pub_plane_points = nh->advertise<sensor_msgs::PointCloud2>("points_plane_slam", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_plane_points.getTopic().c_str());
  pub_plane_constraints = nh->advertise<visualization_msgs::Marker>("pt_on_plane_constarints", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_plane_constraints.getTopic().c_str());
  pub_plane_sim = nh->advertise<visualization_msgs::MarkerArray>("plane_map", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_plane_sim.getTopic().c_str());
  pub_plane_slam = nh->advertise<visualization_msgs::MarkerArray>("plane_slam", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_plane_slam.getTopic().c_str());
  pub_plane_slam_map = nh->advertise<visualization_msgs::MarkerArray>("plane_slam_map", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_plane_slam_map.getTopic().c_str());

  // Our tracking image
  it_pub_tracks = it.advertise("trackhist", 2);
  PRINT_DEBUG("Publishing: %s\n", it_pub_tracks.getTopic().c_str());
  it_pub_tracksplane = it.advertise("planehist", 2);
  PRINT_DEBUG("Publishing: %s\n", it_pub_tracksplane.getTopic().c_str());
  it_pub_tracksplanenorm = it.advertise("planenorm", 2);
  PRINT_DEBUG("Publishing: %s\n", it_pub_tracksplanenorm.getTopic().c_str());
  it_pub_ardisp = it.advertise("planeardisp", 2);
  PRINT_DEBUG("Publishing: %s\n", it_pub_ardisp.getTopic().c_str());

  // Groundtruth publishers
  pub_posegt = nh->advertise<geometry_msgs::PoseStamped>("posegt", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_posegt.getTopic().c_str());
  pub_pathgt = nh->advertise<nav_msgs::Path>("pathgt", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_pathgt.getTopic().c_str());

  // Loop closure publishers
  pub_loop_pose = nh->advertise<nav_msgs::Odometry>("loop_pose", 2);
  pub_loop_point = nh->advertise<sensor_msgs::PointCloud>("loop_feats", 2);
  pub_loop_extrinsic = nh->advertise<nav_msgs::Odometry>("loop_extrinsic", 2);
  pub_loop_intrinsics = nh->advertise<sensor_msgs::CameraInfo>("loop_intrinsics", 2);
  it_pub_loop_img_depth = it.advertise("loop_depth", 2);
  it_pub_loop_img_depth_color = it.advertise("loop_depth_colored", 2);

  // option to enable publishing of global to IMU transformation
  nh->param<bool>("publish_global_to_imu_tf", publish_global2imu_tf, true);
  nh->param<bool>("publish_calibration_tf", publish_calibration_tf, true);

  // Load AR diaplay object if we have it
  nh->param<std::string>("path_object", path_to_object, "/home/chuchu/workspace/catkin_ws_plane/src/ov_plane/resources/teapot.obj");
  if (!path_to_object.empty()) {
    PRINT_DEBUG(" AR object file is: %s\n", path_to_object.c_str());
  }

  // Load groundtruth if we have it and are not doing simulation
  // NOTE: needs to be a csv ASL format file
  if (nh->hasParam("path_gt") && _sim == nullptr) {
    std::string path_to_gt;
    nh->param<std::string>("path_gt", path_to_gt, "");
    if (!path_to_gt.empty()) {
      DatasetReader::load_gt_file(path_to_gt, gt_states);
      PRINT_DEBUG("gt file path is: %s\n", path_to_gt.c_str());
    }
  }

  // Load if we should save the total state to file
  // If so, then open the file and create folders as needed
  nh->param<bool>("save_total_state", save_total_state, false);
  if (save_total_state) {

    // files we will open
    std::string filepath_est, filepath_std, filepath_gt;
    nh->param<std::string>("filepath_est", filepath_est, "state_estimate.txt");
    nh->param<std::string>("filepath_std", filepath_std, "state_deviation.txt");
    nh->param<std::string>("filepath_gt", filepath_gt, "state_groundtruth.txt");

    // If it exists, then delete it
    if (boost::filesystem::exists(filepath_est))
      boost::filesystem::remove(filepath_est);
    if (boost::filesystem::exists(filepath_std))
      boost::filesystem::remove(filepath_std);

    // Create folder path to this location if not exists
    boost::filesystem::create_directories(boost::filesystem::path(filepath_est.c_str()).parent_path());
    boost::filesystem::create_directories(boost::filesystem::path(filepath_std.c_str()).parent_path());

    // Open the files
    of_state_est.open(filepath_est.c_str());
    of_state_std.open(filepath_std.c_str());
    of_state_est << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;
    of_state_std << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;

    // Groundtruth if we are simulating
    if (_sim != nullptr) {
      if (boost::filesystem::exists(filepath_gt))
        boost::filesystem::remove(filepath_gt);
      boost::filesystem::create_directories(boost::filesystem::path(filepath_gt.c_str()).parent_path());
      of_state_gt.open(filepath_gt.c_str());
      of_state_gt << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;
    }
  }

  // Start thread for the image publishing
  if (_app->get_params().use_multi_threading_pubs) {
    std::thread thread([&] {
      ros::Rate loop_rate(20);
      while (ros::ok()) {
        publish_images();
        loop_rate.sleep();
      }
    });
    thread.detach();
  }
}

void ROS1Visualizer::setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser) {

  // We need a valid parser
  assert(parser != nullptr);

  // Create imu subscriber (handle legacy ros param info)
  std::string topic_imu;
  _nh->param<std::string>("topic_imu", topic_imu, "/imu0");
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  sub_imu = _nh->subscribe(topic_imu, 1000, &ROS1Visualizer::callback_inertial, this);

  // Logic for sync stereo subscriber
  // https://answers.ros.org/question/96346/subscribe-to-two-image_raws-with-one-function/?answer=96491#post-id-96491
  if (_app->get_params().state_options.num_cameras == 2) {
    // Read in the topics
    std::string cam_topic0, cam_topic1;
    _nh->param<std::string>("topic_camera" + std::to_string(0), cam_topic0, "/cam" + std::to_string(0) + "/image_raw");
    _nh->param<std::string>("topic_camera" + std::to_string(1), cam_topic1, "/cam" + std::to_string(1) + "/image_raw");
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(0), "rostopic", cam_topic0);
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(1), "rostopic", cam_topic1);
    // Create sync filter (they have unique pointers internally, so we have to use move logic here...)
    auto image_sub0 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(*_nh, cam_topic0, 1);
    auto image_sub1 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(*_nh, cam_topic1, 1);
    auto sync = std::make_shared<message_filters::Synchronizer<sync_pol>>(sync_pol(10), *image_sub0, *image_sub1);
    sync->registerCallback(boost::bind(&ROS1Visualizer::callback_stereo, this, _1, _2, 0, 1));
    // Append to our vector of subscribers
    sync_cam.push_back(sync);
    sync_subs_cam.push_back(image_sub0);
    sync_subs_cam.push_back(image_sub1);
    PRINT_DEBUG("subscribing to cam (stereo): %s\n", cam_topic0.c_str());
    PRINT_DEBUG("subscribing to cam (stereo): %s\n", cam_topic1.c_str());
  } else {
    // Now we should add any non-stereo callbacks here
    for (int i = 0; i < _app->get_params().state_options.num_cameras; i++) {
      // read in the topic
      std::string cam_topic;
      _nh->param<std::string>("topic_camera" + std::to_string(i), cam_topic, "/cam" + std::to_string(i) + "/image_raw");
      parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
      // create subscriber
      subs_cam.push_back(_nh->subscribe<sensor_msgs::Image>(cam_topic, 10, boost::bind(&ROS1Visualizer::callback_monocular, this, _1, i)));
      PRINT_DEBUG("subscribing to cam (mono): %s\n", cam_topic.c_str());
    }
  }
}

void ROS1Visualizer::visualize() {

  // Return if we have already visualized
  if (last_visualization_timestamp == _app->get_state()->_timestamp && _app->initialized())
    return;
  last_visualization_timestamp = _app->get_state()->_timestamp;

  // Start timing
  boost::posix_time::ptime rT0_1, rT0_2;
  rT0_1 = boost::posix_time::microsec_clock::local_time();

  // publish current image (only if not multi-threaded)
  if (!_app->get_params().use_multi_threading_pubs)
    publish_images();

  // Return if we have not inited
  if (!_app->initialized())
    return;

  // Save the start time of this dataset
  if (!start_time_set) {
    rT1 = boost::posix_time::microsec_clock::local_time();
    start_time_set = true;
  }

  // publish state
  publish_state();

  // publish points
  publish_features();

  // Publish gt if we have it
  publish_groundtruth();

  // Publish keyframe information
  publish_loopclosure_information();

  // Publish planes
  publish_planes();

  // Publish plane information
  publish_plane_information();

  // Save total state
  if (save_total_state) {
    ROSVisualizerHelper::sim_save_total_state_to_file(_app->get_state(), _sim, of_state_est, of_state_std, of_state_gt);
  }

  // Print how much time it took to publish / displaying things
  rT0_2 = boost::posix_time::microsec_clock::local_time();
  double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for visualization\n" RESET, time_total);
}

void ROS1Visualizer::visualize_odometry(double timestamp) {

  // Return if we have not inited
  if (!_app->initialized())
    return;

  // Get fast propagate state at the desired timestamp
  std::shared_ptr<State> state = _app->get_state();
  Eigen::Matrix<double, 13, 1> state_plus = Eigen::Matrix<double, 13, 1>::Zero();
  Eigen::Matrix<double, 12, 12> cov_plus = Eigen::Matrix<double, 12, 12>::Zero();
  if (!_app->get_propagator()->fast_state_propagate(state, timestamp, state_plus, cov_plus))
    return;

  // Publish our odometry message if requested
  if (pub_odomimu.getNumSubscribers() != 0) {

    nav_msgs::Odometry odomIinM;
    odomIinM.header.stamp = ros::Time(timestamp);
    odomIinM.header.frame_id = "global";

    // The POSE component (orientation and position)
    odomIinM.pose.pose.orientation.x = state_plus(0);
    odomIinM.pose.pose.orientation.y = state_plus(1);
    odomIinM.pose.pose.orientation.z = state_plus(2);
    odomIinM.pose.pose.orientation.w = state_plus(3);
    odomIinM.pose.pose.position.x = state_plus(4);
    odomIinM.pose.pose.position.y = state_plus(5);
    odomIinM.pose.pose.position.z = state_plus(6);

    // The TWIST component (angular and linear velocities)
    odomIinM.child_frame_id = "imu";
    odomIinM.twist.twist.linear.x = state_plus(7);   // vel in local frame
    odomIinM.twist.twist.linear.y = state_plus(8);   // vel in local frame
    odomIinM.twist.twist.linear.z = state_plus(9);   // vel in local frame
    odomIinM.twist.twist.angular.x = state_plus(10); // we do not estimate this...
    odomIinM.twist.twist.angular.y = state_plus(11); // we do not estimate this...
    odomIinM.twist.twist.angular.z = state_plus(12); // we do not estimate this...

    // Finally set the covariance in the message (in the order position then orientation as per ros convention)
    Eigen::Matrix<double, 12, 12> Phi = Eigen::Matrix<double, 12, 12>::Zero();
    Phi.block(0, 3, 3, 3).setIdentity();
    Phi.block(3, 0, 3, 3).setIdentity();
    Phi.block(6, 6, 6, 6).setIdentity();
    cov_plus = Phi * cov_plus * Phi.transpose();
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        odomIinM.pose.covariance[6 * r + c] = cov_plus(r, c);
      }
    }
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        odomIinM.twist.covariance[6 * r + c] = cov_plus(r + 6, c + 6);
      }
    }
    pub_odomimu.publish(odomIinM);
  }

  // Publish our transform on TF
  // NOTE: since we use JPL we have an implicit conversion to Hamilton when we publish
  // NOTE: a rotation from GtoI in JPL has the same xyzw as a ItoG Hamilton rotation
  auto odom_pose = std::make_shared<ov_type::PoseJPL>();
  odom_pose->set_value(state_plus.block(0, 0, 7, 1));
  tf::StampedTransform trans = ROSVisualizerHelper::get_stamped_transform_from_pose(odom_pose, false);
  trans.frame_id_ = "global";
  trans.child_frame_id_ = "imu";
  if (publish_global2imu_tf) {
    mTfBr->sendTransform(trans);
  }

  // Loop through each camera calibration and publish it
  for (const auto &calib : state->_calib_IMUtoCAM) {
    tf::StampedTransform trans_calib = ROSVisualizerHelper::get_stamped_transform_from_pose(calib.second, true);
    trans_calib.frame_id_ = "imu";
    trans_calib.child_frame_id_ = "cam" + std::to_string(calib.first);
    if (publish_calibration_tf) {
      mTfBr->sendTransform(trans_calib);
    }
  }
}

void ROS1Visualizer::visualize_final() {

  // Final time offset value
  if (_app->get_state()->_options.do_calib_camera_timeoffset) {
    PRINT_INFO(REDPURPLE "camera-imu timeoffset = %.5f\n\n" RESET, _app->get_state()->_calib_dt_CAMtoIMU->value()(0));
  }

  // Final camera intrinsics
  if (_app->get_state()->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = _app->get_state()->_cam_intrinsics.at(i);
      PRINT_INFO(REDPURPLE "cam%d intrinsics:\n" RESET, (int)i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n" RESET, calib->value()(0), calib->value()(1), calib->value()(2), calib->value()(3));
      PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,%.5f\n\n" RESET, calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Final camera extrinsics
  if (_app->get_state()->_options.do_calib_camera_pose) {
    for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = _app->get_state()->_calib_IMUtoCAM.at(i);
      Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
      T_CtoI.block(0, 0, 3, 3) = quat_2_Rot(calib->quat()).transpose();
      T_CtoI.block(0, 3, 3, 1) = -T_CtoI.block(0, 0, 3, 3) * calib->pos();
      PRINT_INFO(REDPURPLE "T_C%dtoI:\n" RESET, i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(0, 0), T_CtoI(0, 1), T_CtoI(0, 2), T_CtoI(0, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(1, 0), T_CtoI(1, 1), T_CtoI(1, 2), T_CtoI(1, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(2, 0), T_CtoI(2, 1), T_CtoI(2, 2), T_CtoI(2, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n\n" RESET, T_CtoI(3, 0), T_CtoI(3, 1), T_CtoI(3, 2), T_CtoI(3, 3));
    }
  }

  // Publish RMSE if we have it
  if (!gt_states.empty()) {
    PRINT_INFO(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_INFO(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
  }

  // Publish RMSE and NEES if doing simulation
  if (_sim != nullptr) {
    PRINT_WARNING(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_WARNING(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
    PRINT_WARNING(REDPURPLE "NEES: %.3f (deg) orientation\n" RESET, summed_nees_ori / summed_number);
    PRINT_WARNING(REDPURPLE "NEES: %.3f (m) position\n\n" RESET, summed_nees_pos / summed_number);
  }

  // Print the total time
  rT2 = boost::posix_time::microsec_clock::local_time();
  PRINT_WARNING(REDPURPLE "TIME: %.3f seconds\n\n" RESET, (rT2 - rT1).total_microseconds() * 1e-6);
}

void ROS1Visualizer::callback_inertial(const sensor_msgs::Imu::ConstPtr &msg) {

  // convert into correct format
  ov_core::ImuData message;
  message.timestamp = msg->header.stamp.toSec();
  message.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  message.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

  // send it to our VIO system
  _app->feed_measurement_imu(message);
  visualize_odometry(message.timestamp);

  // If the processing queue is currently active / running just return so we can keep getting measurements
  // Otherwise create a second thread to do our update in an async manor
  // The visualization of the state, images, and features will be synchronous with the update!
  if (thread_update_running)
    return;
  thread_update_running = true;
  std::thread thread([&] {
    // Lock on the queue (prevents new images from appending)
    std::lock_guard<std::mutex> lck(camera_queue_mtx);

    // Count how many unique image streams
    std::map<int, bool> unique_cam_ids;
    for (const auto &cam_msg : camera_queue) {
      unique_cam_ids[cam_msg.sensor_ids.at(0)] = true;
    }

    // If we do not have enough unique cameras then we need to wait
    // We should wait till we have one of each camera to ensure we propagate in the correct order
    auto params = _app->get_params();
    size_t num_unique_cameras = (params.state_options.num_cameras == 2) ? 1 : params.state_options.num_cameras;
    if (unique_cam_ids.size() == num_unique_cameras) {

      // Loop through our queue and see if we are able to process any of our camera measurements
      // We are able to process if we have at least one IMU measurement greater than the camera time
      double timestamp_imu_inC = message.timestamp - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
      while (!camera_queue.empty() && camera_queue.at(0).timestamp < timestamp_imu_inC) {
        auto rT0_1 = boost::posix_time::microsec_clock::local_time();
        double update_dt = 100.0 * (timestamp_imu_inC - camera_queue.at(0).timestamp);
        _app->feed_measurement_camera(camera_queue.at(0));
        visualize();
        camera_queue.pop_front();
        auto rT0_2 = boost::posix_time::microsec_clock::local_time();
        double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
        PRINT_INFO(BLUE "[TIME]: %.4f seconds total (%.1f hz, %.2f ms behind)\n" RESET, time_total, 1.0 / time_total, update_dt);
      }
    }
    thread_update_running = false;
  });

  // If we are single threaded, then run single threaded
  // Otherwise detach this thread so it runs in the background!
  if (!_app->get_params().use_multi_threading_subs) {
    thread.join();
  } else {
    thread.detach();
  }
}

void ROS1Visualizer::callback_monocular(const sensor_msgs::ImageConstPtr &msg0, int cam_id0) {

  // Check if we should drop this image
  double timestamp = msg0->header.stamp.toSec();
  double time_delta = 1.0 / _app->get_params().track_frequency;
  if (camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && timestamp < camera_last_timestamp.at(cam_id0) + time_delta) {
    return;
  }
  camera_last_timestamp[cam_id0] = timestamp;

  // Get the image
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Create the measurement
  ov_core::CameraData message;
  message.timestamp = cv_ptr->header.stamp.toSec();
  message.sensor_ids.push_back(cam_id0);
  message.images.push_back(cv_ptr->image.clone());

  // Load the mask if we are using it, else it is empty
  // TODO: in the future we should get this from external pixel segmentation
  if (_app->get_params().use_mask) {
    message.masks.push_back(_app->get_params().masks.at(cam_id0));
  } else {
    message.masks.push_back(cv::Mat::zeros(cv_ptr->image.rows, cv_ptr->image.cols, CV_8UC1));
  }

  // append it to our queue of images
  std::lock_guard<std::mutex> lck(camera_queue_mtx);
  camera_queue.push_back(message);
  std::sort(camera_queue.begin(), camera_queue.end());
}

void ROS1Visualizer::callback_stereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1, int cam_id0,
                                     int cam_id1) {

  // Check if we should drop this image
  double timestamp = msg0->header.stamp.toSec();
  double time_delta = 1.0 / _app->get_params().track_frequency;
  if (camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && timestamp < camera_last_timestamp.at(cam_id0) + time_delta) {
    return;
  }
  camera_last_timestamp[cam_id0] = timestamp;

  // Get the image
  cv_bridge::CvImageConstPtr cv_ptr0;
  try {
    cv_ptr0 = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR("cv_bridge exception: %s\n", e.what());
    return;
  }

  // Get the image
  cv_bridge::CvImageConstPtr cv_ptr1;
  try {
    cv_ptr1 = cv_bridge::toCvShare(msg1, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR("cv_bridge exception: %s\n", e.what());
    return;
  }

  // Create the measurement
  ov_core::CameraData message;
  message.timestamp = cv_ptr0->header.stamp.toSec();
  message.sensor_ids.push_back(cam_id0);
  message.sensor_ids.push_back(cam_id1);
  message.images.push_back(cv_ptr0->image.clone());
  message.images.push_back(cv_ptr1->image.clone());

  // Load the mask if we are using it, else it is empty
  // TODO: in the future we should get this from external pixel segmentation
  if (_app->get_params().use_mask) {
    message.masks.push_back(_app->get_params().masks.at(cam_id0));
    message.masks.push_back(_app->get_params().masks.at(cam_id1));
  } else {
    // message.masks.push_back(cv::Mat(cv_ptr0->image.rows, cv_ptr0->image.cols, CV_8UC1, cv::Scalar(255)));
    message.masks.push_back(cv::Mat::zeros(cv_ptr0->image.rows, cv_ptr0->image.cols, CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv_ptr1->image.rows, cv_ptr1->image.cols, CV_8UC1));
  }

  // append it to our queue of images
  std::lock_guard<std::mutex> lck(camera_queue_mtx);
  camera_queue.push_back(message);
  std::sort(camera_queue.begin(), camera_queue.end());
}

void ROS1Visualizer::publish_state() {

  // Get the current state
  std::shared_ptr<State> state = _app->get_state();

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = state->_timestamp + t_ItoC;

  // Create pose of IMU (note we use the bag time)
  geometry_msgs::PoseWithCovarianceStamped poseIinM;
  poseIinM.header.stamp = ros::Time(timestamp_inI);
  poseIinM.header.seq = poses_seq_imu;
  poseIinM.header.frame_id = "global";
  poseIinM.pose.pose.orientation.x = state->_imu->quat()(0);
  poseIinM.pose.pose.orientation.y = state->_imu->quat()(1);
  poseIinM.pose.pose.orientation.z = state->_imu->quat()(2);
  poseIinM.pose.pose.orientation.w = state->_imu->quat()(3);
  poseIinM.pose.pose.position.x = state->_imu->pos()(0);
  poseIinM.pose.pose.position.y = state->_imu->pos()(1);
  poseIinM.pose.pose.position.z = state->_imu->pos()(2);

  // Finally set the covariance in the message (in the order position then orientation as per ros convention)
  std::vector<std::shared_ptr<Type>> statevars;
  statevars.push_back(state->_imu->pose()->p());
  statevars.push_back(state->_imu->pose()->q());
  Eigen::Matrix<double, 6, 6> covariance_posori = StateHelper::get_marginal_covariance(_app->get_state(), statevars);
  for (int r = 0; r < 6; r++) {
    for (int c = 0; c < 6; c++) {
      poseIinM.pose.covariance[6 * r + c] = covariance_posori(r, c);
    }
  }
  pub_poseimu.publish(poseIinM);

  //=========================================================
  //=========================================================
  // Append to our pose vector
  geometry_msgs::PoseStamped posetemp;
  posetemp.header = poseIinM.header;
  posetemp.pose = poseIinM.pose.pose;
  poses_imu.push_back(posetemp);

  // Create our path (imu)
  // NOTE: We downsample the number of poses as needed to prevent rviz crashes
  // NOTE: https://github.com/ros-visualization/rviz/issues/1107
  nav_msgs::Path arrIMU;
  arrIMU.header.stamp = ros::Time::now();
  arrIMU.header.seq = poses_seq_imu;
  arrIMU.header.frame_id = "global";
  for (size_t i = 0; i < poses_imu.size(); i += std::floor((double)poses_imu.size() / 16384.0) + 1) {
    arrIMU.poses.push_back(poses_imu.at(i));
  }
  pub_pathimu.publish(arrIMU);

  // Move them forward in time
  poses_seq_imu++;
}

void ROS1Visualizer::publish_images() {

  // Return if we have already visualized
  if (_app->get_state() == nullptr)
    return;
  if (last_visualization_timestamp_image == _app->get_state()->_timestamp && _app->initialized())
    return;
  last_visualization_timestamp_image = _app->get_state()->_timestamp;

  // Check if we have subscribers
  if (it_pub_tracks.getNumSubscribers() != 0) {

    // Get our image of history tracks
    cv::Mat img_history = _app->get_historical_viz_image();
    if (!img_history.empty()) {

      // Create our message
      std_msgs::Header header;
      header.stamp = ros::Time::now();
      header.frame_id = "cam0";
      sensor_msgs::ImagePtr exl_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

      // Publish
      it_pub_tracks.publish(exl_msg);
    }
  }

  // Check if we have subscribers
  if (it_pub_tracksplanenorm.getNumSubscribers() != 0) {

    // Get our image of history tracks
    cv::Mat img_history = _app->get_historical_viz_image_tri();
    if (!img_history.empty()) {

      // Create our message
      std_msgs::Header header;
      header.stamp = ros::Time::now();
      header.frame_id = "cam0";
      sensor_msgs::ImagePtr exl_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

      // Publish
      it_pub_tracksplanenorm.publish(exl_msg);
    }
  }

  // Check if we have subscribers
  if (it_pub_tracksplane.getNumSubscribers() != 0) {

    // Get our image of history tracks
    cv::Mat img_history = _app->get_historical_viz_image_plane();
    if (!img_history.empty()) {

      // Create our message
      std_msgs::Header header;
      header.stamp = ros::Time::now();
      header.frame_id = "cam0";
      sensor_msgs::ImagePtr exl_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

      // Publish
      it_pub_tracksplane.publish(exl_msg);
    }
  }
}

void ROS1Visualizer::publish_features() {

  // Check if we have subscribers
  if (pub_points_msckf.getNumSubscribers() == 0 && pub_points_slam.getNumSubscribers() == 0 && pub_points_aruco.getNumSubscribers() == 0 &&
      pub_points_sim.getNumSubscribers() == 0 && pub_points_tracker.getNumSubscribers() == 0)
    return;

  // Get our good MSCKF features
  std::vector<Eigen::Vector3d> feats_msckf = _app->get_good_features_MSCKF();
  sensor_msgs::PointCloud2 cloud = ROSVisualizerHelper::get_ros_pointcloud(feats_msckf);
  pub_points_msckf.publish(cloud);

  // Get our good SLAM features
  std::vector<Eigen::Vector3d> feats_slam = _app->get_features_SLAM();
  sensor_msgs::PointCloud2 cloud_SLAM = ROSVisualizerHelper::get_ros_pointcloud(feats_slam);
  pub_points_slam.publish(cloud_SLAM);

  // Get our good ARUCO features
  std::vector<Eigen::Vector3d> feats_aruco = _app->get_features_ARUCO();
  sensor_msgs::PointCloud2 cloud_ARUCO = ROSVisualizerHelper::get_ros_pointcloud(feats_aruco);
  pub_points_aruco.publish(cloud_ARUCO);

  // Publish recovered features in our feat tracker
  std::shared_ptr<TrackPlane> trackPlane = std::dynamic_pointer_cast<TrackPlane>(_app->get_feattrack());
  if (trackPlane != nullptr) {
    std::vector<Eigen::Vector3d> feats_tracker;
    for (auto const &tmp : trackPlane->get_features())
      feats_tracker.push_back(tmp.second);
    sensor_msgs::PointCloud2 cloud_TRACK = ROSVisualizerHelper::get_ros_pointcloud(feats_tracker);
    pub_points_tracker.publish(cloud_TRACK);
  }

  // Get our good SIMULATION features
  if (_sim != nullptr) {
    std::map<size_t, std::vector<Eigen::Vector3d>> feats_plane;
    for (auto const &feat : _sim->get_map_vec()) {
      assert(feat.rows() == 4);
      if ((int)feat(3) == -1) {
        feats_plane[0].push_back(feat.block(0, 0, 3, 1));
      } else {
        feats_plane[(size_t)feat(3)].push_back(feat.block(0, 0, 3, 1));
      }
    }
    std::map<size_t, Eigen::Vector3d> colors;
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (auto const &feat : feats_plane) {
      std::mt19937_64 rng(feat.first);
      Eigen::Vector3d color = Eigen::Vector3d::Zero();
      while (color.norm() < 0.8)
        color << unif(rng), unif(rng), unif(rng);
      colors[feat.first] = color;
    }
    sensor_msgs::PointCloud2 cloud_SIM = ROSVisualizerHelper::get_ros_pointcloud(feats_plane, colors);
    pub_points_sim.publish(cloud_SIM);
  }
}

void ROS1Visualizer::publish_groundtruth() {

  // Our groundtruth state
  Eigen::Matrix<double, 17, 1> state_gt;

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = _app->get_state()->_timestamp + t_ItoC;

  // Check that we have the timestamp in our GT file [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
  if (_sim == nullptr && (gt_states.empty() || !DatasetReader::get_gt_state(timestamp_inI, state_gt, gt_states))) {
    return;
  }

  // Get the simulated groundtruth
  // NOTE: we get the true time in the IMU clock frame
  if (_sim != nullptr) {
    timestamp_inI = _app->get_state()->_timestamp + _sim->get_true_parameters().calib_camimu_dt;
    if (!_sim->get_state(timestamp_inI, state_gt))
      return;
  }

  // Get the GT and system state state
  Eigen::Matrix<double, 16, 1> state_ekf = _app->get_state()->_imu->value();

  // Create pose of IMU
  geometry_msgs::PoseStamped poseIinM;
  poseIinM.header.stamp = ros::Time(timestamp_inI);
  poseIinM.header.seq = poses_seq_gt;
  poseIinM.header.frame_id = "global";
  poseIinM.pose.orientation.x = state_gt(1, 0);
  poseIinM.pose.orientation.y = state_gt(2, 0);
  poseIinM.pose.orientation.z = state_gt(3, 0);
  poseIinM.pose.orientation.w = state_gt(4, 0);
  poseIinM.pose.position.x = state_gt(5, 0);
  poseIinM.pose.position.y = state_gt(6, 0);
  poseIinM.pose.position.z = state_gt(7, 0);
  pub_posegt.publish(poseIinM);

  // Append to our pose vector
  poses_gt.push_back(poseIinM);

  // Create our path (imu)
  // NOTE: We downsample the number of poses as needed to prevent rviz crashes
  // NOTE: https://github.com/ros-visualization/rviz/issues/1107
  nav_msgs::Path arrIMU;
  arrIMU.header.stamp = ros::Time::now();
  arrIMU.header.seq = poses_seq_gt;
  arrIMU.header.frame_id = "global";
  for (size_t i = 0; i < poses_gt.size(); i += std::floor((double)poses_gt.size() / 16384.0) + 1) {
    arrIMU.poses.push_back(poses_gt.at(i));
  }
  pub_pathgt.publish(arrIMU);

  // Move them forward in time
  poses_seq_gt++;

  // Publish our transform on TF
  tf::StampedTransform trans;
  trans.stamp_ = ros::Time::now();
  trans.frame_id_ = "global";
  trans.child_frame_id_ = "truth";
  tf::Quaternion quat(state_gt(1, 0), state_gt(2, 0), state_gt(3, 0), state_gt(4, 0));
  trans.setRotation(quat);
  tf::Vector3 orig(state_gt(5, 0), state_gt(6, 0), state_gt(7, 0));
  trans.setOrigin(orig);
  if (publish_global2imu_tf) {
    mTfBr->sendTransform(trans);
  }

  //==========================================================================
  //==========================================================================

  // Difference between positions
  double dx = state_ekf(4, 0) - state_gt(5, 0);
  double dy = state_ekf(5, 0) - state_gt(6, 0);
  double dz = state_ekf(6, 0) - state_gt(7, 0);
  double err_pos = std::sqrt(dx * dx + dy * dy + dz * dz);

  // Quaternion error
  Eigen::Matrix<double, 4, 1> quat_gt, quat_st, quat_diff;
  quat_gt << state_gt(1, 0), state_gt(2, 0), state_gt(3, 0), state_gt(4, 0);
  quat_st << state_ekf(0, 0), state_ekf(1, 0), state_ekf(2, 0), state_ekf(3, 0);
  quat_diff = quat_multiply(quat_st, Inv(quat_gt));
  double err_ori = (180 / M_PI) * 2 * quat_diff.block(0, 0, 3, 1).norm();

  //==========================================================================
  //==========================================================================

  // Get covariance of pose
  std::vector<std::shared_ptr<Type>> statevars;
  statevars.push_back(_app->get_state()->_imu->q());
  statevars.push_back(_app->get_state()->_imu->p());
  Eigen::Matrix<double, 6, 6> covariance = StateHelper::get_marginal_covariance(_app->get_state(), statevars);

  // Calculate NEES values
  // NOTE: need to manually multiply things out to make static asserts work
  // NOTE: https://github.com/rpng/open_vins/pull/226
  // NOTE: https://github.com/rpng/open_vins/issues/236
  // NOTE: https://gitlab.com/libeigen/eigen/-/issues/1664
  Eigen::Vector3d quat_diff_vec = quat_diff.block(0, 0, 3, 1);
  Eigen::Vector3d cov_vec = covariance.block(0, 0, 3, 3).inverse() * 2 * quat_diff.block(0, 0, 3, 1);
  double ori_nees = 2 * quat_diff_vec.dot(cov_vec);
  Eigen::Vector3d errpos = state_ekf.block(4, 0, 3, 1) - state_gt.block(5, 0, 3, 1);
  double pos_nees = errpos.transpose() * covariance.block(3, 3, 3, 3).inverse() * errpos;

  //==========================================================================
  //==========================================================================

  // Update our average variables
  if (!std::isnan(ori_nees) && !std::isnan(pos_nees)) {
    summed_mse_ori += err_ori * err_ori;
    summed_mse_pos += err_pos * err_pos;
    summed_nees_ori += ori_nees;
    summed_nees_pos += pos_nees;
    summed_number++;
  }

  // Nice display for the user
  PRINT_INFO(REDPURPLE "error to gt => %.3f, %.3f (deg,m) | rmse => %.3f, %.3f (deg,m) | called %d times\n" RESET, err_ori, err_pos,
             std::sqrt(summed_mse_ori / summed_number), std::sqrt(summed_mse_pos / summed_number), (int)summed_number);
  PRINT_INFO(REDPURPLE "nees => %.1f, %.1f (ori,pos) | avg nees = %.1f, %.1f (ori,pos)\n" RESET, ori_nees, pos_nees,
             summed_nees_ori / summed_number, summed_nees_pos / summed_number);

  // Debug exit if bad...
  //  if (summed_nees_ori / summed_number > 5.0 || summed_nees_pos / summed_number > 5.0) {
  //    PRINT_ERROR(BOLDRED " too big of average NEES, exiting....\n" RESET);
  //    std::exit(EXIT_FAILURE);
  //  }

  //==========================================================================
  //==========================================================================
}

void ROS1Visualizer::publish_loopclosure_information() {

  // Get the current tracks in this frame
  double active_tracks_time1 = -1;
  double active_tracks_time2 = -1;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  cv::Mat active_cam0_image;
  _app->get_active_tracks(active_tracks_time1, active_tracks_posinG, active_tracks_uvd);
  _app->get_active_image(active_tracks_time2, active_cam0_image);
  if (active_tracks_time1 == -1)
    return;
  if (_app->get_state()->_clones_IMU.find(active_tracks_time1) == _app->get_state()->_clones_IMU.end())
    return;
  Eigen::Vector4d quat = _app->get_state()->_clones_IMU.at(active_tracks_time1)->quat();
  Eigen::Vector3d pos = _app->get_state()->_clones_IMU.at(active_tracks_time1)->pos();
  if (active_tracks_time1 != active_tracks_time2)
    return;

  // Default header
  std_msgs::Header header;
  header.stamp = ros::Time(active_tracks_time1);

  //======================================================
  // Check if we have subscribers for the pose odometry, camera intrinsics, or extrinsics
  if (pub_loop_pose.getNumSubscribers() != 0 || pub_loop_extrinsic.getNumSubscribers() != 0 ||
      pub_loop_intrinsics.getNumSubscribers() != 0) {

    // PUBLISH HISTORICAL POSE ESTIMATE
    nav_msgs::Odometry odometry_pose;
    odometry_pose.header = header;
    odometry_pose.header.frame_id = "global";
    odometry_pose.pose.pose.position.x = pos(0);
    odometry_pose.pose.pose.position.y = pos(1);
    odometry_pose.pose.pose.position.z = pos(2);
    odometry_pose.pose.pose.orientation.x = quat(0);
    odometry_pose.pose.pose.orientation.y = quat(1);
    odometry_pose.pose.pose.orientation.z = quat(2);
    odometry_pose.pose.pose.orientation.w = quat(3);
    pub_loop_pose.publish(odometry_pose);

    // PUBLISH IMU TO CAMERA0 EXTRINSIC
    // need to flip the transform to the IMU frame
    Eigen::Vector4d q_ItoC = _app->get_state()->_calib_IMUtoCAM.at(0)->quat();
    Eigen::Vector3d p_CinI = -_app->get_state()->_calib_IMUtoCAM.at(0)->Rot().transpose() * _app->get_state()->_calib_IMUtoCAM.at(0)->pos();
    nav_msgs::Odometry odometry_calib;
    odometry_calib.header = header;
    odometry_calib.header.frame_id = "imu";
    odometry_calib.pose.pose.position.x = p_CinI(0);
    odometry_calib.pose.pose.position.y = p_CinI(1);
    odometry_calib.pose.pose.position.z = p_CinI(2);
    odometry_calib.pose.pose.orientation.x = q_ItoC(0);
    odometry_calib.pose.pose.orientation.y = q_ItoC(1);
    odometry_calib.pose.pose.orientation.z = q_ItoC(2);
    odometry_calib.pose.pose.orientation.w = q_ItoC(3);
    pub_loop_extrinsic.publish(odometry_calib);

    // PUBLISH CAMERA0 INTRINSICS
    bool is_fisheye = (std::dynamic_pointer_cast<ov_core::CamEqui>(_app->get_params().camera_intrinsics.at(0)) != nullptr);
    sensor_msgs::CameraInfo cameraparams;
    cameraparams.header = header;
    cameraparams.header.frame_id = "cam0";
    cameraparams.distortion_model = is_fisheye ? "equidistant" : "plumb_bob";
    Eigen::VectorXd cparams = _app->get_state()->_cam_intrinsics.at(0)->value();
    cameraparams.D = {cparams(4), cparams(5), cparams(6), cparams(7)};
    cameraparams.K = {cparams(0), 0, cparams(2), 0, cparams(1), cparams(3), 0, 0, 1};
    pub_loop_intrinsics.publish(cameraparams);
  }

  //======================================================
  // PUBLISH FEATURE TRACKS IN THE GLOBAL FRAME OF REFERENCE
  if (pub_loop_point.getNumSubscribers() != 0) {

    // Construct the message
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;
    point_cloud.header.frame_id = "global";
    for (const auto &feattimes : active_tracks_posinG) {

      // Get this feature information
      size_t featid = feattimes.first;
      Eigen::Vector3d uvd = Eigen::Vector3d::Zero();
      if (active_tracks_uvd.find(featid) != active_tracks_uvd.end()) {
        uvd = active_tracks_uvd.at(featid);
      }
      Eigen::Vector3d pFinG = active_tracks_posinG.at(featid);

      // Push back 3d point
      geometry_msgs::Point32 p;
      p.x = pFinG(0);
      p.y = pFinG(1);
      p.z = pFinG(2);
      point_cloud.points.push_back(p);

      // Push back the uv_norm, uv_raw, and feature id
      // NOTE: we don't use the normalized coordinates to save time here
      // NOTE: they will have to be re-normalized in the loop closure code
      sensor_msgs::ChannelFloat32 p_2d;
      p_2d.values.push_back(0);
      p_2d.values.push_back(0);
      p_2d.values.push_back(uvd(0));
      p_2d.values.push_back(uvd(1));
      p_2d.values.push_back(featid);
      point_cloud.channels.push_back(p_2d);
    }
    pub_loop_point.publish(point_cloud);
  }

  //======================================================
  // Depth images of sparse points and its colorized version
  if (it_pub_loop_img_depth.getNumSubscribers() != 0 || it_pub_loop_img_depth_color.getNumSubscribers() != 0) {

    // Create the images we will populate with the depths
    std::pair<int, int> wh_pair = {active_cam0_image.cols, active_cam0_image.rows};
    cv::Mat depthmap = cv::Mat::zeros(wh_pair.second, wh_pair.first, CV_16UC1);
    cv::Mat depthmap_viz = active_cam0_image;

    // Loop through all points and append
    for (const auto &feattimes : active_tracks_uvd) {

      // Get this feature information
      size_t featid = feattimes.first;
      Eigen::Vector3d uvd = active_tracks_uvd.at(featid);

      // Skip invalid points
      double dw = 4;
      if (uvd(0) < dw || uvd(0) > wh_pair.first - dw || uvd(1) < dw || uvd(1) > wh_pair.second - dw) {
        continue;
      }

      // Append the depth
      // NOTE: scaled by 1000 to fit the 16U
      // NOTE: access order is y,x (stupid opencv convention stuff)
      depthmap.at<uint16_t>((int)uvd(1), (int)uvd(0)) = (uint16_t)(1000 * uvd(2));

      // Taken from LSD-SLAM codebase segment into 0-4 meter segments:
      // https://github.com/tum-vision/lsd_slam/blob/d1e6f0e1a027889985d2e6b4c0fe7a90b0c75067/lsd_slam_core/src/util/globalFuncs.cpp#L87-L96
      float id = 1.0f / (float)uvd(2);
      float r = (0.0f - id) * 255 / 1.0f;
      if (r < 0)
        r = -r;
      float g = (1.0f - id) * 255 / 1.0f;
      if (g < 0)
        g = -g;
      float b = (2.0f - id) * 255 / 1.0f;
      if (b < 0)
        b = -b;
      uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
      uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
      uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);
      cv::Scalar color(255 - rc, 255 - gc, 255 - bc);

      // Small square around the point (note the above bound check needs to take into account this width)
      cv::Point p0(uvd(0) - dw, uvd(1) - dw);
      cv::Point p1(uvd(0) + dw, uvd(1) + dw);
      cv::rectangle(depthmap_viz, p0, p1, color, -1);
    }

    // Create our messages
    header.frame_id = "cam0";
    sensor_msgs::ImagePtr exl_msg1 = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_16UC1, depthmap).toImageMsg();
    it_pub_loop_img_depth.publish(exl_msg1);
    header.stamp = ros::Time::now();
    header.frame_id = "cam0";
    sensor_msgs::ImagePtr exl_msg2 = cv_bridge::CvImage(header, "bgr8", depthmap_viz).toImageMsg();
    it_pub_loop_img_depth_color.publish(exl_msg2);
  }
}

void ROS1Visualizer::publish_planes() {

  // return if no subs
  if (pub_plane_slam.getNumSubscribers() == 0 && it_pub_ardisp.getNumSubscribers() == 0 && pub_plane_slam_map.getNumSubscribers() == 0 &&
      pub_plane_sim.getNumSubscribers() == 0)
    return;

  //============================================================================
  // active slam planes
  // TODO: do not recompute the projection each time (slow for realtime ops)
  //============================================================================
  if (pub_plane_slam.getNumSubscribers() != 0 || it_pub_ardisp.getNumSubscribers() != 0) {

    // Generate plane colors
    std::map<size_t, Eigen::Vector3d> colors;
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (auto const &planepair : _app->get_good_features_PLANE_SLAM()) {
      std::mt19937_64 rng(planepair.first);
      Eigen::Vector3d color = Eigen::Vector3d::Zero();
      while (color.norm() < 0.8)
        color << unif(rng), unif(rng), unif(rng);
      colors[planepair.first] = color;
    }

    // Get the current image (cam0) we can render in
    bool render_ar = (it_pub_ardisp.getNumSubscribers() != 0);
    double active_tracks_time1 = -1;
    cv::Mat active_cam0_image;
    _app->get_active_image_raw(active_tracks_time1, active_cam0_image);
    if (active_tracks_time1 == -1 || active_cam0_image.empty())
      return;
    if (_app->get_state()->_clones_IMU.find(active_tracks_time1) == _app->get_state()->_clones_IMU.end())
      return;

    // TODO: support fisheye undistortion
    if (std::dynamic_pointer_cast<ov_core::CamRadtan>(_app->get_state()->_cam_intrinsics_cameras.at(0)) == nullptr) {
      PRINT_WARNING(YELLOW "ar-render only supports radtan camera, skipping...\n" RESET);
      return;
    }

    // Undistort the image
    // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d
    cv::Mat img_ar;
    cv::Matx33d K = _app->get_state()->_cam_intrinsics_cameras.at(0)->get_K();
    cv::Vec4d D = _app->get_state()->_cam_intrinsics_cameras.at(0)->get_D();
    if (render_ar) {
      cv::undistort(active_cam0_image, img_ar, K, D, K);
      cv::cvtColor(img_ar, img_ar, cv::COLOR_GRAY2RGB);
    }

    // Helper function that transforms a feature in global into local camera image
    // TODO: maybe this should return depth so we can sort things?
    Eigen::Matrix3d R_GtoI = _app->get_state()->_clones_IMU.at(active_tracks_time1)->Rot();
    Eigen::Vector3d p_IinG = _app->get_state()->_clones_IMU.at(active_tracks_time1)->pos();
    Eigen::Matrix3d R_ItoC = _app->get_state()->_calib_IMUtoCAM.at(0)->Rot();
    Eigen::Vector3d p_IinC = _app->get_state()->_calib_IMUtoCAM.at(0)->pos();
    auto global2image = [&](const Eigen::Vector3d &p_FinG, cv::Matx33d K, cv::Point2f &pt) {
      // Transform into the camera frame
      Eigen::Vector3d p_FinC = R_ItoC * R_GtoI * (p_FinG - p_IinG) + p_IinC;
      if (p_FinC(2) < 0.01)
        return false;
      // Image space
      cv::Point3d uv_n, uv_r;
      uv_n.x = (p_FinC(0) / p_FinC(2));
      uv_n.y = (p_FinC(1) / p_FinC(2));
      uv_n.z = 1.0;
      uv_r = K * uv_n;
      uv_r /= uv_r.z;
      // return the raw point
      pt.x = (float)uv_r.x;
      pt.y = (float)uv_r.y;
      return true;
    };
    auto draw_on_image = [](cv::Mat &img, const cv::Point2f &pt1, const cv::Point2f &pt2, Eigen::Vector3d color, int thick = 2) {
      cv::line(img, pt1, pt2, cv::Scalar(color(0), color(1), color(2)), thick);
    };

    // Load example object (downscaled with meshlab)
    // https://groups.csail.mit.edu/graphics/classes/6.837/F03/models/
    // TODO: why do we need this extra rotation of the model's coordinate frame
    static std::shared_ptr<Model> model = nullptr;
    static Eigen::Matrix3d R_model = Eigen::Matrix3d::Identity();
    if (render_ar && boost::filesystem::exists(path_to_object) && model == nullptr) {
      model = std::make_shared<Model>(path_to_object.c_str());
      R_model = ov_core::rot_x(-90.0 * M_PI / 180.0);
    }

    // Lets get all the planes
    visualization_msgs::MarkerArray marker_arr;
    for (auto const &planepair : _app->get_good_features_PLANE_SLAM()) {

      // Get the plane id and 3D points that are on the plane
      size_t planeid = planepair.first;
      auto featmap = planepair.second;

      // Our plane will be a line list
      visualization_msgs::Marker marker_plane;
      marker_plane.header.frame_id = "global";
      marker_plane.header.stamp = ros::Time::now();
      marker_plane.ns = "slam_planes";
      marker_plane.id = (int)planeid;
      marker_plane.type = visualization_msgs::Marker::LINE_LIST; // TRIANGLE_LIST
      marker_plane.action = visualization_msgs::Marker::MODIFY;
      marker_plane.scale.x = 0.01;
      marker_plane.pose.orientation.w = 1.0;
      marker_plane.color.r = (float)colors.at(planeid)(0);
      marker_plane.color.g = (float)colors.at(planeid)(1);
      marker_plane.color.b = (float)colors.at(planeid)(2);
      marker_plane.color.a = 1.0;

      // Skip any who has been marginalized from the state
      if (_app->get_state()->_features_PLANE.find(planeid) == _app->get_state()->_features_PLANE.end()) {
        marker_arr.markers.push_back(marker_plane);
        continue;
      }

      // If we have a lot of points, then we should try to downsample them
      // This should speed up the rendering of the AR and the meshing...
      if (featmap.size() > 200) {

        // Create a vector of points (IKD tree input form)
        KD_TREE<ikdTree_PointType>::PointVector vec_points;
        vec_points.reserve(featmap.size());
        for (const auto &feat : featmap) {
          ikdTree_PointType p;
          p.x = (float)feat.second[0];
          p.y = (float)feat.second[1];
          p.z = (float)feat.second[2];
          p.id = feat.first;
          vec_points.push_back(p);
        }

        // Feature position in current camera
        // Normalize the depth of the feature to the range [0.2, 5.0]
        // Then interpolate between our min and max allowed downsampling scale value (meters)
        double depth = 5.0;
        for (const auto &feat : featmap) {
          Eigen::Vector3d p_FinCi = R_ItoC * R_GtoI * (feat.second - p_IinG) + p_IinC;
          double depthi = std::max(0.2, std::min(5.0, p_FinCi(2)));
          depth = std::min(depth, depthi);
        }
        double lambda = (depth - 0.2) / (5.0 - 0.2);
        // double scale = (1.0 - lambda) * 0.01 + lambda * 0.25;
        double scale = 0.02;
        if (lambda < 0.1)
          scale = 0.04;
        else if (lambda < 0.3)
          scale = 0.08;
        else if (lambda < 0.4)
          scale = 0.10;
        else if (lambda < 0.5)
          scale = 0.12;
        else if (lambda < 0.8)
          scale = 0.18;
        else if (lambda < 0.9)
          scale = 0.24;
        else
          scale = 0.30;

        // Clean IKD-Tree and build with a new pointcloud
        // NOTE: it seems that creating the ikd each time can takes >10ms
        // NOTE: thus we go through this process of clearing the old one and re-using
        if (ikd_tree->initialized())
          ikd_tree->reset();
        ikd_tree->set_downsample_param((float)scale);
        ikd_tree->Build({vec_points.at(0)});
        ikd_tree->Add_Points(vec_points, true);
        KD_TREE<ikdTree_PointType>::PointVector vec_points_return;
        ikd_tree->flatten(ikd_tree->Root_Node, vec_points_return, NOT_RECORD);

        // Recreate the feature map using returned downsampled point cloud
        featmap.clear();
        for (auto const &point : vec_points_return) {
          Eigen::Vector3d feat;
          feat << point.x, point.y, point.z;
          featmap.insert({point.id, feat});
        }
      }

      // Get estimate of plane
      Eigen::Vector3d cp_inG = _app->get_state()->_features_PLANE.at(planeid)->value().block(0, 0, 3, 1);
      Eigen::Matrix3d R_GtoPi;
      ov_init::InitializerHelper::gram_schmidt(cp_inG / cp_inG.norm(), R_GtoPi);
      R_GtoPi = R_GtoPi.transpose().eval();

      // Project all points onto the plane so we can triangulate them
      // NOTE: we skip points that are close to each other as this causes issues with CDT library..
      // TODO: this too close logic is really slow for many features on the planes....
      // TODO: we should really downsample for visualization using a KD tree!!!
      std::vector<size_t> feat2d_id;
      std::vector<CDT::V2d<float>> feat2d_verts;
      for (auto const &feat : featmap) {
        Eigen::Vector3d p_FinPi = R_GtoPi * (feat.second - cp_inG);
        bool too_close = false;
        for (auto const &tmp : feat2d_verts) {
          if (std::abs(tmp.x - (float)p_FinPi(0)) < 0.01 && std::abs(tmp.y - (float)p_FinPi(1)) < 0.01) {
            too_close = true;
            break;
          }
        }
        if (!too_close) {
          //  std::cout << planeid << "P -> " << feat.first << " - " << p_FinPi.transpose() << std::endl;
          feat2d_id.emplace_back(feat.first);
          feat2d_verts.emplace_back(CDT::V2d<float>::make((float)p_FinPi(0), (float)p_FinPi(1)));
        }
      }

      // Skip if not enough features
      if (feat2d_id.size() < 3) {
        marker_arr.markers.push_back(marker_plane);
        continue;
      }

      // Now lets perform our Delaunay Triangulation
      // https://github.com/artem-ogre/CDT
      CDT::TriangleVec triangles;
      try {
        CDT::Triangulation<float> cdt(CDT::VertexInsertionOrder::Enum::AsProvided);
        cdt.insertVertices(feat2d_verts);
        cdt.eraseSuperTriangle();
        triangles = cdt.triangles;
      } catch (...) {
        marker_arr.markers.push_back(marker_plane);
        continue;
      }

      // Skip if no triangles
      if (triangles.empty()) {
        marker_arr.markers.push_back(marker_plane);
        continue;
      }

      // Done lets visualize
      for (auto const &tri : triangles) {

        // assert that we don't have any invalid..
        assert(tri.vertices[0] != CDT::noVertex);
        assert(tri.vertices[1] != CDT::noVertex);
        assert(tri.vertices[2] != CDT::noVertex);
        size_t id1 = feat2d_id[tri.vertices[0]];
        size_t id2 = feat2d_id[tri.vertices[1]];
        size_t id3 = feat2d_id[tri.vertices[2]];

        // Push back lines connecting all of them
        // Visualize the 3D mesh in rviz
        auto eigen2geo = [](Eigen::Vector3d feat) {
          geometry_msgs::Point pt;
          pt.x = feat(0);
          pt.y = feat(1);
          pt.z = feat(2);
          return pt;
        };
        marker_plane.points.push_back(eigen2geo(featmap.at(id1)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id2)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id2)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id3)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id1)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id3)));
        if (marker_plane.points.empty()) {
          std::exit(EXIT_FAILURE);
        }

        // Render in our AR image
        // Draw the 3D points on the 2D image
        if (render_ar) {
          cv::Point2f pt1, pt2, pt3;
          auto pt1g = global2image(featmap.at(id1), K, pt1);
          auto pt2g = global2image(featmap.at(id2), K, pt2);
          auto pt3g = global2image(featmap.at(id3), K, pt3);
          if (pt1g && pt2g)
            draw_on_image(img_ar, pt1, pt2, 255.0 * colors.at(planeid));
          if (pt2g && pt3g)
            draw_on_image(img_ar, pt2, pt3, 255.0 * colors.at(planeid));
          if (pt1g && pt3g)
            draw_on_image(img_ar, pt3, pt1, 255.0 * colors.at(planeid));
        }
      }
      if (!marker_plane.points.empty()) {
        marker_arr.markers.push_back(marker_plane);
      }
    }
    if (!marker_arr.markers.empty()) {
      pub_plane_slam.publish(marker_arr);
    }

    // Render the model on each plane
    if (render_ar) {
      for (auto const &planepair : _app->get_good_features_PLANE_SLAM()) {

        // Data
        size_t planeid = planepair.first;
        auto featmap = planepair.second;

        // Skip any who has been marginalized from the state
        if (_app->get_state()->_features_PLANE.find(planeid) == _app->get_state()->_features_PLANE.end())
          continue;

        // Get estimate of plane
        Eigen::Vector3d cp_inG = _app->get_state()->_features_PLANE.at(planeid)->value().block(0, 0, 3, 1);
        Eigen::Matrix3d R_GtoPi;
        ov_init::InitializerHelper::gram_schmidt(cp_inG / cp_inG.norm(), R_GtoPi);
        R_GtoPi = R_GtoPi.transpose().eval();
        auto plane2image = [&](const Eigen::Vector3d &p_FinPi, cv::Matx33d K, cv::Point2f &pt) {
          // We fix the AR to be in the middle of the plane
          if (ar_location_inPi.find(planeid) == ar_location_inPi.end()) {
            Eigen::Vector3d p_FinG_avg = Eigen::Vector3d::Zero();
            int count = 0;
            for (auto const &feat : featmap) {
              p_FinG_avg += feat.second;
              count++;
            }
            p_FinG_avg /= (double)count;
            // ar_location_inPi[planeid] = p_FinG_avg;
            ar_location_inPi[planeid] = p_FinG_avg - cp_inG;
            Eigen::Vector3d p_FinCi = R_ItoC * R_GtoI * (p_FinG_avg - p_IinG) + p_IinC;
            Eigen::Vector3d norm_inCi = R_ItoC * R_GtoI * (cp_inG / cp_inG.norm());
            ar_flip_inPi[planeid] = (norm_inCi.dot(p_FinCi) < 0) ? -1.0 : 1.0;
            ar_rotation_inPi[planeid] = R_GtoPi;
          }
          // Transform into the camera frame
          Eigen::Matrix3d R = ar_flip_inPi.at(planeid) * ar_rotation_inPi.at(planeid).transpose() * R_model;
          Eigen::Vector3d p_FinG = R * p_FinPi + ar_location_inPi.at(planeid) + cp_inG;
          Eigen::Vector3d p_FinC = R_ItoC * R_GtoI * (p_FinG - p_IinG) + p_IinC;
          if (p_FinC(2) < 0.01)
            return false;
          // Image space
          cv::Point3d uv_n, uv_r;
          uv_n.x = (p_FinC(0) / p_FinC(2));
          uv_n.y = (p_FinC(1) / p_FinC(2));
          uv_n.z = 1.0;
          uv_r = K * uv_n;
          uv_r /= uv_r.z;
          // return the raw point
          pt.x = (float)uv_r.x;
          pt.y = (float)uv_r.y;
          return true;
        };

        // If I have the model then render it!
        if (model != nullptr) {
          for (int i = 0; i < model->nfaces(); i++) {
            std::vector<int> face = model->face(i);
            for (int j = 0; j < 3; j++) {
              cv::Point2f pt1, pt2;
              auto pt1g = plane2image(model->vert(face[j]), K, pt1);
              auto pt2g = plane2image(model->vert(face[(j + 1) % 3]), K, pt2);
              if (pt1g && pt2g)
                draw_on_image(img_ar, pt1, pt2, 100.0 * colors.at(planeid), 1);
            }
          }
        }
      }

      // Display a rendered AR image
      std_msgs::Header header;
      header.stamp = ros::Time(active_tracks_time1);
      header.frame_id = "cam0";
      sensor_msgs::ImagePtr exl_msg1 = cv_bridge::CvImage(header, "rgb8", img_ar).toImageMsg();
      it_pub_ardisp.publish(exl_msg1);
    }
  }

  //============================================================================
  // active slam planes (MAP)
  // TODO: do not recompute the projection each time (slow for realtime ops)
  //============================================================================

  // Append newest estimates
  for (auto const &planepair : _app->get_good_features_PLANE_SLAM()) {
    if (good_features_PLANE_SLAM_MAP.find(planepair.first) == good_features_PLANE_SLAM_MAP.end()) {
      good_features_PLANE_SLAM_MAP[planepair.first] = planepair.second;
    } else {
      for (auto const &featpair : planepair.second) {
        good_features_PLANE_SLAM_MAP[planepair.first][featpair.first] = featpair.second;
      }
    }
  }

  // Record current cp planes from the state
  for (auto const &type : _app->get_state()->_features_PLANE) {
    features_PLANE_MAP[type.first] = type.second->value().block(0, 0, 3, 1);
  }

  // Now display them if requested
  if (pub_plane_slam_map.getNumSubscribers() != 0) {

    // Generate plane colors
    std::map<size_t, Eigen::Vector3d> colors;
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (auto const &planepair : good_features_PLANE_SLAM_MAP) {
      std::mt19937_64 rng(planepair.first);
      Eigen::Vector3d color = Eigen::Vector3d::Zero();
      while (color.norm() < 0.8)
        color << unif(rng), unif(rng), unif(rng);
      colors[planepair.first] = color;
    }

    // Lets get all the planes
    visualization_msgs::MarkerArray marker_arr;
    for (auto const &planepair : good_features_PLANE_SLAM_MAP) {

      // Data
      size_t planeid = planepair.first;
      auto featmap = planepair.second;

      // Our plane will be a line list
      visualization_msgs::Marker marker_plane;
      marker_plane.header.frame_id = "global";
      marker_plane.header.stamp = ros::Time::now();
      marker_plane.ns = "slam_planes_map";
      marker_plane.id = (int)planeid;
      marker_plane.type = visualization_msgs::Marker::LINE_LIST; // TRIANGLE_LIST
      marker_plane.action = visualization_msgs::Marker::MODIFY;
      marker_plane.scale.x = 0.01;
      marker_plane.pose.orientation.w = 1.0;
      marker_plane.color.r = (float)colors.at(planeid)(0);
      marker_plane.color.g = (float)colors.at(planeid)(1);
      marker_plane.color.b = (float)colors.at(planeid)(2);
      marker_plane.color.a = 1.0;

      // Skip any who has been marginalized from the state
      if (features_PLANE_MAP.find(planeid) == features_PLANE_MAP.end()) {
        marker_arr.markers.push_back(marker_plane);
        continue;
      }

      // If we have a lot of points, then we should try to downsample them
      // This should speed up the rendering of the AR and the meshing...
      if (featmap.size() > 200) {

        // Create a vector of points (IKD tree input form)
        KD_TREE<ikdTree_PointType>::PointVector vec_points;
        vec_points.reserve(featmap.size());
        for (const auto &feat : featmap) {
          ikdTree_PointType p;
          p.x = (float)feat.second[0];
          p.y = (float)feat.second[1];
          p.z = (float)feat.second[2];
          p.id = feat.first;
          vec_points.push_back(p);
        }

        // Clean IKD-Tree and build with a new pointcloud
        // NOTE: it seems that creating the ikd each time can takes >10ms
        // NOTE: thus we go through this process of clearing the old one and re-using
        if (ikd_tree->initialized())
          ikd_tree->reset();
        ikd_tree->set_downsample_param(0.05); // m?
        ikd_tree->Build({vec_points.at(0)});
        ikd_tree->Add_Points(vec_points, true);
        KD_TREE<ikdTree_PointType>::PointVector vec_points_return;
        ikd_tree->flatten(ikd_tree->Root_Node, vec_points_return, NOT_RECORD);

        // Recreate the feature map using returned downsampled point cloud
        featmap.clear();
        for (auto const &point : vec_points_return) {
          Eigen::Vector3d feat;
          feat << point.x, point.y, point.z;
          featmap.insert({point.id, feat});
        }
      }

      // Get estimate of plane
      Eigen::Vector3d cp_inG = features_PLANE_MAP.at(planeid);
      Eigen::Matrix3d R_GtoPi;
      ov_init::InitializerHelper::gram_schmidt(cp_inG, R_GtoPi);
      R_GtoPi = R_GtoPi.transpose().eval();

      // Project all points onto the plane so we can triangulate them
      // NOTE: we skip points that are close to each other as this causes issues with CDT library..
      std::vector<size_t> feat2d_id;
      std::vector<CDT::V2d<float>> feat2d_verts;
      for (auto const &feat : featmap) {
        Eigen::Vector3d p_FinPi = R_GtoPi * (feat.second - cp_inG);
        bool too_close = false;
        for (auto const &tmp : feat2d_verts) {
          if (std::abs(tmp.x - (float)p_FinPi(0)) < 0.01 && std::abs(tmp.y - (float)p_FinPi(1)) < 0.01) {
            too_close = true;
            break;
          }
        }
        if (!too_close) {
          // std::cout << planeid << "P -> " << feat.first << " - " << p_FinPi.transpose() << std::endl;
          feat2d_id.emplace_back(feat.first);
          feat2d_verts.emplace_back(CDT::V2d<float>::make((float)p_FinPi(0), (float)p_FinPi(1)));
        }
      }

      // Skip if not enough features
      if (feat2d_id.size() < 3) {
        marker_arr.markers.push_back(marker_plane);
        continue;
      }

      // Now lets perform our Delaunay Triangulation
      // https://github.com/artem-ogre/CDT
      CDT::TriangleVec triangles;
      try {
        CDT::Triangulation<float> cdt(CDT::VertexInsertionOrder::Enum::AsProvided);
        cdt.insertVertices(feat2d_verts);
        cdt.eraseSuperTriangle();
        triangles = cdt.triangles;
      } catch (...) {
        marker_arr.markers.push_back(marker_plane);
        continue;
      }

      // Skip if no triangles
      if (triangles.empty()) {
        marker_arr.markers.push_back(marker_plane);
        continue;
      }

      // Done lets visualize
      for (auto const &tri : triangles) {

        // assert that we don't have any invalid..
        assert(tri.vertices[0] != CDT::noVertex);
        assert(tri.vertices[1] != CDT::noVertex);
        assert(tri.vertices[2] != CDT::noVertex);
        size_t id1 = feat2d_id[tri.vertices[0]];
        size_t id2 = feat2d_id[tri.vertices[1]];
        size_t id3 = feat2d_id[tri.vertices[2]];

        // Push back lines connecting all of them
        auto eigen2geo = [](Eigen::Vector3d feat) {
          geometry_msgs::Point pt;
          pt.x = feat(0);
          pt.y = feat(1);
          pt.z = feat(2);
          return pt;
        };
        marker_plane.points.push_back(eigen2geo(featmap.at(id1)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id2)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id2)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id3)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id1)));
        marker_plane.points.push_back(eigen2geo(featmap.at(id3)));
      }
      marker_arr.markers.push_back(marker_plane);
    }
    pub_plane_slam_map.publish(marker_arr);
  }

  //============================================================================
  // simulation map of planes
  //============================================================================

  if (_sim != nullptr) {

    // Given a plane, this converts it into the marker type we can vizualize in RVIZ
    auto convert_plane = [](visualization_msgs::Marker &marker_plane, const SimPlane &plane) {
      // Convert our 4 points to the right format
      geometry_msgs::Point pt_tl, pt_tr, pt_bl, pt_br;
      pt_tl.x = plane.pt_top_left(0);
      pt_tl.y = plane.pt_top_left(1);
      pt_tl.z = plane.pt_top_left(2);
      pt_tr.x = plane.pt_top_right(0);
      pt_tr.y = plane.pt_top_right(1);
      pt_tr.z = plane.pt_top_right(2);
      pt_bl.x = plane.pt_bottom_left(0);
      pt_bl.y = plane.pt_bottom_left(1);
      pt_bl.z = plane.pt_bottom_left(2);
      pt_br.x = plane.pt_bottom_right(0);
      pt_br.y = plane.pt_bottom_right(1);
      pt_br.z = plane.pt_bottom_right(2);

      // Add the 4 bounding points
      marker_plane.points.push_back(pt_tl);
      marker_plane.points.push_back(pt_tr);
      marker_plane.points.push_back(pt_tr);
      marker_plane.points.push_back(pt_br);
      marker_plane.points.push_back(pt_br);
      marker_plane.points.push_back(pt_bl);
      marker_plane.points.push_back(pt_bl);
      marker_plane.points.push_back(pt_tl);

      // Add cross across middle of the plane
      marker_plane.points.push_back(pt_tl);
      marker_plane.points.push_back(pt_br);
      marker_plane.points.push_back(pt_tr);
      marker_plane.points.push_back(pt_bl);
    };

    // Our marker array
    visualization_msgs::MarkerArray marker_arr;

    // Lets get all the planes
    int ct = 0;
    for (auto &plane : _sim->get_planes()) {

      // Our plane will be a line list
      visualization_msgs::Marker marker_plane;
      marker_plane.header.frame_id = "global";
      marker_plane.header.stamp = ros::Time::now();
      marker_plane.ns = "sim_planes";
      marker_plane.id = ct;
      marker_plane.type = visualization_msgs::Marker::LINE_LIST;
      marker_plane.action = visualization_msgs::Marker::MODIFY;
      marker_plane.scale.x = 0.06;
      marker_plane.color.b = 1.0;
      marker_plane.color.a = 1.0;
      convert_plane(marker_plane, plane);

      // Append and move plane count forward
      marker_arr.markers.push_back(marker_plane);
      ct++;
    }
    pub_plane_sim.publish(marker_arr);
  }
}

void ROS1Visualizer::publish_plane_information() {

  // Coloring with plane id
  std::map<size_t, std::vector<Eigen::Vector3d>> feats_plane_slam = _app->get_good_features_PLANE();
  std::map<size_t, Eigen::Vector3d> colors;
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  for (auto const &feat : feats_plane_slam) {
    std::mt19937_64 rng(feat.first);
    Eigen::Vector3d color = Eigen::Vector3d::Zero();
    while (color.norm() < 0.8)
      color << unif(rng), unif(rng), unif(rng);
    colors[feat.first] = color;
  }

  // Visualize our planar features
  if (pub_plane_points.getNumSubscribers() != 0 && !feats_plane_slam.empty()) {
    sensor_msgs::PointCloud2 cloud_PLANES = ROSVisualizerHelper::get_ros_pointcloud(feats_plane_slam, colors);
    pub_plane_points.publish(cloud_PLANES);
  }

  // Publish line markers connecting poses and update points!
  if (pub_plane_constraints.getNumSubscribers() != 0) {

    // Our line list for this robot
    visualization_msgs::Marker line_list;
    line_list.header.frame_id = "global";
    line_list.header.stamp = ros::Time::now();
    line_list.ns = "plane_matches";
    line_list.pose.orientation.w = 1.0;
    line_list.id = 0;
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    line_list.scale.x = 0.01; // 0.05;

    // Loop through each point, and append if we are able
    for (auto const &featpair : feats_plane_slam) {
      for (auto const &feat : featpair.second) {

        // Feature point
        geometry_msgs::Point p1;
        p1.x = feat(0);
        p1.y = feat(1);
        p1.z = feat(2);
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        p << p1.x, p1.y, p1.z;
        if (p.norm() == 0)
          continue;

        // Connect to current IMU pose
        geometry_msgs::Point p0;
        p0.x = _app->get_state()->_imu->pos()(0);
        p0.y = _app->get_state()->_imu->pos()(1);
        p0.z = _app->get_state()->_imu->pos()(2);
        line_list.points.push_back(p0);
        line_list.points.push_back(p1);
        std_msgs::ColorRGBA color;
        color.r = (float)colors.at(featpair.first)(0);
        color.g = (float)colors.at(featpair.first)(1);
        color.b = (float)colors.at(featpair.first)(2);
        color.a = 1.0f;
        line_list.colors.push_back(color);
        line_list.colors.push_back(color);
      }
    }
    pub_plane_constraints.publish(line_list);
  }
}
