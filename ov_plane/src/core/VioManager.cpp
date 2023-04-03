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

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackAruco.h"
#include "track/TrackSIM.h"
#include "track_plane/TrackPlane.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterPlane.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_plane;

VioManager::VioManager(VioManagerOptions &params_) : thread_init_running(false), thread_init_success(false) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OV_PLANE ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,";
    if (state->_options.use_plane_constraint) {
      of_statistics << "plane init,";
    }
    of_statistics << "msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  // If we are recording statistics, then open our file
  if (params.record_plane_tracking_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_plane_tracking_filepath)) {
      boost::filesystem::remove(params.record_plane_tracking_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_plane_tracking_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics_tracking.open(params.record_plane_tracking_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics_tracking << "# timestamp (sec),feat/plane,num plane,track length(avg),track length(std),track length(max),";
    of_statistics_tracking << "num constraint updates,state planes,";
    of_statistics_tracking << "triangulation,delaunay,matching,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  trackFEATS = std::shared_ptr<TrackBase>(new TrackPlane(
      state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
      params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.trackplane_options));

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(state->_cam_intrinsics_cameras, state->_options.max_aruco_features,
                                                           params.use_stereo, params.histogram_method, params.downsize_aruco));
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // Our state initialize
  initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);

  // Our plane initializer class
  updaterPLANE = std::make_shared<UpdaterPlane>(params.msckf_options, params.featinit_options);

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity);
  }
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {

  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  double oldest_time = state->margtimestep();
  if (oldest_time > state->_timestamp) {
    oldest_time = -1;
  }
  if (!is_initialized_vio) {
    oldest_time = message.timestamp - params.init_options.init_window_time + state->_calib_dt_CAMtoIMU->value()(0) - 0.10;
  }
  propagator->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if we have it
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  sim_map_feat2plane.clear();
  for (auto const &featscam : feats) {
    for (auto const &feat : featscam) {
      if ((int)feat.second(2) == -1)
        continue;
      // TrackSIM adds 1 to sim id and for each aruco we have 4 points
      size_t id = feat.first + 4 * state->_options.max_aruco_features + 1;
      sim_map_feat2plane[id] = (size_t)feat.second(2);
    }
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }
  do_feature_propagate_update(message);
}

void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // Perform our feature tracking!
  trackFEATS->feed_new_camera(message);

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, message.timestamp);
    }
    if (did_zupt_update) {
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }

  // Call on our propagate and update function
  do_feature_propagate_update(message);
}

void VioManager::do_feature_propagate_update(const ov_core::CameraData &message) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != message.timestamp) {
    propagator->propagate_and_clone(state, message.timestamp);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);

  // Don't need to get the oldest features until we reach our max number of clones
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    for (const auto &camuvpair : (*it1)->uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps) {
      if ((int)cams.second.size() > state->_options.max_clone_size) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // Note: if we have a slam feature that has lost tracking, then we should marginalize it out
  // Note: we only enforce this if the current camera message is where the feature was seen from
  // Note: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  StateHelper::marginalize_slam(state);

  // Separate our SLAM features into new ones, and old ones
  std::set<size_t> feat_do_not_add;
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    feat_do_not_add.insert(feats_slam.at(i)->featid);
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF_tmp = feats_lost;
  featsup_MSCKF_tmp.insert(featsup_MSCKF_tmp.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF_tmp.insert(featsup_MSCKF_tmp.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Get the current plane matches
  // NOTE: Also we will merge any SLAM planes together if the frontend combined them here
  // NOTE: This should only happen in the realworld as in SIM, planes are fixed ids
  // NOTE: Any planes not actively seen will be marginalized here
  std::map<size_t, size_t> feat2plane;
  std::map<size_t, std::set<size_t>> plane2oldplane;
  std::shared_ptr<TrackPlane> trackPlane = nullptr;
  TrackPlane::PlaneTrackingInfo track_info;
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
    if (message.timestamp - startup_time >= params.dt_slam_delay) {
      trackPlane = std::dynamic_pointer_cast<TrackPlane>(trackFEATS);
      if (trackPlane != nullptr) {
        feat2plane = trackPlane->get_feature2plane();
        plane2oldplane = trackPlane->get_plane2oldplane();
        trackPlane->get_tracking_info(track_info);
      }
      std::shared_ptr<TrackSIM> trackPlaneSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
      if (trackPlaneSIM != nullptr) {
        feat2plane = sim_map_feat2plane;
      }
      StateHelper::merge_planes_and_marginalize(state, feat2plane, plane2oldplane);
    }
  }

  // Count how many features are on the plane
  std::map<size_t, size_t> plane2featct;
  for (auto const &tmp : feat2plane) {
    plane2featct[tmp.second]++;
  }

  // Seperate our MSCKF features into ones that lie on a plane and those that do not
  std::set<size_t> current_planes;
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF_plane;
  for (auto const &feat : featsup_MSCKF_tmp) {
    feat_do_not_add.insert(feat->featid);
    if (feat2plane.find(feat->featid) != feat2plane.end()) {
      current_planes.insert(feat2plane.at(feat->featid));
      featsup_MSCKF_plane.push_back(feat);
    }
  }

  // Get more MSCKF features for each plane if we have them
  // If we have an active feature that has 5 or more measurements
  // Then we should try to use it in this update since it lies on the plane...
  if (state->_options.plane_collect_init_feats) {
    for (auto const &featplanepair : feat2plane) {
      size_t featid = featplanepair.first;
      size_t planeid = featplanepair.second;
      // skip if this plane is not been selected for use yet
      if (current_planes.find(planeid) == current_planes.end())
        continue;
      // skip if the feature is a slam feature
      if (feat_do_not_add.find(featid) != feat_do_not_add.end())
        continue;
      // skip if this plane has not that many in it
      if (plane2featct.at(planeid) < 4)
        continue;
      // else lets get it from our feature database to update with!
      std::shared_ptr<Feature> feat = trackFEATS->get_feature_database()->get_feature(featid);
      if (feat != nullptr) {
        int ct_clones = 0;
        for (const auto &pair : feat->timestamps)
          ct_clones = std::max(ct_clones, (int)feat->timestamps[pair.first].size());
        if (ct_clones > state->_options.max_clone_size - 1) {
          featsup_MSCKF_plane.push_back(feat);
          // std::cout << "appending new feat to plane " << planeid << std::endl;
        }
      }
    }
  }

  // Try to initialize any planes that we can with the current measurements
  std::vector<std::shared_ptr<ov_core::Feature>> featsup_INIT_used;
  if (state->_options.use_plane_constraint && state->_options.use_plane_slam_feats &&
      message.timestamp - startup_time >= params.dt_slam_delay) {
    updaterPLANE->init_vio_plane(state, featsup_MSCKF_plane, featsup_INIT_used, feat2plane);
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Remove any features used in the plane init
  std::set<size_t> feats_used_ids;
  for (auto const &feat : featsup_INIT_used)
    feats_used_ids.insert(feat->featid);
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF;
  for (auto const &feat : featsup_MSCKF_tmp) {
    if (feats_used_ids.find(feat->featid) == feats_used_ids.end()) {
      assert(!feat->to_delete);
      featsup_MSCKF.push_back(feat);
    }
  }

  // Sort based on track length
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  // NOTE: we recompute the active PLANES we have
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);
  current_planes.clear();
  for (auto const &feat : featsup_MSCKF) {
    feat_do_not_add.insert(feat->featid);
    if (feat2plane.find(feat->featid) != feat2plane.end()) {
      current_planes.insert(feat2plane.at(feat->featid));
    }
  }

  // Collect long-MSCKF feature we can try to update with
  // TODO: this can be an issue as we could try to update with a feature
  // TODO: that in the next frame will be a SLAM feature.... how to handle?
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF_extra;
  if (state->_options.plane_collect_msckf_feats) {
    for (auto const &featplanepair : feat2plane) {
      size_t featid = featplanepair.first;
      size_t planeid = featplanepair.second;
      // skip if this plane is not been selected for use yet
      if (current_planes.find(planeid) == current_planes.end())
        continue;
      // skip if the feature is a slam feature
      if (feat_do_not_add.find(featid) != feat_do_not_add.end())
        continue;
      // skip if the plane is a slam plane
      if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end())
        continue;
      // skip if this plane has not that many in it
      if (plane2featct.at(planeid) < 4)
        continue;
      // else lets get it from our feature database to update with!
      std::shared_ptr<Feature> feat = trackFEATS->get_feature_database()->get_feature(featid);
      if (feat != nullptr) {
        int ct_clones = 0;
        for (const auto &pair : feat->timestamps)
          ct_clones = std::max(ct_clones, (int)feat->timestamps[pair.first].size());
        if (ct_clones > state->_options.max_clone_size - 1) {
          featsup_MSCKF_extra.push_back(feat);
        }
      }
    }
    if (!featsup_MSCKF_extra.empty()) {
      PRINT_INFO("[PLANE-MSCKF]: added %zu extra MSCKF features to do a plane update\n", featsup_MSCKF_extra.size());
    }
  }

  // Pass them to our MSCKF updater
  std::vector<std::shared_ptr<ov_core::Feature>> featsup_MSCKF_used;
  updaterMSCKF->update(state, featsup_MSCKF, featsup_MSCKF_extra, featsup_MSCKF_used, feat2plane);
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    updaterSLAM->update(state, featsup_TEMP, feat2plane);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  rT6 = boost::posix_time::microsec_clock::local_time();

  // Try to initialize the latest ones
  updaterSLAM->delayed_init(state, feats_slam_DELAYED, feat2plane);

  //  // If we still have room, append some that are currently observed
  //  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
  //      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
  //
  //    // Get features we have actively tracked (skip ones used / deleted)
  //    std::vector<std::shared_ptr<Feature>> feats_new, feats_new_plane, feats_new_noplane;
  //    feats_new = trackFEATS->get_feature_database()->features_containing(state->_timestamp, false, true);
  //    for (auto const &feat : feats_new) {
  //      size_t ct_meas = 0;
  //      for (const auto &pair : feat->timestamps)
  //        ct_meas += pair.second.size();
  //      if (ct_meas < 2)
  //        continue;
  //      if (feat2plane.find(feat->featid) != feat2plane.end() && plane2featct.at(feat2plane.at(feat->featid)) >= 5) {
  //        feats_new_plane.push_back(feat);
  //      } else {
  //        feats_new_noplane.push_back(feat);
  //      }
  //    }
  //
  //    // Sort the plane and non-plane based on length
  //    // We will bias SLAM features to be inserted if they lie on a plane!
  //    // TODO: why doesn't this work?
  //    // std::sort(feats_new_plane.begin(), feats_new_plane.end(), compare_feat);     // longest at end
  //    // std::sort(feats_new_noplane.begin(), feats_new_noplane.end(), compare_feat); // longest at end
  //    feats_new = feats_new_noplane;
  //    feats_new.insert(feats_new.end(), feats_new_plane.begin(), feats_new_plane.end());
  //    std::sort(feats_new.begin(), feats_new.end(), compare_feat); // longest at end
  //
  //    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
  //    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
  //    int valid_amount = (amount_to_add > (int)feats_new.size()) ? (int)feats_new.size() : amount_to_add;
  //    if (valid_amount > 0) {
  //      feats_new.erase(feats_new.begin(), feats_new.end() - valid_amount);
  //      updaterSLAM->delayed_init(state, feats_new, feat2plane);
  //      feats_slam_DELAYED.insert(feats_slam_DELAYED.begin(), feats_new.begin(), feats_new.end());
  //    }
  //  }
  rT7 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Push back updated states to tracker
  if (trackPlane != nullptr) {
    for (auto const &clone : state->_clones_IMU) {
      trackPlane->hist_state[clone.first] = clone.second->value();
    }
    for (auto const &calib : state->_calib_IMUtoCAM) {
      trackPlane->hist_calib[calib.first] = calib.second->value();
    }
  }

  // Active set of planes
  std::set<size_t> active_planes;
  for (auto const &featpair : feat2plane)
    active_planes.insert(featpair.second);

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
    good_features_PLANE.clear();
    for (auto const &planepair : good_features_PLANE_SLAM) {
      size_t planeid = planepair.first;
      int planeid_new = -1;
      for (auto const &planeset : plane2oldplane) {
        if (planeset.second.find(planeid) != planeset.second.end()) {
          planeid_new = (int)planeset.first;
        }
      }
      // merge the old plane features into the new one!
      if (planeid_new != -1) {
        good_features_PLANE_SLAM[planeid_new].insert(planepair.second.begin(), planepair.second.end());
      }
    }
    for (auto const &planepair : good_features_PLANE_SLAM) {
      if (active_planes.find(planepair.first) == active_planes.end()) {
        good_features_PLANE_SLAM[planepair.first].clear();
      }
    }
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  // Save all that were used in the plane update
  std::set<size_t> feats_updated_with_plane_constraint;
  if (state->_options.use_plane_constraint) {
    for (auto const &feat : featsup_INIT_used) {
      assert(feat2plane.find(feat->featid) != feat2plane.end());
      size_t planeid = feat2plane.at(feat->featid);
      good_features_PLANE[planeid].push_back(feat->p_FinG);
      feats_updated_with_plane_constraint.insert(feat->featid);
      if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end()) {
        good_features_PLANE_SLAM[planeid][feat->featid] = feat->p_FinG;
      }
    }
    if (state->_options.use_plane_constraint_msckf) {
      for (auto const &feat : featsup_MSCKF_used) {
        if (feat2plane.find(feat->featid) == feat2plane.end())
          continue;
        size_t planeid = feat2plane.at(feat->featid);
        good_features_PLANE[planeid].push_back(feat->p_FinG);
        feats_updated_with_plane_constraint.insert(feat->featid);
        if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end()) {
          good_features_PLANE_SLAM[planeid][feat->featid] = feat->p_FinG;
        }
      }
    }
    if (state->_options.use_plane_constraint_slamd) {
      for (auto const &feat : feats_slam_DELAYED) {
        if (feat2plane.find(feat->featid) == feat2plane.end())
          continue;
        size_t planeid = feat2plane.at(feat->featid);
        if (state->_features_PLANE.find(planeid) == state->_features_PLANE.end())
          continue;
        if (state->_features_SLAM.find(feat->featid) == state->_features_SLAM.end())
          continue;
        assert(state->_features_SLAM.at(feat->featid)->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D);
        Eigen::Vector3d p_FinG = state->_features_SLAM.at(feat->featid)->get_xyz(false);
        good_features_PLANE[planeid].push_back(p_FinG);
        good_features_PLANE_SLAM[planeid][feat->featid] = p_FinG;
        feats_updated_with_plane_constraint.insert(feat->featid);
      }
    }
    if (state->_options.use_plane_constraint_slamu) {
      for (auto const &feat : feats_slam_UPDATE) {
        if (feat2plane.find(feat->featid) == feat2plane.end())
          continue;
        size_t planeid = feat2plane.at(feat->featid);
        if (state->_features_PLANE.find(planeid) == state->_features_PLANE.end())
          continue;
        if (state->_features_SLAM.find(feat->featid) == state->_features_SLAM.end())
          continue;
        assert(state->_features_SLAM.at(feat->featid)->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D);
        Eigen::Vector3d p_FinG = state->_features_SLAM.at(feat->featid)->get_xyz(false);
        good_features_PLANE[planeid].push_back(p_FinG);
        good_features_PLANE_SLAM[planeid][feat->featid] = p_FinG;
        feats_updated_with_plane_constraint.insert(feat->featid);
      }
    }
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  updaterSLAM->change_anchors(state);

  // Cleanup any features older then the marginalization time
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
    }
  }

  // Finally marginalize the oldest clone if needed
  StateHelper::marginalize_old_clone(state);
  rT8 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_planeinit = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_msckf = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_update = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_slam_delay = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_marg = (rT8 - rT7).total_microseconds() * 1e-6;
  double time_total = (rT8 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  if (state->_options.use_plane_constraint) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for plane init (%d feats used)\n" RESET, time_planeinit, (int)featsup_INIT_used.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << ",";
    if (state->_options.use_plane_constraint) {
      of_statistics << time_planeinit << ",";
    }
    of_statistics << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Save plane tracking info
  if (params.record_plane_tracking_information && of_statistics_tracking.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics_tracking << std::fixed << std::setprecision(15) << timestamp_inI << ",";
    of_statistics_tracking << std::fixed << std::setprecision(2) << track_info.avg_feat_per_plane << "," << track_info.plane_per_frame
                           << "," << track_info.avg_track_length << "," << track_info.std_track_length << ",";
    of_statistics_tracking << std::fixed << std::setprecision(2) << track_info.max_track_length << ",";
    of_statistics_tracking << std::fixed << std::setprecision(2) << feats_updated_with_plane_constraint.size() << ","
                           << state->_features_PLANE.size() << ",";
    of_statistics_tracking << std::fixed << std::setprecision(5) << track_info.tracking_time_triangulation << "," << std::setprecision(5)
                           << track_info.tracking_time_delaunay << "," << std::setprecision(5) << track_info.tracking_time_matching << ","
                           << std::setprecision(5) << track_info.tracking_time << std::endl;
    of_statistics_tracking.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
             state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
             state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
             state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

  // Debug for camera imu offset
  if (state->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state->_calib_dt_CAMtoIMU->value()(0));
  }

  // Debug for camera intrinsics
  if (state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }
  PRINT_INFO("total of %zu planes and %zu SLAM features in state\n", state->_features_PLANE.size(), state->_features_SLAM.size());
}
