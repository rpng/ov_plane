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

#include "Simulator.h"

#include "cam/CamBase.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "sim/BsplineSE3.h"
#include "utils/colors.h"
#include "utils/dataset_reader.h"

using namespace ov_core;
using namespace ov_plane;

Simulator::Simulator(VioManagerOptions &params_) {

  //===============================================================
  //===============================================================

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OV_PLANE VISUAL-INERTIAL SIMULATOR STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Store a copy of our params
  // NOTE: We need to explicitly create a copy of our shared pointers to the camera objects
  // NOTE: Otherwise if we perturb it would also change our "true" parameters
  this->params = params_;
  params.camera_intrinsics.clear();
  for (auto const &tmp : params_.camera_intrinsics) {
    auto tmp_cast = std::dynamic_pointer_cast<ov_core::CamEqui>(tmp.second);
    if (tmp_cast != nullptr) {
      params.camera_intrinsics.insert({tmp.first, std::make_shared<ov_core::CamEqui>(tmp.second->w(), tmp.second->h())});
      params.camera_intrinsics.at(tmp.first)->set_value(params_.camera_intrinsics.at(tmp.first)->get_value());
    } else {
      params.camera_intrinsics.insert({tmp.first, std::make_shared<ov_core::CamRadtan>(tmp.second->w(), tmp.second->h())});
      params.camera_intrinsics.at(tmp.first)->set_value(params_.camera_intrinsics.at(tmp.first)->get_value());
    }
  }

  // Load the groundtruth trajectory and its spline
  DatasetReader::load_simulated_trajectory(params.sim_traj_path, traj_data);

  // Make average z-axis at z=0
  double avg_z = 0.0;
  for (auto const &pose : traj_data)
    avg_z += pose(3);
  avg_z /= (double)traj_data.size();
  for (size_t i = 0; i < traj_data.size(); i++) {
    traj_data.at(i)(3) -= avg_z;
  }

  // Create the spline!
  spline = std::make_shared<ov_core::BsplineSE3>();
  spline->feed_trajectory(traj_data);

  // Set all our timestamps as starting from the minimum spline time
  timestamp = spline->get_start_time();
  timestamp_last_imu = spline->get_start_time();
  timestamp_last_cam = spline->get_start_time();

  // Get the pose at the current timestep
  Eigen::Matrix3d R_GtoI_init;
  Eigen::Vector3d p_IinG_init;
  bool success_pose_init = spline->get_pose(timestamp, R_GtoI_init, p_IinG_init);
  if (!success_pose_init) {
    PRINT_ERROR(RED "[SIM]: unable to find the first pose in the spline...\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Find the timestamp that we move enough to be considered "moved"
  double distance = 0.0;
  double distancethreshold = params.sim_distance_threshold;
  while (true) {

    // Get the pose at the current timestep
    Eigen::Matrix3d R_GtoI;
    Eigen::Vector3d p_IinG;
    bool success_pose = spline->get_pose(timestamp, R_GtoI, p_IinG);

    // Check if it fails
    if (!success_pose) {
      PRINT_ERROR(RED "[SIM]: unable to find jolt in the groundtruth data to initialize at\n" RESET);
      std::exit(EXIT_FAILURE);
    }

    // Append to our scalar distance
    distance += (p_IinG - p_IinG_init).norm();
    p_IinG_init = p_IinG;

    // Now check if we have an acceleration, else move forward in time
    if (distance > distancethreshold) {
      break;
    } else {
      timestamp += 1.0 / params.sim_freq_cam;
      timestamp_last_imu += 1.0 / params.sim_freq_cam;
      timestamp_last_cam += 1.0 / params.sim_freq_cam;
    }
  }
  PRINT_DEBUG("[SIM]: moved %.3f seconds into the dataset where it starts moving\n", timestamp - spline->get_start_time());

  // Append the current true bias to our history
  hist_true_bias_time.push_back(timestamp_last_imu - 1.0 / params.sim_freq_imu);
  hist_true_bias_accel.push_back(true_bias_accel);
  hist_true_bias_gyro.push_back(true_bias_gyro);
  hist_true_bias_time.push_back(timestamp_last_imu);
  hist_true_bias_accel.push_back(true_bias_accel);
  hist_true_bias_gyro.push_back(true_bias_gyro);
  hist_true_bias_time.push_back(timestamp_last_imu + 1.0 / params.sim_freq_imu);
  hist_true_bias_accel.push_back(true_bias_accel);
  hist_true_bias_gyro.push_back(true_bias_gyro);

  // Our simulation is running
  is_running = true;

  //===============================================================
  //===============================================================

  // Load the seeds for the random number generators
  gen_state_init = std::mt19937(params.sim_seed_state_init);
  gen_state_init.seed(params.sim_seed_state_init);
  gen_state_perturb = std::mt19937(params.sim_seed_preturb);
  gen_state_perturb.seed(params.sim_seed_preturb);
  gen_meas_imu = std::mt19937(params.sim_seed_measurements);
  gen_meas_imu.seed(params.sim_seed_measurements);

  // Create generator for our camera
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    gen_meas_cams.push_back(std::mt19937(params.sim_seed_measurements));
    gen_meas_cams.at(i).seed(params.sim_seed_measurements);
  }

  //===============================================================
  //===============================================================

  // Perturb all calibration if we should
  if (params.sim_do_perturbation) {

    // Do the perturbation
    perturb_parameters(gen_state_perturb, params_);

    // Debug print simulation parameters
    params.print_and_load_estimator();
    params.print_and_load_noise();
    params.print_and_load_state();
    params.print_and_load_trackers();
    params.print_and_load_simulation();
  }

  //===============================================================
  //===============================================================

  // Generate our planes :)
  generate_planes();

  // We will create synthetic camera frames and ensure that each has enough features
  // double dt = 0.25/freq_cam;
  double dt = 0.25;
  size_t mapsize = featmap.size();
  PRINT_DEBUG("[SIM]: Generating map features at %d rate\n", (int)(1.0 / dt));

  // Loop through each camera
  // NOTE: we loop through cameras here so that the feature map for camera 1 will always be the same
  // NOTE: thus when we add more cameras the first camera should get the same measurements
  for (int i = 0; i < params.state_options.num_cameras; i++) {

    // Reset the start time
    double time_synth = spline->get_start_time();

    // Loop through each pose and generate our feature map in them!!!!
    while (true) {

      // Get the pose at the current timestep
      Eigen::Matrix3d R_GtoI;
      Eigen::Vector3d p_IinG;
      bool success_pose = spline->get_pose(time_synth, R_GtoI, p_IinG);

      // We have finished generating features
      if (!success_pose)
        break;

      // Get the uv features for this frame
      std::vector<std::pair<size_t, Eigen::VectorXf>> uvs = project_pointcloud(R_GtoI, p_IinG, i, featmap);

      // Count how many on on the plane
      int count_free = 0;
      int count_plane = 0;
      for (auto const &featpair : uvs) {
        assert(featpair.second.rows() == 3);
        if ((int)featpair.second(2) == -1) {
          count_free++;
        } else {
          count_plane++;
        }
      }

      // If we do not have enough, generate more
      if (count_free < params.num_pts) {
        generate_points(R_GtoI, p_IinG, i, featmap, params.num_pts - count_free, false);
      }
      if (count_plane < params.num_pts_plane) {
        generate_points(R_GtoI, p_IinG, i, featmap, params.num_pts_plane - count_plane, true);
      }

      // Move forward in time
      time_synth += dt;
    }

    // Debug print
    PRINT_DEBUG("[SIM]: Generated %d map features in total over %d frames (camera %d)\n", (int)(featmap.size() - mapsize),
                (int)((time_synth - spline->get_start_time()) / dt), i);
    mapsize = featmap.size();
  }

  // Nice sleep so the user can look at the printout
  sleep(1);
}

void Simulator::perturb_parameters(std::mt19937 gen_state, VioManagerOptions &params_) {

  // One std generator
  std::normal_distribution<double> w(0, 1);

  // Camera IMU offset
  params_.calib_camimu_dt += 0.01 * w(gen_state);

  // Camera intrinsics and extrinsics
  for (int i = 0; i < params_.state_options.num_cameras; i++) {

    // Camera intrinsic properties (k1, k2, p1, p2, r1, r2, r3, r4)
    Eigen::MatrixXd intrinsics = params_.camera_intrinsics.at(i)->get_value();
    for (int r = 0; r < 4; r++) {
      intrinsics(r) += 1.0 * w(gen_state);
    }
    for (int r = 4; r < 8; r++) {
      intrinsics(r) += 0.005 * w(gen_state);
    }
    params_.camera_intrinsics.at(i)->set_value(intrinsics);

    // Our camera extrinsics transform (orientation)
    Eigen::Vector3d w_vec;
    w_vec << 0.001 * w(gen_state), 0.001 * w(gen_state), 0.001 * w(gen_state);
    params_.camera_extrinsics.at(i).block(0, 0, 4, 1) =
        rot_2_quat(exp_so3(w_vec) * quat_2_Rot(params_.camera_extrinsics.at(i).block(0, 0, 4, 1)));

    // Our camera extrinsics transform (position)
    for (int r = 4; r < 7; r++) {
      params_.camera_extrinsics.at(i)(r) += 0.01 * w(gen_state);
    }
  }
}

bool Simulator::get_state(double desired_time, Eigen::Matrix<double, 17, 1> &imustate) {

  // Set to default state
  imustate.setZero();
  imustate(4) = 1;

  // Current state values
  Eigen::Matrix3d R_GtoI;
  Eigen::Vector3d p_IinG, w_IinI, v_IinG;

  // Get the pose, velocity, and acceleration
  bool success_vel = spline->get_velocity(desired_time, R_GtoI, p_IinG, w_IinI, v_IinG);

  // Find the bounding bias values
  bool success_bias = false;
  size_t id_loc = 0;
  for (size_t i = 0; i < hist_true_bias_time.size() - 1; i++) {
    if (hist_true_bias_time.at(i) < desired_time && hist_true_bias_time.at(i + 1) >= desired_time) {
      id_loc = i;
      success_bias = true;
      break;
    }
  }

  // If failed, then that means we don't have any more spline or bias
  if (!success_vel || !success_bias) {
    return false;
  }

  // Interpolate our biases (they will be at every IMU timestep)
  double lambda = (desired_time - hist_true_bias_time.at(id_loc)) / (hist_true_bias_time.at(id_loc + 1) - hist_true_bias_time.at(id_loc));
  Eigen::Vector3d true_bg_interp = (1 - lambda) * hist_true_bias_gyro.at(id_loc) + lambda * hist_true_bias_gyro.at(id_loc + 1);
  Eigen::Vector3d true_ba_interp = (1 - lambda) * hist_true_bias_accel.at(id_loc) + lambda * hist_true_bias_accel.at(id_loc + 1);

  // Finally lets create the current state
  imustate(0, 0) = desired_time;
  imustate.block(1, 0, 4, 1) = rot_2_quat(R_GtoI);
  imustate.block(5, 0, 3, 1) = p_IinG;
  imustate.block(8, 0, 3, 1) = v_IinG;
  imustate.block(11, 0, 3, 1) = true_bg_interp;
  imustate.block(14, 0, 3, 1) = true_ba_interp;
  return true;
}

bool Simulator::get_next_imu(double &time_imu, Eigen::Vector3d &wm, Eigen::Vector3d &am) {

  // Return if the camera measurement should go before us
  if (timestamp_last_cam + 1.0 / params.sim_freq_cam < timestamp_last_imu + 1.0 / params.sim_freq_imu)
    return false;

  // Else lets do a new measurement!!!
  timestamp_last_imu += 1.0 / params.sim_freq_imu;
  timestamp = timestamp_last_imu;
  time_imu = timestamp_last_imu;

  // Current state values
  Eigen::Matrix3d R_GtoI;
  Eigen::Vector3d p_IinG, w_IinI, v_IinG, alpha_IinI, a_IinG;

  // Get the pose, velocity, and acceleration
  // NOTE: we get the acceleration between our two IMU
  // NOTE: this is because we are using a constant measurement model for integration
  // bool success_accel = spline->get_acceleration(timestamp+0.5/freq_imu, R_GtoI, p_IinG, w_IinI, v_IinG, alpha_IinI, a_IinG);
  bool success_accel = spline->get_acceleration(timestamp, R_GtoI, p_IinG, w_IinI, v_IinG, alpha_IinI, a_IinG);

  // If failed, then that means we don't have any more spline
  // Thus we should stop the simulation
  if (!success_accel) {
    is_running = false;
    return false;
  }

  // Transform omega and linear acceleration into imu frame
  Eigen::Vector3d omega_inI = w_IinI;
  Eigen::Vector3d gravity;
  gravity << 0.0, 0.0, params.gravity_mag;
  Eigen::Vector3d accel_inI = R_GtoI * (a_IinG + gravity);

  // Calculate the bias values for this IMU reading
  // NOTE: we skip the first ever bias since we have already appended it
  double dt = 1.0 / params.sim_freq_imu;
  std::normal_distribution<double> w(0, 1);
  if (has_skipped_first_bias) {

    // Move the biases forward in time
    true_bias_gyro(0) += params.imu_noises.sigma_wb * std::sqrt(dt) * w(gen_meas_imu);
    true_bias_gyro(1) += params.imu_noises.sigma_wb * std::sqrt(dt) * w(gen_meas_imu);
    true_bias_gyro(2) += params.imu_noises.sigma_wb * std::sqrt(dt) * w(gen_meas_imu);
    true_bias_accel(0) += params.imu_noises.sigma_ab * std::sqrt(dt) * w(gen_meas_imu);
    true_bias_accel(1) += params.imu_noises.sigma_ab * std::sqrt(dt) * w(gen_meas_imu);
    true_bias_accel(2) += params.imu_noises.sigma_ab * std::sqrt(dt) * w(gen_meas_imu);

    // Append the current true bias to our history
    hist_true_bias_time.push_back(timestamp_last_imu);
    hist_true_bias_gyro.push_back(true_bias_gyro);
    hist_true_bias_accel.push_back(true_bias_accel);
  }
  has_skipped_first_bias = true;

  // Now add noise to these measurements
  wm(0) = omega_inI(0) + true_bias_gyro(0) + params.imu_noises.sigma_w / std::sqrt(dt) * w(gen_meas_imu);
  wm(1) = omega_inI(1) + true_bias_gyro(1) + params.imu_noises.sigma_w / std::sqrt(dt) * w(gen_meas_imu);
  wm(2) = omega_inI(2) + true_bias_gyro(2) + params.imu_noises.sigma_w / std::sqrt(dt) * w(gen_meas_imu);
  am(0) = accel_inI(0) + true_bias_accel(0) + params.imu_noises.sigma_a / std::sqrt(dt) * w(gen_meas_imu);
  am(1) = accel_inI(1) + true_bias_accel(1) + params.imu_noises.sigma_a / std::sqrt(dt) * w(gen_meas_imu);
  am(2) = accel_inI(2) + true_bias_accel(2) + params.imu_noises.sigma_a / std::sqrt(dt) * w(gen_meas_imu);

  // Return success
  return true;
}

bool Simulator::get_next_cam(double &time_cam, std::vector<int> &camids,
                             std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Return if the imu measurement should go before us
  if (timestamp_last_imu + 1.0 / params.sim_freq_imu < timestamp_last_cam + 1.0 / params.sim_freq_cam)
    return false;

  // Else lets do a new measurement!!!
  timestamp_last_cam += 1.0 / params.sim_freq_cam;
  timestamp = timestamp_last_cam;
  time_cam = timestamp_last_cam - params.calib_camimu_dt;

  // Get the pose at the current timestep
  Eigen::Matrix3d R_GtoI;
  Eigen::Vector3d p_IinG;
  bool success_pose = spline->get_pose(timestamp, R_GtoI, p_IinG);

  // We have finished generating measurements
  if (!success_pose) {
    is_running = false;
    return false;
  }

  // Loop through each camera
  for (int i = 0; i < params.state_options.num_cameras; i++) {

    // Get the uv features for this frame
    std::vector<std::pair<size_t, Eigen::VectorXf>> uvs = project_pointcloud(R_GtoI, p_IinG, i, featmap);

    // If we do not have enough, generate more
    if ((int)uvs.size() < (params.num_pts + params.num_pts_plane)) {
      PRINT_WARNING(YELLOW "[SIM]: cam %d was unable to generate enough features (%d < %d projections)\n" RESET, (int)i, (int)uvs.size(),
                    (params.num_pts + params.num_pts_plane));
    }

    // If greater than only select the first set
    if ((int)uvs.size() > (params.num_pts + params.num_pts_plane)) {
      uvs.erase(uvs.begin() + (params.num_pts + params.num_pts_plane), uvs.end());
    }

    // Append the map size so all cameras have unique features in them (but the same map)
    // Only do this if we are not enforcing stereo constraints between all our cameras
    for (size_t f = 0; f < uvs.size() && !params.use_stereo; f++) {
      uvs.at(f).first += i * featmap.size();
    }

    // Loop through and add noise to each uv measurement
    std::normal_distribution<double> w(0, 1);
    for (size_t j = 0; j < uvs.size(); j++) {
      uvs.at(j).second(0) += params.msckf_options.sigma_pix * w(gen_meas_cams.at(i));
      uvs.at(j).second(1) += params.msckf_options.sigma_pix * w(gen_meas_cams.at(i));
    }

    // Push back for this camera
    feats.push_back(uvs);
    camids.push_back(i);
  }

  // Return success
  return true;
}

std::vector<std::pair<size_t, Eigen::VectorXf>> Simulator::project_pointcloud(const Eigen::Matrix3d &R_GtoI, const Eigen::Vector3d &p_IinG,
                                                                              int camid, const std::map<size_t, Eigen::VectorXd> &feats) {

  // Assert we have good camera
  assert(camid < params.state_options.num_cameras);
  assert((int)params.camera_intrinsics.size() == params.state_options.num_cameras);
  assert((int)params.camera_extrinsics.size() == params.state_options.num_cameras);

  // Grab our extrinsic and intrinsic values
  Eigen::Matrix<double, 3, 3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(camid).block(0, 0, 4, 1));
  Eigen::Matrix<double, 3, 1> p_IinC = params.camera_extrinsics.at(camid).block(4, 0, 3, 1);
  std::shared_ptr<ov_core::CamBase> camera = params.camera_intrinsics.at(camid);

  // Our projected uv true measurements
  std::vector<std::pair<size_t, Eigen::VectorXf>> uvs;

  // Loop through our map
  double sub_divide = 10.0; // pixels
  Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(std::floor(camera->w() / sub_divide) + 1, std::floor(camera->h() / sub_divide) + 1);
  for (const auto &feat : feats) {

    // Transform feature into current camera frame
    Eigen::Vector3d p_FinG = feat.second.block(0, 0, 3, 1);
    Eigen::Vector3d p_FinI = R_GtoI * (p_FinG - p_IinG);
    Eigen::Vector3d p_FinC = R_ItoC * p_FinI + p_IinC;

    // Skip cloud if too far away
    if (p_FinC(2) > params.sim_max_feature_gen_distance || p_FinC(2) < 0.1)
      continue;

    // Project to normalized coordinates
    Eigen::Vector2f uv_norm;
    uv_norm << (float)(p_FinC(0) / p_FinC(2)), (float)(p_FinC(1) / p_FinC(2));

    // Distort the normalized coordinates
    Eigen::Vector2f uv_dist;
    uv_dist = camera->distort_f(uv_norm);

    // Check that it is inside our bounds
    if (uv_dist(0) < 0 || uv_dist(0) > camera->w() || uv_dist(1) < 0 || uv_dist(1) > camera->h()) {
      continue;
    }

    // Check mask (prevents reprojection to same location...)
    if (mask((int)std::floor(uv_dist(0) / sub_divide), (int)std::floor(uv_dist(1) / sub_divide)) == 1)
      continue;
    mask((int)std::floor(uv_dist(0) / sub_divide), (int)std::floor(uv_dist(1) / sub_divide)) = 1;

    // Else we can add this as a good projection
    Eigen::VectorXf feat_data = Eigen::VectorXf::Zero(3);
    feat_data.block(0, 0, 2, 1) = uv_dist;
    feat_data(2) = (float)feat.second(3); // plane id
    uvs.push_back({feat.first, feat_data});
  }

  // Return our projections
  return uvs;
}

void Simulator::generate_points(const Eigen::Matrix3d &R_GtoI, const Eigen::Vector3d &p_IinG, int camid,
                                std::map<size_t, Eigen::VectorXd> &feats, int numpts, bool on_plane) {

  // Assert we have good camera
  assert(camid < params.state_options.num_cameras);
  assert((int)params.camera_intrinsics.size() == params.state_options.num_cameras);
  assert((int)params.camera_extrinsics.size() == params.state_options.num_cameras);

  // Grab our extrinsic and intrinsic values
  Eigen::Matrix<double, 3, 3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(camid).block(0, 0, 4, 1));
  Eigen::Matrix<double, 3, 1> p_IinC = params.camera_extrinsics.at(camid).block(4, 0, 3, 1);
  std::shared_ptr<ov_core::CamBase> camera = params.camera_intrinsics.at(camid);

  // Loop through our map and mask were we can't generate feastures
  double sub_divide = 10.0; // pixels
  Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(std::floor(camera->w() / sub_divide) + 1, std::floor(camera->h() / sub_divide) + 1);
  for (const auto &feat : feats) {
    Eigen::Vector3d p_FinG = feat.second.block(0, 0, 3, 1);
    Eigen::Vector3d p_FinI = R_GtoI * (p_FinG - p_IinG);
    Eigen::Vector3d p_FinC = R_ItoC * p_FinI + p_IinC;
    if (p_FinC(2) > params.sim_max_feature_gen_distance || p_FinC(2) < 0.1)
      continue;
    Eigen::Vector2f uv_norm;
    uv_norm << (float)(p_FinC(0) / p_FinC(2)), (float)(p_FinC(1) / p_FinC(2));
    Eigen::Vector2f uv_dist;
    uv_dist = camera->distort_f(uv_norm);
    if (uv_dist(0) < 0 || uv_dist(0) > camera->w() || uv_dist(1) < 0 || uv_dist(1) > camera->h()) {
      continue;
    }
    mask((int)std::floor(uv_dist(0) / sub_divide), (int)std::floor(uv_dist(1) / sub_divide)) = 1;
  }

  // Generate the desired number of features
  int try_count = 0;
  for (int i = 0; i < numpts; i++) {

    // Uniformly randomly generate within our fov
    // NOTE: also ensure that we are not generating points on top of others!!
    std::uniform_real_distribution<double> gen_u(0, camera->w());
    std::uniform_real_distribution<double> gen_v(0, camera->h());
    double u_dist = gen_u(gen_state_init);
    double v_dist = gen_v(gen_state_init);
    int count = 0;
    while (mask((int)std::floor(u_dist / sub_divide), (int)std::floor(v_dist / sub_divide)) == 1) {
      u_dist = gen_u(gen_state_init);
      v_dist = gen_v(gen_state_init);
      if (count > 5000) {
        std::cout << "unable to generate feature uv in the mask, are you using too many features???" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      count++;
    }

    // Convert to opencv format
    cv::Point2f uv_dist((float)u_dist, (float)v_dist);

    // Undistort this point to our normalized coordinates
    cv::Point2f uv_norm;
    uv_norm = camera->undistort_cv(uv_dist);
    Eigen::Vector3d bearing;
    bearing << uv_norm.x, uv_norm.y, 1;

    // Generate a random depth
    int id_plane = -1;
    Eigen::Vector3d cp_inG;
    double depth = INFINITY;
    if (!on_plane) {
      std::uniform_real_distribution<double> gen_depth(params.sim_min_feature_gen_distance, params.sim_max_feature_gen_distance);
      depth = gen_depth(gen_state_init);
    } else {

      // pt-on-plane: intersect the point with the planes and get the depth
      Eigen::Matrix<double, 6, 1> ray;
      ray.head(3) = p_IinG - R_GtoI.transpose() * R_ItoC.transpose() * p_IinC; // the position of the camera in global (origin of the ray)
      ray(3) = bearing(0);
      ray(4) = bearing(1);
      ray(5) = bearing(2);
      ray.tail(3) = R_GtoI.transpose() * R_ItoC.transpose() * ray.tail(3); // rotate the bearing to global

      // Loop all the planes and find the intersection values
      bool found_intersection = false;
      for (auto const &plane : planes) {
        double tmp_range = 0.0;
        if (plane.calculate_intersection(ray, tmp_range)) {
          if (tmp_range < depth) {
            depth = tmp_range;
            id_plane = (int)plane.plane_id;
            cp_inG = plane.cp();
          }
          found_intersection = true;
        }
      }
      if (!found_intersection) {
        std::cout << "unable to intersect ray with planes, are the planes loaded?" << std::endl;
        // std::exit(EXIT_FAILURE);
      }
    }

    // Get the 3d point
    Eigen::Vector3d p_FinC;
    p_FinC = depth * bearing;

    // Move to the global frame of reference
    Eigen::Vector3d p_FinI = R_ItoC.transpose() * (p_FinC - p_IinC);
    Eigen::Vector3d p_FinG = R_GtoI.transpose() * p_FinI + p_IinG;

    // Find the closest point to this new feature
    // We want to reject features that are close to each other in 3d also!
    double closest_dist = INFINITY;
    for (const auto &feat : feats) {
      closest_dist = std::min(closest_dist, (feat.second.block(0, 0, 3, 1) - p_FinG).norm());
    }

    // Check that depth of feature is not too far
    // Else try again to get a new feature that has good depth
    if (p_FinC(2) < 0.1 || p_FinC(2) > params.sim_max_feature_gen_distance || closest_dist < 0.10) {
      if (try_count < 100) {
        i--;
        try_count++;
      } else {
        try_count = 0;
      }
      continue;
    }
    try_count = 0;
    mask((int)std::floor(u_dist / sub_divide), (int)std::floor(v_dist / sub_divide)) = 1;

    // Append this as a new feature
    Eigen::VectorXd feat_data = Eigen::VectorXd::Zero(4);
    feat_data.block(0, 0, 3, 1) = p_FinG;
    feat_data(3) = id_plane; // plane id
    featmap.insert({id_map, feat_data});
    id_map++;
  }
}

void Simulator::generate_planes() {

  // Calculate min and max of the trajectory
  Eigen::Vector3d min = INFINITY * Eigen::Vector3d::Ones();
  Eigen::Vector3d max = -INFINITY * Eigen::Vector3d::Ones();
  for (size_t i = 0; i < traj_data.size() - 1; i++) {
    if (traj_data.at(i)(0) < spline->get_start_time())
      continue;
    Eigen::Vector3d pos = traj_data.at(i).block(1, 0, 3, 1);
    min = min.cwiseMin(pos);
    max = max.cwiseMax(pos);
  }
  double multi_xy = 0.7; // 1.60
  double multi_z = 0.24; // 0.70
  min.block(0, 0, 2, 1) -= multi_xy * params.sim_min_feature_gen_distance * Eigen::Vector2d::Ones();
  min(2) -= multi_z * params.sim_min_feature_gen_distance;
  max.block(0, 0, 2, 1) += multi_xy * params.sim_min_feature_gen_distance * Eigen::Vector2d::Ones();
  max(2) += multi_z * params.sim_min_feature_gen_distance;
  Eigen::Vector3d delta = max - min;

  // Each corner of the cube (bottom, then top)
  //(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1),  (0, 1, 1)
  Eigen::Vector3d b1 = Eigen::Vector3d(min(0), min(1), min(2));
  Eigen::Vector3d b2 = Eigen::Vector3d(min(0) + delta(0), min(1), min(2));
  Eigen::Vector3d b3 = Eigen::Vector3d(min(0), min(1) + delta(1), min(2));
  Eigen::Vector3d b4 = Eigen::Vector3d(min(0) + delta(0), min(1) + delta(1), min(2));
  Eigen::Vector3d t1 = Eigen::Vector3d(b1(0), b1(1), b1(2) + delta(2));
  Eigen::Vector3d t2 = Eigen::Vector3d(b2(0), b2(1), b2(2) + delta(2));
  Eigen::Vector3d t3 = Eigen::Vector3d(b3(0), b3(1), b3(2) + delta(2));
  Eigen::Vector3d t4 = Eigen::Vector3d(b4(0), b4(1), b4(2) + delta(2));

  // add the corner of the room
  //  double scale = 1.5;
  //  Eigen::Vector3d b24 = (b2 + b4) / scale;
  //  Eigen::Vector3d b34 = (b4 + b3) / scale;
  //  Eigen::Vector3d t24 = (t2 + t4) / scale;
  //  Eigen::Vector3d t34 = (t4 + t3) / scale;
  //  Eigen::Vector3d b13 = (b1 + b3) / scale;
  //  Eigen::Vector3d t13 = (t1 + t3) / scale;
  //  Eigen::Vector3d t12 = (t1 + t2) / scale;
  //  Eigen::Vector3d b12 = (b1 + b2) / scale;

  // Create our *cube* around the trajectory
  // Eigen::Vector3d &_pt_top_left, Eigen::Vector3d &_pt_top_right, Eigen::Vector3d &_pt_bottom_left, Eigen::Vector3d &_pt_bottom_right
  size_t plane_id = 1;
  planes.emplace_back(plane_id++, b1, b2, b3, b4);
  planes.emplace_back(plane_id++, t3, t4, t2, t1);
  planes.emplace_back(plane_id++, t3, t1, b3, b1);
  planes.emplace_back(plane_id++, t1, t2, b1, b2);
  planes.emplace_back(plane_id++, t2, t4, b2, b4);
  planes.emplace_back(plane_id++, t4, t3, b4, b3);

  // add the corner of the room
  //  planes.emplace_back(plane_id++, t24, t34, b24, b34);
  //  planes.emplace_back(plane_id++, t13, t34, b13, b34);
  //  planes.emplace_back(plane_id++, t13, t12, b13, b12);
  //  planes.emplace_back(plane_id++, t24, t12, b24, b12);

  PRINT_DEBUG("[SIM]: Generated %zu planes (cube = %.1f,%.1f,%.1f)\n", planes.size(), delta(0), delta(1), delta(2));
  for (auto const &plane : planes) {
    PRINT_DEBUG("[SIM]: plane %zu = %.3f, %.3f, %.3f\n", plane.plane_id, plane.cp()(0), plane.cp()(1), plane.cp()(2));
  }
}
