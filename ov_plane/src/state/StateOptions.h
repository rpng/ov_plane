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

#ifndef OV_PLANE_STATE_OPTIONS_H
#define OV_PLANE_STATE_OPTIONS_H

#include "types/LandmarkRepresentation.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

#include <climits>

namespace ov_plane {

/**
 * @brief Struct which stores all our filter options
 */
struct StateOptions {

  // MSCKF VIO State Options ==========================

  /// Bool to determine whether or not to do first estimate Jacobians
  bool do_fej = true;

  /// Bool to determine whether or not use imu message averaging
  bool imu_avg = false;

  /// Bool to determine if we should use Rk4 imu integration
  bool use_rk4_integration = true;

  /// Bool to determine whether or not to calibrate imu-to-camera pose
  bool do_calib_camera_pose = false;

  /// Bool to determine whether or not to calibrate camera intrinsics
  bool do_calib_camera_intrinsics = false;

  /// Bool to determine whether or not to calibrate camera to IMU time offset
  bool do_calib_camera_timeoffset = false;

  /// Max clone size of sliding window
  int max_clone_size = 11;

  /// Max number of estimated SLAM features
  int max_slam_features = 25;

  /// Max number of SLAM features we allow to be included in a single EKF update.
  int max_slam_in_update = 1000;

  /// Max number of MSCKF features we will use at a given image timestep.
  int max_msckf_in_update = 1000;

  /// Max number of estimated ARUCO features
  int max_aruco_features = 1024;

  /// Number of distinct cameras that we will observe features in
  int num_cameras = 1;

  /// What representation our features are in (msckf features)
  ov_type::LandmarkRepresentation::Representation feat_rep_msckf = ov_type::LandmarkRepresentation::Representation::GLOBAL_3D;

  /// What representation our features are in (slam features)
  ov_type::LandmarkRepresentation::Representation feat_rep_slam = ov_type::LandmarkRepresentation::Representation::GLOBAL_3D;

  /// What representation our features are in (aruco tag features)
  ov_type::LandmarkRepresentation::Representation feat_rep_aruco = ov_type::LandmarkRepresentation::Representation::GLOBAL_3D;

  // Plane State Options ==========================

  /// Bool to determine whether or not to use point-on-plane constraints
  bool use_plane_constraint = false;

  /// Bool to determine whether or not to use point-on-plane constraints for MSCKF feature
  bool use_plane_constraint_msckf = false;

  /// Bool to determine whether or not to use point-on-plane constraints for SLAM feature update
  bool use_plane_constraint_slamu = false;

  /// Bool to determine whether or not to use point-on-plane constraints for SLAM feature delayed initialization
  bool use_plane_constraint_slamd = false;

  /// Bool to determine whether or not to use plane SLAM features
  bool use_plane_slam_feats = false;

  /// Bool to determine whether or not to refine planes use co-plane point features
  bool use_refine_plane_feat = true;

  /// Bool to determine whether or not to use feature ground truth
  bool use_groundtruths = false;

  /// Point-on-plane noise sigma
  double sigma_constraint = 0.01;

  /// Point-on-plane constraint noise inflation multiplier (1 if do not inflate)
  double const_init_multi = 1.00;

  /// Plane initialization chi2 check
  double const_init_chi2 = 1.00;

  /// Max number of MSCKF planes allowed
  int max_msckf_plane = 20;

  /// Plane merge update noise sigma
  double sigma_plane_merge = 0.001;

  /// Plane merge update chi2 check
  double plane_merge_chi2 = 1.00;

  /// Max degree allowed to merg two planes
  double plane_merge_deg_max = 1.00;

  /// Bool to determine whether or not to collect more active MSCKF features on the plane
  bool plane_collect_init_feats = true;

  /// Bool to determine whether or not to collect long-MSCKF feature we can try to update with
  bool plane_collect_msckf_feats = false;

  /// MIN number of on-plane point features required to initialize a plane feature
  int plane_init_min_feat = 20;

  /// MAX condition number check for plane fitting (when initialize a SLAM plane)
  double plane_init_max_cond = 100.00;

  /// MIN MSCKF point feature number for plane fitting (when update with a MSCKF plane)
  int plane_msckf_min_feat = 20;

  /// MAX condition number check for plane fitting (when update with a MSCKF plane)
  double plane_msckf_max_cond = 100.00;

  /// Seed to randomize the orientation a bit to introduce randomness
  int rand_init_ori_seed = 0;

  /// Nice print function of what parameters we have loaded
  void print(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("use_fej", do_fej);
      parser->parse_config("use_imuavg", imu_avg);
      parser->parse_config("use_rk4int", use_rk4_integration);
      parser->parse_config("calib_cam_extrinsics", do_calib_camera_pose);
      parser->parse_config("calib_cam_intrinsics", do_calib_camera_intrinsics);
      parser->parse_config("calib_cam_timeoffset", do_calib_camera_timeoffset);
      parser->parse_config("max_clones", max_clone_size);
      parser->parse_config("max_slam", max_slam_features);
      parser->parse_config("max_slam_in_update", max_slam_in_update);
      parser->parse_config("max_msckf_in_update", max_msckf_in_update);
      parser->parse_config("num_aruco", max_aruco_features);
      parser->parse_config("max_cameras", num_cameras);
      std::string rep1 = ov_type::LandmarkRepresentation::as_string(feat_rep_msckf);
      parser->parse_config("feat_rep_msckf", rep1);
      feat_rep_msckf = ov_type::LandmarkRepresentation::from_string(rep1);
      std::string rep2 = ov_type::LandmarkRepresentation::as_string(feat_rep_slam);
      parser->parse_config("feat_rep_slam", rep2);
      feat_rep_slam = ov_type::LandmarkRepresentation::from_string(rep2);
      std::string rep3 = ov_type::LandmarkRepresentation::as_string(feat_rep_aruco);
      parser->parse_config("feat_rep_aruco", rep3);
      feat_rep_aruco = ov_type::LandmarkRepresentation::from_string(rep3);
      parser->parse_config("use_plane_constraint", use_plane_constraint);
      parser->parse_config("use_plane_constraint_msckf", use_plane_constraint_msckf);
      parser->parse_config("use_plane_constraint_slamu", use_plane_constraint_slamu);
      parser->parse_config("use_plane_constraint_slamd", use_plane_constraint_slamd);
      parser->parse_config("use_plane_slam_feats", use_plane_slam_feats);
      parser->parse_config("use_refine_plane_feat", use_refine_plane_feat);
      parser->parse_config("use_groundtruths", use_groundtruths);
      parser->parse_config("sigma_constraint", sigma_constraint);
      parser->parse_config("const_init_multi", const_init_multi);
      parser->parse_config("const_init_chi2", const_init_chi2);
      parser->parse_config("max_msckf_plane", max_msckf_plane);
      parser->parse_config("sigma_plane_merge", sigma_plane_merge);
      parser->parse_config("plane_merge_chi2", plane_merge_chi2);
      parser->parse_config("plane_merge_deg_max", plane_merge_deg_max);
      parser->parse_config("plane_collect_init_feats", plane_collect_init_feats);
      parser->parse_config("plane_collect_msckf_feats", plane_collect_msckf_feats);
      parser->parse_config("plane_init_min_feat", plane_init_min_feat);
      parser->parse_config("plane_init_max_cond", plane_init_max_cond);
      parser->parse_config("plane_msckf_min_feat", plane_msckf_min_feat);
      parser->parse_config("plane_msckf_max_cond", plane_msckf_max_cond);
      parser->parse_config("rand_init_ori_seed", rand_init_ori_seed, false);
    }
    PRINT_DEBUG("  - use_fej: %d\n", do_fej);
    PRINT_DEBUG("  - use_imuavg: %d\n", imu_avg);
    PRINT_DEBUG("  - use_rk4int: %d\n", use_rk4_integration);
    PRINT_DEBUG("  - calib_cam_extrinsics: %d\n", do_calib_camera_pose);
    PRINT_DEBUG("  - calib_cam_intrinsics: %d\n", do_calib_camera_intrinsics);
    PRINT_DEBUG("  - calib_cam_timeoffset: %d\n", do_calib_camera_timeoffset);
    PRINT_DEBUG("  - max_clones: %d\n", max_clone_size);
    PRINT_DEBUG("  - max_slam: %d\n", max_slam_features);
    PRINT_DEBUG("  - max_slam_in_update: %d\n", max_slam_in_update);
    PRINT_DEBUG("  - max_msckf_in_update: %d\n", max_msckf_in_update);
    PRINT_DEBUG("  - max_aruco: %d\n", max_aruco_features);
    PRINT_DEBUG("  - max_cameras: %d\n", num_cameras);
    PRINT_DEBUG("  - feat_rep_msckf: %s\n", ov_type::LandmarkRepresentation::as_string(feat_rep_msckf).c_str());
    PRINT_DEBUG("  - feat_rep_slam: %s\n", ov_type::LandmarkRepresentation::as_string(feat_rep_slam).c_str());
    PRINT_DEBUG("  - feat_rep_aruco: %s\n", ov_type::LandmarkRepresentation::as_string(feat_rep_aruco).c_str());
    PRINT_DEBUG("  - use_plane_constraint: %d\n", use_plane_constraint);
    PRINT_DEBUG("  - use_plane_constraint_msckf: %d\n", use_plane_constraint_msckf);
    PRINT_DEBUG("  - use_plane_constraint_slamu: %d\n", use_plane_constraint_slamu);
    PRINT_DEBUG("  - use_plane_constraint_slamd: %d\n", use_plane_constraint_slamd);
    PRINT_DEBUG("  - use_plane_slam_feats: %d\n", use_plane_slam_feats);
    PRINT_DEBUG("  - use_refine_plane_feat: %d\n", use_refine_plane_feat);
    PRINT_DEBUG("  - use_groundtruths: %d\n", use_groundtruths);
    PRINT_DEBUG("  - sigma_constraint: %.4f\n", sigma_constraint);
    PRINT_DEBUG("  - const_init_multi: %.4f\n", const_init_multi);
    PRINT_DEBUG("  - const_init_chi2: %.4f\n", const_init_chi2);
    PRINT_DEBUG("  - max_msckf_plane: %d\n", max_msckf_plane);
    PRINT_DEBUG("  - sigma_plane_merge: %.4f\n", sigma_plane_merge);
    PRINT_DEBUG("  - plane_merge_chi2: %.4f\n", plane_merge_chi2);
    PRINT_DEBUG("  - plane_merge_deg_max: %.4f\n", plane_merge_deg_max);
    PRINT_DEBUG("  - plane_collect_init_feats: %d\n", plane_collect_init_feats);
    PRINT_DEBUG("  - plane_collect_msckf_feats: %d\n", plane_collect_msckf_feats);
    PRINT_DEBUG("  - plane_init_min_feat: %d\n", plane_init_min_feat);
    PRINT_DEBUG("  - plane_init_max_cond: %.4f\n", plane_init_max_cond);
    PRINT_DEBUG("  - plane_msckf_min_feat: %d\n", plane_msckf_min_feat);
    PRINT_DEBUG("  - plane_msckf_max_cond: %.4f\n", plane_msckf_max_cond);
    PRINT_DEBUG("  - rand_init_ori_seed: %d\n", rand_init_ori_seed);
  }
};

} // namespace ov_plane

#endif // OV_PLANE_STATE_OPTIONS_H