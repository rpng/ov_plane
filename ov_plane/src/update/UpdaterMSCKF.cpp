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

#include "UpdaterMSCKF.h"

#include "UpdaterHelper.h"
#include "UpdaterPlane.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "track_plane/PlaneFitting.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_plane;

UpdaterMSCKF::UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec,
                          std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec_extra,
                          std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec_used, const std::map<size_t, size_t> &feat2plane) {

  // Return if no features
  if (feature_vec.empty())
    return;
  auto rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  it0 = feature_vec_extra.begin();
  while (it0 != feature_vec_extra.end()) {

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {
      it0 = feature_vec_extra.erase(it0);
    } else {
      it0++;
    }
  }
  auto rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  std::map<size_t, Eigen::Vector3d> features_p_FinG_original;
  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
    }
    features_p_FinG_original.insert({(*it1)->featid, (*it1)->p_FinG}); // TODO: handle anchored...

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    it1++;
  }
  it1 = feature_vec_extra.begin();
  while (it1 != feature_vec_extra.end()) {

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
    }
    features_p_FinG_original.insert({(*it1)->featid, (*it1)->p_FinG}); // TODO: handle anchored...

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      it1 = feature_vec_extra.erase(it1);
      continue;
    }
    it1++;
  }
  auto rT2 = boost::posix_time::microsec_clock::local_time();

  // For each plane we have, lets recover its CP linearization point
  std::map<size_t, Eigen::Vector3d> plane_estimates_cp_inG;
  std::map<size_t, std::vector<std::shared_ptr<Feature>>> plane_feats; // index by plane id
  if (state->_options.use_plane_constraint && (state->_options.use_plane_constraint_msckf || state->_options.use_plane_constraint_slamu)) {

    // MSCKF: Check how many features lie on the same plane!
    // MSCKF: Each feature must have at least 3 measurements of the plane
    std::map<size_t, size_t> plane_feat_count;
    std::map<size_t, std::set<double>> plane_feat_clones;
    if (state->_options.use_plane_constraint_msckf) {
      for (auto &feat : feature_vec) {
        if (feat2plane.find(feat->featid) == feat2plane.end())
          continue;
        plane_feat_count[feat2plane.at(feat->featid)]++;
        plane_feats[feat2plane.at(feat->featid)].push_back(feat);
        for (auto const &calib : feat->timestamps) {
          for (auto const &time : calib.second) {
            plane_feat_clones[feat2plane.at(feat->featid)].insert(time);
          }
        }
      }
      for (auto &feat : feature_vec_extra) {
        if (feat2plane.find(feat->featid) == feat2plane.end())
          continue;
        plane_feat_count[feat2plane.at(feat->featid)]++;
        plane_feats[feat2plane.at(feat->featid)].push_back(feat);
        for (auto const &calib : feat->timestamps) {
          for (auto const &time : calib.second) {
            plane_feat_clones[feat2plane.at(feat->featid)].insert(time);
          }
        }
      }
    }

    // SLAM: append features if they lie on a plane!
    // SLAM: only add features whose planes are not in the state
    // SLAM: if a plane is in the state, the UpdateSLAM will do the point-on-plane update!
    // TODO: if we do this, the whole system seems to be a lot worst
    // TODO: how can we see if the SLAM feature is an inlier or not????
    if (state->_options.use_plane_constraint_slamu) {
      for (auto &feat : state->_features_SLAM) {
        if (feat2plane.find(feat.first) == feat2plane.end())
          continue;
        size_t planeid = feat2plane.at(feat.first);
        if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end())
          continue;
        // Ensure that the current feature has not failed to match
        if (state->_features_SLAM_to_PLANE.find(feat.first) == state->_features_SLAM_to_PLANE.end() ||
            state->_features_SLAM_to_PLANE.at(feat.first) != 0) {
          plane_feat_count[planeid]++;
          auto featptr = std::make_shared<Feature>();
          featptr->featid = feat.second->_featid;
          featptr->p_FinG = feat.second->get_xyz(false);
          assert(feat.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D);
          plane_feats[planeid].push_back(featptr);
        }
      }
    }

    // Debug print out stats
    for (auto const &planect : plane_feat_count) {
      size_t clonect = (plane_feat_clones.find(planect.first) == plane_feat_clones.end()) ? 0 : plane_feat_clones.at(planect.first).size();
      PRINT_DEBUG(BOLDCYAN "[PLANE-MSCKF]: plane %zu has %zu feats (%zu clones)!\n" RESET, planect.first, planect.second, clonect);
    }

    // Try to triangulate the planes
    for (const auto &featspair : plane_feats) {

      // If the plane is in the state, no need to triangulate it
      if (state->_features_PLANE.find(featspair.first) != state->_features_PLANE.end()) {

        // Current state estimate
        Eigen::Vector3d cp_inG = state->_features_PLANE.at(featspair.first)->value().block(0, 0, 3, 1);

        // Try to optimize the set of features to the current plane estimate
        // TODO: be smarter about how we get focal length here...
        double focal_length = state->_cam_intrinsics_cameras.at(0)->get_value()(0);
        double sigma_px_norm = _options.sigma_pix / focal_length;
        double sigma_c = state->_options.sigma_constraint;
        Eigen::VectorXd stateI = state->_imu->pose()->value();
        Eigen::VectorXd calib0 = state->_calib_IMUtoCAM.at(0)->value();
        if (state->_options.use_refine_plane_feat &&
            !PlaneFitting::optimize_plane(plane_feats[featspair.first], cp_inG, clones_cam, sigma_px_norm, sigma_c, true, stateI, calib0))
          continue;

        // Set groundtruth if we have it and can set it
        // NOTE: don't set the GT for the plane since it is in our state...
        if (state->_options.use_groundtruths && !state->_true_planes.empty() && !state->_true_features.empty()) {
          Eigen::Vector3d cpdiff = cp_inG - state->_true_planes.at(featspair.first);
          PRINT_INFO(YELLOW "[PLANE-GT]: plane %zu | cp_diff = %.3f, %.3f,%.3f | IN STATE |\n" RESET, featspair.first, cpdiff(0), cpdiff(1),
                     cpdiff(2));
          double featd_avg_norm = 0.0;
          Eigen::Vector3d featd_avg = Eigen::Vector3d::Zero();
          for (auto const &featpair : featspair.second) {
            Eigen::Vector3d featd = featpair->p_FinG - state->_true_features.at(featpair->featid);
            featd_avg += featd;
            featd_avg_norm += featd.norm();
            PRINT_INFO(YELLOW "[PLANE-GT]: feat %zu | feat_diff = %.3f, %.3f,%.3f |\n" RESET, featpair->featid, featd(0), featd(1),
                       featd(2));
            featpair->p_FinG = state->_true_features.at(featpair->featid);
          }
          featd_avg /= (double)featspair.second.size();
          featd_avg_norm /= (double)featspair.second.size();
          PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff = %.3f, %.3f,%.3f |\n" RESET, featd_avg(0), featd_avg(1), featd_avg(2));
          PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff_norm = %.3f |\n" RESET, featd_avg_norm);
        }

        // Report error and append the cp plane
        double avg_error = 0.0;
        for (const auto &feat : featspair.second)
          avg_error += std::abs((feat->p_FinG.transpose() * cp_inG)(0) / cp_inG.norm() - cp_inG.norm());
        avg_error /= (double)featspair.second.size();
        plane_estimates_cp_inG.insert({featspair.first, cp_inG});
        PRINT_INFO(BOLDCYAN "[PLANE-MSCKF]: plane %zu with %.3f avg distance error (in state)!\n" RESET, featspair.first, avg_error);
        continue;
      }

      // Skip the plane if not enough measurements
      // If the plane is not in the state, we need more than 3 features to do an update!
      if (featspair.second.size() < 4)
        continue;

      // The plane is NOT in the state, try to find an initial guess of the plane!!!
      Eigen::Vector4d abcd;
      if (!PlaneFitting::plane_fitting(plane_feats[featspair.first], abcd, state->_options.plane_msckf_min_feat,
                                       state->_options.plane_msckf_max_cond))
        continue;
      double avg_error_tri = 0.0;
      for (const auto &feat : featspair.second)
        avg_error_tri += std::abs(PlaneFitting::point_to_plane_distance(feat->p_FinG, abcd));
      avg_error_tri /= (double)featspair.second.size();

      // Print stats of feature before we try to optimize them...
      if (state->_options.use_groundtruths && !state->_true_planes.empty() && !state->_true_features.empty()) {
        double featd_avg_norm = 0.0;
        Eigen::Vector3d featd_avg = Eigen::Vector3d::Zero();
        for (auto const &featpair : featspair.second) {
          Eigen::Vector3d featd = featpair->p_FinG - state->_true_features.at(featpair->featid);
          featd_avg += featd;
          featd_avg_norm += featd.norm();
        }
        featd_avg /= (double)featspair.second.size();
        featd_avg_norm /= (double)featspair.second.size();
        PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff = %.3f, %.3f,%.3f | BEFORE\n" RESET, featd_avg(0), featd_avg(1), featd_avg(2));
        PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff_norm = %.3f | BEFORE\n" RESET, featd_avg_norm);
      }

      // Try to optimize this plane
      // TODO: be smarter about how we get focal length here...
      double focal_length = state->_cam_intrinsics_cameras.at(0)->get_value()(0);
      double sigma_px_norm = _options.sigma_pix / focal_length;
      double sigma_c = state->_options.sigma_constraint;
      Eigen::Vector3d cp_inG = -abcd.head(3) * abcd(3);
      Eigen::VectorXd stateI = state->_imu->pose()->value();
      Eigen::VectorXd calib0 = state->_calib_IMUtoCAM.at(0)->value();
      if (state->_options.use_refine_plane_feat &&
          !PlaneFitting::optimize_plane(plane_feats[featspair.first], cp_inG, clones_cam, sigma_px_norm, sigma_c, false, stateI, calib0))
        continue;
      abcd.head(3) = cp_inG / cp_inG.norm();
      abcd(3) = -cp_inG.norm();
      double avg_error_opt = 0.0;
      for (const auto &feat : featspair.second)
        avg_error_opt += std::abs(PlaneFitting::point_to_plane_distance(feat->p_FinG, abcd));
      avg_error_opt /= (double)featspair.second.size();

      // Set groundtruth if we have it and can set it
      if (state->_options.use_groundtruths && !state->_true_planes.empty() && !state->_true_features.empty()) {
        Eigen::Vector3d cpdiff = cp_inG - state->_true_planes.at(featspair.first);
        PRINT_INFO(YELLOW "[PLANE-GT]: plane %zu | cp_diff = %.3f, %.3f,%.3f |\n" RESET, featspair.first, cpdiff(0), cpdiff(1), cpdiff(2));
        cp_inG = state->_true_planes.at(featspair.first);
        double featd_avg_norm = 0.0;
        Eigen::Vector3d featd_avg = Eigen::Vector3d::Zero();
        for (auto const &featpair : featspair.second) {
          Eigen::Vector3d featd = featpair->p_FinG - state->_true_features.at(featpair->featid);
          featd_avg += featd;
          featd_avg_norm += featd.norm();
          PRINT_INFO(YELLOW "[PLANE-GT]: feat %zu | feat_diff = %.3f, %.3f,%.3f |\n" RESET, featpair->featid, featd(0), featd(1), featd(2));
          featpair->p_FinG = state->_true_features.at(featpair->featid);
        }
        featd_avg /= (double)featspair.second.size();
        featd_avg_norm /= (double)featspair.second.size();
        PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff = %.3f, %.3f,%.3f |\n" RESET, featd_avg(0), featd_avg(1), featd_avg(2));
        PRINT_INFO(YELLOW "[PLANE-GT]: feat avg | feat_diff_norm = %.3f |\n" RESET, featd_avg_norm);
      }

      // Ensure we have at least 1 non-slam feature
      // TODO: maybe in the future we could try to update SLAM features with this?
      bool has_msckf_feat = false;
      for (auto const &featpair : featspair.second) {
        if (state->_features_SLAM.find(featpair->featid) == state->_features_SLAM.end()) {
          has_msckf_feat = true;
          break;
        }
      }
      if (!has_msckf_feat)
        continue;

      // Skip the plane if not enough measurements
      // if (plane_feats[featspair.first].size() < 4 || (int)plane_feats[featspair.first].size() < state->_options.plane_msckf_min_feat)
      if (plane_feats[featspair.first].size() < 4)
        continue;

      // Success! Lets add the plane!
      plane_estimates_cp_inG.insert({featspair.first, cp_inG});
      PRINT_INFO(BOLDCYAN "[PLANE-MSCKF]: plane %zu | %.3f err tri | %.3f err opt |\n" RESET, featspair.first, avg_error_tri,
                 avg_error_opt);
    }
  }
  auto rT3 = boost::posix_time::microsec_clock::local_time();

  // ==============================================================================================================
  // ==============================================================================================================
  // ==============================================================================================================

  // Loop over all planes we have
  std::set<size_t> features_used_already;
  for (auto const &planepair : plane_estimates_cp_inG) {

    // Get all features / variables for this plane
    // Also get if this plane is in the state vector or not
    // If it is in the state vector, we can directly update
    // Otherwise we will need to nullspace project it away!!
    size_t planeid = planepair.first;
    bool is_slam_plane = (state->_features_PLANE.find(planeid) != state->_features_PLANE.end());
    std::vector<std::shared_ptr<Feature>> features = plane_feats.at(planeid);
    assert(planeid != 0);
    assert(is_slam_plane || features.size() > 3);

    // Calculate the max possible measurement size (1 constraint for each feat)
    size_t max_meas_size = 0;
    for (size_t i = 0; i < features.size(); i++) {
      for (const auto &pair : features.at(i)->timestamps) {
        max_meas_size += 3 * features.at(i)->timestamps[pair.first].size();
      }
      if (features.at(i)->timestamps.empty()) {
        max_meas_size += 1; // slam feature has constraint measurement
      }
    }
    size_t max_hx_size = state->max_covariance_size();

    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd Hcp_big = Eigen::MatrixXd::Zero(max_meas_size, 3);
    std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
    std::vector<std::shared_ptr<Type>> Hx_order_big;
    std::vector<UpdaterHelper::UpdaterHelperFeature> feat_temp;
    size_t ct_jacob = 0;
    size_t ct_meas = 0;

    // Compute linear system for each feature, nullspace project, and reject
    for (auto const &feature : features) {

      // If we are a SLAM feature, then we should append the feature Jacobian to Hx
      // Otherwise, if we are a MSCKF feature, then we should nullspace project
      // Thus there will be two set of logics below depending on this flag!
      // NOTE: this does not work yet for when we have an aruco tag feature....
      bool is_slam_feature = (state->_features_SLAM.find(feature->featid) != state->_features_SLAM.end());
      assert((int)feature->featid >= state->_options.max_aruco_features);

      // Convert our feature into our current format
      UpdaterHelper::UpdaterHelperFeature feat;
      feat.featid = feature->featid;
      feat.uvs = feature->uvs;
      feat.uvs_norm = feature->uvs_norm;
      feat.timestamps = feature->timestamps;

      // Append plane info from triangulation or state if we have it
      // If plane is in state, then we can always use it
      // But if it is not, we need at least 3 to do the nullspace projection (per feature)
      if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end()) {
        feat.planeid = planeid;
        feat.cp_FinG = state->_features_PLANE.at(planeid)->value();
        feat.cp_FinG_fej = state->_features_PLANE.at(planeid)->fej();
      } else {
        feat.planeid = planeid;
        feat.cp_FinG = plane_estimates_cp_inG.at(planeid);
        feat.cp_FinG_fej = plane_estimates_cp_inG.at(planeid);
      }

      // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
      feat.feat_representation = state->_options.feat_rep_msckf;
      if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
      }

      // Save the position and its fej value
      if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
        feat.anchor_cam_id = feature->anchor_cam_id;
        feat.anchor_clone_timestamp = feature->anchor_clone_timestamp;
        if (is_slam_feature) {
          feat.p_FinA = state->_features_SLAM.at(feature->featid)->get_xyz(false);
          feat.p_FinA_fej = state->_features_SLAM.at(feature->featid)->get_xyz(true);
        } else {
          feat.p_FinA = feature->p_FinA;
          feat.p_FinA_fej = feature->p_FinA;
        }
      } else {
        if (is_slam_feature) {
          feat.p_FinG = state->_features_SLAM.at(feature->featid)->get_xyz(false);
          feat.p_FinG_fej = state->_features_SLAM.at(feature->featid)->get_xyz(true);
        } else {
          feat.p_FinG = feature->p_FinG;
          feat.p_FinG_fej = feature->p_FinG;
        }
      }

      // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
      Eigen::MatrixXd H_f_tmp;
      Eigen::MatrixXd H_x_tmp;
      Eigen::VectorXd res;
      std::vector<std::shared_ptr<Type>> Hx_order_tmp;

      // Get the Jacobian for this feature
      double sigma_c = state->_options.sigma_constraint;
      UpdaterHelper::get_feature_jacobian_full(state, feat, _options.sigma_pix, sigma_c, H_f_tmp, H_x_tmp, res, Hx_order_tmp);

      // Separate our the derivative in respect to the plane
      // If the plane is NOT in the state, then its jacobian is in the Hf jacobian output...
      Eigen::MatrixXd H_cp, H_f, H_x;
      std::vector<std::shared_ptr<Type>> Hx_order;
      if (is_slam_plane) {
        int planeindex = state->_features_PLANE.at(planeid)->id();
        int planesize = state->_features_PLANE.at(planeid)->size();
        H_x = Eigen::MatrixXd::Zero(H_x_tmp.rows(), H_x_tmp.cols() - planesize);
        assert(planeindex != -1);
        int ct_hx = 0;
        int ct_hx_new = 0;
        for (const auto &var : Hx_order_tmp) {
          if (var->id() == planeindex) {
            H_cp = H_x_tmp.block(0, ct_hx, H_x_tmp.rows(), var->size());
          } else {
            Hx_order.push_back(var);
            H_x.block(0, ct_hx_new, H_x_tmp.rows(), var->size()) = H_x_tmp.block(0, ct_hx, H_x_tmp.rows(), var->size());
            ct_hx_new += var->size();
          }
          ct_hx += var->size();
        }
        H_f = H_f_tmp;
      } else {
        H_cp = H_f_tmp.block(0, H_f_tmp.cols() - 3, H_f_tmp.rows(), 3);
        H_f = H_f_tmp.block(0, 0, H_f_tmp.rows(), 3); // TODO: handle single depth
        H_x = H_x_tmp;
        Hx_order = Hx_order_tmp;
      }
      // std::cout << "H_x" << std::endl << H_x << std::endl << std::endl;
      // std::cout << "Hx_order " << Hx_order.size() << std::endl << std::endl;
      // std::cout << "H_cp" << std::endl << H_cp << std::endl << std::endl;
      // std::cout << "H_f" << std::endl << H_f << std::endl << std::endl;
      assert(H_cp.rows() > 0);
      assert(H_f.rows() > 0);

      // Append to Hx if SLAM feature, else nullspace project (if this is a MSCKF feature)
      if (is_slam_feature) {
        Eigen::MatrixXd H_xf = H_x;
        H_xf.conservativeResize(H_x.rows(), H_x.cols() + H_f.cols());
        H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;
        std::vector<std::shared_ptr<Type>> Hxf_order = Hx_order;
        Hxf_order.push_back(state->_features_SLAM.at(feature->featid));
        H_x = H_xf;
        Hx_order = Hxf_order;
      } else {
        UpdaterPlane::nullspace_project_inplace(H_f, H_x, H_cp, res);
      }

      // We are good!!! Append to our large H vector
      feat_temp.push_back(feat);
      size_t ct_hx = 0;
      for (const auto &var : Hx_order) {
        if (Hx_mapping.find(var) == Hx_mapping.end()) {
          Hx_mapping.insert({var, ct_jacob});
          Hx_order_big.push_back(var);
          ct_jacob += var->size();
        }
        Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
        ct_hx += var->size();
      }
      Hcp_big.block(ct_meas, 0, H_cp.rows(), H_cp.cols()) = H_cp;
      res_big.block(ct_meas, 0, res.rows(), 1) = res;
      ct_meas += res.rows();
    }

    // Now we have stacked all features, resize to the smaller amount
    assert(ct_meas > 0);
    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    Hcp_big.conservativeResize(ct_meas, 3);

    // Perform measurement compression to reduce update size
    UpdaterPlane::measurement_compress_inplace(Hx_big, Hcp_big, res_big);
    assert(Hx_big.rows() > 0);

    // If we are a slam plane we should append it to our Hx jacobian state here and Hx_order
    // If the plane is not in the state, then lets nullspace project away the plane!
    if (is_slam_plane) {
      Eigen::MatrixXd H_xcp = Hx_big;
      H_xcp.conservativeResize(Hx_big.rows(), Hx_big.cols() + Hcp_big.cols());
      H_xcp.block(0, Hx_big.cols(), Hx_big.rows(), Hcp_big.cols()) = Hcp_big;
      std::vector<std::shared_ptr<Type>> Hxcp_order = Hx_order_big;
      Hxcp_order.push_back(state->_features_PLANE.at(planeid));
      Hx_big = H_xcp;
      Hx_order_big = Hxcp_order;
    } else {
      assert(ct_meas > 3);
      UpdaterHelper::nullspace_project_inplace(Hcp_big, Hx_big, res_big);
    }

    // Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order_big);
    Eigen::MatrixXd S = Hx_big * P_marg * Hx_big.transpose();
    S.diagonal() += Eigen::VectorXd::Ones(S.rows());
    double chi2 = res_big.dot(S.llt().solve(res_big));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    if (res_big.rows() < 500) {
      chi2_check = chi_squared_table[res_big.rows()];
    } else {
      boost::math::chi_squared chi_squared_dist(res_big.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
      PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res_big.rows());
    }

    // Check if this plane update failed or not
    if (chi2 > _options.chi2_multipler * chi2_check) {
      PRINT_INFO(BOLDRED "[PLANE-MSCKF]: plane %zu FAILED chi2 with %zu feats and %d meas (%.3f > %.3f)!\n" RESET, planeid, features.size(),
                 res_big.rows(), chi2, _options.chi2_multipler * chi2_check)
      for (auto const &feat : feat_temp) {
        if (feat.planeid != 0 && state->_features_SLAM.find(feat.featid) != state->_features_SLAM.end())
          state->_features_SLAM_to_PLANE[feat.featid] = 0;
      }
      continue;
    }
    PRINT_INFO(BOLDCYAN "[PLANE-MSCKF]: plane %zu SUCCESS chi2 with %zu feats and %d meas (%.3f < %.3f)!\n" RESET, planeid, features.size(),
               res_big.rows(), chi2, _options.chi2_multipler * chi2_check);

    // Mark features as being used here
    for (auto const &feat : feat_temp) {
      if (feat.planeid != 0 && state->_features_SLAM.find(feat.featid) != state->_features_SLAM.end())
        state->_features_SLAM_to_PLANE[feat.featid] = feat.planeid;
    }
    for (auto const &feature : features) {
      feature->to_delete = true;
      features_used_already.insert(feature->featid);
      feature_vec_used.push_back(feature);
    }

    // We are good!!! Lets update!
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  }
  auto rT4 = boost::posix_time::microsec_clock::local_time();

  // ==============================================================================================================
  // ==============================================================================================================
  // ==============================================================================================================

  // Remove features already used here
  std::vector<std::shared_ptr<ov_core::Feature>> feature_vec_tmp;
  for (auto const &feature : feature_vec) {
    if (features_used_already.find(feature->featid) != features_used_already.end())
      continue;
    // Reset feature estimate to before plane optimization (TODO: handle anchored!!)
    assert(!LandmarkRepresentation::is_relative_representation(state->_options.feat_rep_msckf));
    feature->p_FinG = features_p_FinG_original.at(feature->featid);
    feature_vec_tmp.push_back(feature);
  }
  feature_vec = feature_vec_tmp;
  if (feature_vec.empty())
    return;

  // Calculate the max possible measurement size (1 constraint for each feat)
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) {
      max_meas_size += 3 * feature_vec.at(i)->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
  size_t max_hx_size = state->max_covariance_size();
  for (auto &landmark : state->_features_SLAM) {
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // Finally loop over any features that failed
  // Or features that did not lie on the plane
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Should not have already processed it in the plane update
    assert(features_used_already.find((*it2)->featid) == features_used_already.end());

    // Convert our feature into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = (*it2)->anchor_cam_id;
      feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
      feat.p_FinA = (*it2)->p_FinA;
      feat.p_FinA_fej = (*it2)->p_FinA;
    } else {
      feat.p_FinG = (*it2)->p_FinG;
      feat.p_FinG_fej = (*it2)->p_FinG;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    double sigma_c = state->_options.sigma_constraint;
    UpdaterHelper::get_feature_jacobian_full(state, feat, _options.sigma_pix, sigma_c, H_f, H_x, res, Hx_order);

    // Nullspace project
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

    // Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    S.diagonal() += Eigen::VectorXd::Ones(S.rows());
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    if (res.rows() < 500) {
      chi2_check = chi_squared_table[res.rows()];
    } else {
      boost::math::chi_squared chi_squared_dist(res.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
      PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    }

    // Check if we should delete or not
    if (chi2 > _options.chi2_multipler * chi2_check) {
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      // PRINT_DEBUG("featid = %d\n", feat.featid);
      // PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler*chi2_check);
      // std::stringstream ss;
      // ss << "res = " << std::endl << res.transpose() << std::endl;
      // PRINT_DEBUG(ss.str().c_str());
      continue;
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hx_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  auto rT5 = boost::posix_time::microsec_clock::local_time();

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);

  // 5. Perform measurement compression
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
  if (Hx_big.rows() < 1) {
    return;
  }
  auto rT6 = boost::posix_time::microsec_clock::local_time();

  // 6. With all good features update the state
  // Our noise is isotropic, so make it here after our compression
  Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  auto rT7 = boost::posix_time::microsec_clock::local_time();

  // Debug print timing information
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds FEAT triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds PLANE triangulation (%d planes)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)plane_estimates_cp_inG.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds PLANE updates (%d features)\n", (rT4 - rT3).total_microseconds() * 1e-6,
            (int)features_used_already.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds FEAT system (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)feature_vec.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds FEAT compression\n", (rT6 - rT5).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds FEAT update (%d size)\n", (rT7 - rT6).total_microseconds() * 1e-6, (int)res_big.rows());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", (rT7 - rT1).total_microseconds() * 1e-6);
}
