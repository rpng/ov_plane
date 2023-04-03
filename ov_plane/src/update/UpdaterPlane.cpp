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

#include "UpdaterPlane.h"
#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "track_plane/PlaneFitting.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_plane;

UpdaterPlane::UpdaterPlane(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterPlane::init_vio_plane(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec,
                                  std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec_used,
                                  const std::map<size_t, size_t> &feat2plane) {

  // Return if no features from both msckf and slam state
  if (feature_vec.empty() && state->_features_SLAM.empty())
    return;
  if (feat2plane.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  std::vector<std::shared_ptr<ov_core::Feature>> feature_vec_valid;
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Don't add if this feature is not on a plane
    if (feat2plane.find((*it0)->featid) == feat2plane.end()) {
      it0++;
      continue;
    }

    // Skip if the plane has already been added to the state vector
    size_t planeid = feat2plane.at((*it0)->featid);
    if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end()) {
      it0++;
      continue;
    }

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {
      //(*it0)->to_delete = true; // NOTE: do not delete since could be incomplete track
      it0 = feature_vec.erase(it0);
    } else {
      feature_vec_valid.push_back((*it0));
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

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
  auto it1 = feature_vec_valid.begin();
  while (it1 != feature_vec_valid.end()) {

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

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      //(*it1)->to_delete = true; // NOTE: do not delete since could be incomplete track
      it1 = feature_vec_valid.erase(it1);
      continue;
    }
    it1++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Sort based on track length, want to update with max track MSCKFs
  std::sort(feature_vec_valid.begin(), feature_vec_valid.end(),
            [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
              size_t asize = 0;
              size_t bsize = 0;
              for (const auto &pair : a->timestamps)
                asize += pair.second.size();
              for (const auto &pair : b->timestamps)
                bsize += pair.second.size();
              return asize < bsize;
            });

  // MSCKF: Check how many features lie on the same plane!
  std::map<size_t, size_t> plane_feat_count;
  std::map<size_t, std::vector<std::shared_ptr<Feature>>> plane_feats; // index by plane id
  std::map<size_t, std::set<double>> plane_feat_clones;
  for (auto &feat : feature_vec_valid) {
    if (feat2plane.find(feat->featid) == feat2plane.end())
      continue;
    size_t planeid = feat2plane.at(feat->featid);
    if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end())
      continue;
    if ((int)plane_feat_count[planeid] > state->_options.max_msckf_plane)
      continue;
    plane_feat_count[planeid]++;
    plane_feats[planeid].push_back(feat);
    for (auto const &calib : feat->timestamps) {
      for (auto const &time : calib.second) {
        plane_feat_clones[planeid].insert(time);
      }
    }
  }

  // SLAM: append features if they lie on a plane!
  // TODO: if we do this, the whole system seems to be a lot worst
  // TODO: how can we see if the SLAM feature is an inlier or not????
  //  for (auto &feat : state->_features_SLAM) {
  //    if (feat2plane.find(feat.first) == feat2plane.end())
  //      continue;
  //    size_t planeid = feat2plane.at(feat.first);
  //    if (state->_features_PLANE.find(planeid) != state->_features_PLANE.end())
  //      continue;
  //    plane_feat_count[planeid]++;
  //    auto featptr = std::make_shared<Feature>();
  //    featptr->featid = feat.second->_featid;
  //    featptr->p_FinG = feat.second->get_xyz(false);
  //    assert(feat.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D);
  //    plane_feats[planeid].push_back(featptr);
  //  }

  // Debug print out stats
  for (auto const &planect : plane_feat_count) {
    size_t clonect = (plane_feat_clones.find(planect.first) == plane_feat_clones.end()) ? 0 : plane_feat_clones.at(planect.first).size();
    PRINT_DEBUG(BOLDCYAN "[PLANE-INIT]: plane %zu has %zu feats (%zu clones)!\n" RESET, planect.first, planect.second, clonect);
  }

  // 4. Try to initialize a guess for each plane's CP
  // For each plane we have, lets recover its CP linearization point
  std::map<size_t, Eigen::Vector3d> plane_estimates_cp_inG;
  for (const auto &featspair : plane_feats) {

    // Initial guess of the plane
    Eigen::Vector4d abcd;
    if (!PlaneFitting::plane_fitting(plane_feats[featspair.first], abcd, state->_options.plane_init_min_feat,
                                     state->_options.plane_init_max_cond))
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

    // Try to optimize this plane and features together
    // TODO: be smarter about how we get focal length here...
    double focal_length = state->_cam_intrinsics_cameras.at(0)->get_value()(0);
    double sigma_px_norm = _options.sigma_pix / focal_length;
    double sigma_c = state->_options.sigma_constraint;
    Eigen::Vector3d cp_inG = -abcd.head(3) * abcd(3);
    Eigen::VectorXd stateI = state->_imu->pose()->value();
    Eigen::VectorXd calib0 = state->_calib_IMUtoCAM.at(0)->value();
    if (!PlaneFitting::optimize_plane(plane_feats[featspair.first], cp_inG, clones_cam, sigma_px_norm, sigma_c, false, stateI, calib0))
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

    // Success! Lets add the plane!
    plane_estimates_cp_inG.insert({featspair.first, cp_inG});
    PRINT_INFO(BOLDCYAN "[PLANE-INIT]: plane %zu | %.3f err tri | %.3f err opt |\n" RESET, featspair.first, avg_error_tri, avg_error_opt);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // 5. use the features that are on the same plane to initialize tha plane
  for (auto const &planepair : plane_estimates_cp_inG) {

    // Get all features / variables for this plane
    size_t planeid = planepair.first;
    Eigen::Vector3d cp_inG = planepair.second;
    std::vector<std::shared_ptr<Feature>> features = plane_feats.at(planeid);
    assert(features.size() >= 3);
    assert(state->_features_PLANE.find(planeid) == state->_features_PLANE.end());

    // Calculate the max possible measurement size (1 constraint for each feat time)
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

      // Append plane info from triangulation
      feat.planeid = planeid;
      feat.cp_FinG = cp_inG;
      feat.cp_FinG_fej = cp_inG;

      // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
      feat.feat_representation = (is_slam_feature) ? state->_options.feat_rep_slam : state->_options.feat_rep_msckf;
      if (feat.feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
      }

      // Save the position and its fej value
      assert(!LandmarkRepresentation::is_relative_representation(feat.feat_representation));
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
      Eigen::MatrixXd H_f;
      Eigen::MatrixXd H_x;
      Eigen::VectorXd res;
      std::vector<std::shared_ptr<Type>> Hx_order;

      // Get the Jacobian for this feature
      double sigma_c = state->_options.const_init_multi * state->_options.sigma_constraint;
      UpdaterHelper::get_feature_jacobian_full(state, feat, _options.sigma_pix, sigma_c, H_f, H_x, res, Hx_order);

      // Separate our the derivative in respect to the plane
      assert(H_f.cols() == 6); // TODO: handle single depth
      Eigen::MatrixXd H_cp = H_f.block(0, H_f.cols() - 3, H_f.rows(), 3);
      H_f = H_f.block(0, 0, H_f.rows(), 3).eval();

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
      Hcp_big.block(ct_meas, 0, H_cp.rows(), H_cp.cols()) = H_cp;

      // Append our residual and move forward
      res_big.block(ct_meas, 0, res.rows(), 1) = res;
      ct_meas += res.rows();
    }

    // Now we have stacked all features, resize to the smaller amount
    assert(ct_meas > 3);
    assert(ct_meas <= max_meas_size);
    assert(ct_jacob <= max_hx_size);
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
    Hcp_big.conservativeResize(ct_meas, 3);

    // Perform measurement compression to reduce update size
    UpdaterPlane::measurement_compress_inplace(Hx_big, Hcp_big, res_big);
    assert(Hx_big.rows() > 0);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

    // Create plane feature pointer
    auto plane = std::make_shared<Vec>(3);
    plane->set_value(cp_inG);
    plane->set_fej(cp_inG);

    // Try to initialize (internally checks chi2)
    if (StateHelper::initialize(state, plane, Hx_order_big, Hx_big, Hcp_big, R_big, res_big, state->_options.const_init_chi2)) {

      // Append to the state vector
      state->_features_PLANE.insert({planeid, plane});
      PRINT_INFO(GREEN "[PLANE-INIT]: plane %d inited | cp_init = %.3f,%.3f,%.3f | cp = %.3f,%.3f,%.3f |\n" RESET, planeid, cp_inG(0),
                 cp_inG(1), cp_inG(2), plane->value()(0), plane->value()(1), plane->value()(2));

      // Get what the marginal covariance init'ed was...
      Eigen::MatrixXd cov_marg = StateHelper::get_marginal_covariance(state, {plane});
      Eigen::Vector3d sigmas = cov_marg.diagonal().transpose().cwiseSqrt();
      PRINT_INFO(GREEN "[PLANE-INIT]: plane prior = %.3f, %.3f, %.3f | inflation = %.3f |\n" RESET, sigmas(0), sigmas(1), sigmas(2),
                 state->_options.const_init_multi);

      // Remove all features from the MSCKF vector if we updated with it
      std::set<size_t> ids;
      for (auto const &feature : features) {
        assert(ids.find(feature->featid) == ids.end());
        ids.insert(feature->featid);
        feature_vec_used.push_back(feature);
      }
      it0 = feature_vec.begin();
      while (it0 != feature_vec.end()) {
        if (ids.find((*it0)->featid) != ids.end()) {
          (*it0)->to_delete = true;
          feature_vec_used.push_back((*it0));
          it0 = feature_vec.erase(it0);
        } else {
          it0++;
        }
      }
    } else {
      PRINT_INFO(RED "[PLANE-INIT]: plane %d init failed | cp = %.3f,%.3f,%.3f |\n" RESET, planeid, cp_inG(0), cp_inG(1), cp_inG(2),
                 plane->value()(0));
    }
  }
}

void UpdaterPlane::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_cp, Eigen::VectorXd &res) {

  // Make sure we have enough measurements to project
  assert(H_f.rows() >= H_f.cols());

  // Apply the left nullspace of H_f to all variables
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_f.cols(); ++n) {
    for (int m = (int)H_f.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_cp.block(m - 1, 0, 2, H_cp.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
  // NOTE: need to eigen3 eval here since this experiences aliasing!
  // H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
  H_x = H_x.block(H_f.cols(), 0, H_x.rows() - H_f.cols(), H_x.cols()).eval();
  H_cp = H_cp.block(H_f.cols(), 0, H_cp.rows() - H_f.cols(), H_cp.cols()).eval();
  res = res.block(H_f.cols(), 0, res.rows() - H_f.cols(), res.cols()).eval();

  // Sanity check
  assert(H_x.rows() == res.rows());
  assert(H_cp.rows() == res.rows());
}

void UpdaterPlane::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_cp, Eigen::VectorXd &res) {

  // Return if H_x is a fat matrix (there is no need to compress in this case)
  if (H_x.rows() <= H_x.cols())
    return;

  // Do measurement compression through givens rotations
  // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
  // See page 252, Algorithm 5.2.4 for how these two loops work
  // They use "matlab" index notation, thus we need to subtract 1 from all index
  Eigen::JacobiRotation<double> tempHo_GR;
  for (int n = 0; n < H_x.cols(); n++) {
    for (int m = (int)H_x.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_x(m - 1, n), H_x(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_x.block(m - 1, n, 2, H_x.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_cp.block(m - 1, 0, 2, H_cp.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // If H is a fat matrix, then use the rows
  // Else it should be same size as our state
  int r = std::min(H_x.rows(), H_x.cols());

  // Construct the smaller jacobian and residual after measurement compression
  assert(r <= H_x.rows());
  H_x.conservativeResize(r, H_x.cols());
  H_cp.conservativeResize(r, H_cp.cols());
  res.conservativeResize(r, res.cols());
}
