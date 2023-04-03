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

#ifndef OV_PLANE_UPDATER_PLANE_H
#define OV_PLANE_UPDATER_PLANE_H

#include <Eigen/Eigen>
#include <memory>

#include "feat/FeatureInitializerOptions.h"

#include "UpdaterOptions.h"

namespace ov_core {
class Feature;
class FeatureInitializer;
} // namespace ov_core
namespace ov_type {
class Landmark;
} // namespace ov_type

namespace ov_plane {

class State;

/**
 * @brief Will try to do the plane initialization for features.
 *
 * This really only does plane initialization right now.
 * The logic is that we will try to collect points that we can use to initialize a plane.
 * If we have enough points, we will initialize the plane.
 * Importantly, we ensure to mark the used measurements as "deleted" so that we don't reuse information and become inconsistent.
 *
 *
 */
class UpdaterPlane {

public:
  /**
   * @brief Default constructor for our plane updater
   * @param options Updater options (include measurement noise value) for features
   * @param feat_init_options Feature initializer options
   */
  UpdaterPlane(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options);

  /**
   * @brief This will try to initialize SLAM plane features
   *
   * Given the state and its SLAM features, along with the MSCKF features for the current update,
   * we will try to initialize any *new* planes that we have not yet added into our state vector!
   *
   * @param state State of the filter
   * @param feature_vec Features that can be used for update
   * @param feature_vec_used Features that were used to initialize a new plane!
   * @param feat2plane Feature to planeid mapping!
   */
  void init_vio_plane(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec,
                      std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec_used, const std::map<size_t, size_t> &feat2plane);
  /**
   * @brief This will project the left nullspace of H_f onto the linear system.
   *
   * Please see the @ref update-null for details on how this works.
   * This is the MSCKF nullspace projection which removes the dependency on the feature state.
   * Note that this is done **in place** so all matrices will be different after a function call.
   *
   * NOTE: This function is slightly different from the UpdaterHelper one as we need to apply it to two seperate Jacobians.
   *
   * @param H_f Jacobian with nullspace we want to project onto the system [res = Hx*(x-xhat)+Hf(f-fhat)+n]
   * @param H_x State jacobian
   * @param H_cp CP plane jacobian
   * @param res Measurement residual
   */
  static void nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_cp, Eigen::VectorXd &res);

  /**
   * @brief This will perform measurement compression
   *
   * Please see the @ref update-compress for details on how this works.
   * Note that this is done **in place** so all matrices will be different after a function call.
   *
   * NOTE: This function is slightly different from the UpdaterHelper one as we need to apply it to two seperate Jacobians.
   *
   * @param H_x State jacobian
   * @param H_cp CP plane jacobian
   * @param res Measurement residual
   */
  static void measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_cp, Eigen::VectorXd &res);

protected:
  /// Options used during update for features
  UpdaterOptions _options;

  /// Feature initializer class object
  std::shared_ptr<ov_core::FeatureInitializer> initializer_feat;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, double> chi_squared_table;
};

} // namespace ov_plane

#endif // OV_PLANE_UPDATER_PLANE_H
