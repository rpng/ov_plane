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

#ifndef OV_PLANE_PLANE_FITTING_H
#define OV_PLANE_PLANE_FITTING_H

#include <Eigen/Eigen>
#include <memory>
#include <random>

#include "feat/FeatureInitializer.h"

namespace ov_core {
class Feature;
} // namespace ov_core

namespace ov_plane {

/**
 * @brief Functions to initialize planes from sparse points
 *
 * Planes have already been classified into sets of pointcloud features.
 * The logic here is trying to find the closest-point plane initial value and refining it.
 * To get an initial guess, we a planar RANSAC which classifies inliers based on the point-to-plane distance.
 * From there, the plane and features can be jointly refined to improve their result.
 */
class PlaneFitting {
public:
  /**
   * @brief Given a set of point this will fit a plane to it
   *
   * Creates a linear system of A*norm = b using:
   * ax + by + cz + d = 0
   * p_FinG.dot(norm) + d = 0
   *
   * @param feats Set of features (need valid p_FinG)
   * @param abcd Plane parameters
   * @param cond_thresh Threshold of the linear system condition number
   * @return True on success
   */
  static bool fit_plane(const std::vector<std::shared_ptr<ov_core::Feature>> &feats, Eigen::Vector4d &abcd, double cond_thresh,
                        bool cond_check = true);

  /**
   * @brief Computes the plane to point distance
   * @param feats Set of features (need valid p_FinG)
   * @param abcd Plane parameters (plane in global)
   * @return Scalar distance to plane
   */
  static inline double point_to_plane_distance(const Eigen::Vector3d &point, const Eigen::Vector4d &abcd) {
    return point.dot(abcd.head(3)) + abcd(3);
  }

  /**
   * @brief This function will use ransac to find the plane.
   * @param feats Set of features (will be returned as the inlier set!)
   * @param abcd Plane parameters
   * @return True if success
   */
  static bool plane_fitting(std::vector<std::shared_ptr<ov_core::Feature>> &feats, Eigen::Vector4d &plane_abcd, int min_inlier_num = 5,
                            double max_plane_solver_condition_number = 200.0);

  /**
   * @brief This function will optimize a set of features and plane
   * @param feats Set of features (will optimize p_FinG and returned as the inlier set!)
   * @param cp_inG Closest point plane in global frame
   * @param[in] clonesCAM Map between camera ID to map of timestamp to camera pose estimate (rotation from global to camera, position of
   * camera in global frame)
   * @param[in] sigma_px_norm Sigma of feature observation (normalized image coordinates sigma_n = sigma_raw / focal_length)
   * @param[in] sigma_c Point-on-plane constraint
   * @param[in] fix_plane If the plane estimate should not be optimized
   * @param[in] stateI Current pose [q_GtoI, p_IinG]
   * @param[in] calib0 Calibration from IMU to camera [q_ItoC, p_IinC]
   * @return True if success
   */
  static bool optimize_plane(std::vector<std::shared_ptr<ov_core::Feature>> &feats, Eigen::Vector3d &cp_inG,
                             std::unordered_map<size_t, std::unordered_map<double, ov_core::FeatureInitializer::ClonePose>> &clonesCAM,
                             double sigma_px_norm, double sigma_c, bool fix_plane, const Eigen::VectorXd &stateI,
                             const Eigen::VectorXd &calib0);
};

} // namespace ov_plane

#endif // OV_PLANE_PLANE_FITTING_H