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

#include "Factor_PointOnPlane.h"

using namespace ov_plane;

Factor_PointOnPlane::Factor_PointOnPlane(double sigma_c_) : sigma_c(sigma_c_) {

  // Parameters we are a function of
  set_num_residuals(1);
  mutable_parameter_block_sizes()->push_back(3); // p_FinG
  mutable_parameter_block_sizes()->push_back(3); // cp_inG
}

bool Factor_PointOnPlane::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

  // Recover the current state from our parameters
  Eigen::Vector3d p_FinG = Eigen::Map<const Eigen::Vector3d>(parameters[0]);
  Eigen::Vector3d cp_inG = Eigen::Map<const Eigen::Vector3d>(parameters[1]);

  // Recover cp info
  double d_inG = cp_inG.norm();
  Eigen::Vector3d n_inG = cp_inG / d_inG;

  // Compute residual
  // NOTE: we make this negative ceres cost function is only min||f(x)||^2
  // TODO: is this right? it optimizes if we flip this sign...
  double white_c = 1.0 / sigma_c;
  residuals[0] = -1.0 * white_c * (0.0 - (n_inG.transpose() * p_FinG - d_inG));

  // Compute jacobians if requested by ceres
  if (jacobians) {

    // Jacobian wrt feature p_FinG
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian(jacobians[0]);
      jacobian.block(0, 0, 1, 3) = white_c * n_inG.transpose();
    }

    // Jacobian wrt cp_inG
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian(jacobians[1]);
      jacobian.block(0, 0, 1, 3) =
          white_c * 1.0 / d_inG * (p_FinG.transpose() - n_inG.transpose() * p_FinG * n_inG.transpose() - d_inG * n_inG.transpose());
    }
  }
  return true;
}