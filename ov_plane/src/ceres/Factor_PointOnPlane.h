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

#ifndef OV_PLANE_CERES_POINTONPLANE_H
#define OV_PLANE_CERES_POINTONPLANE_H

#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace ov_plane {

/**
 * @brief Point-on-plane ceres factor.
 *
 * This is a function of the feature and CP plane object.
 * The residual is the distance of the point to the plane in the normal direction.
 */
class Factor_PointOnPlane : public ceres::CostFunction {
public:
  // Measurement constraint noise
  double sigma_c = 1.0;

  /**
   * @brief Default constructor
   * @param sigma_c_ Constraint uncertainty (should be small non-zero)
   */
  Factor_PointOnPlane(double sigma_c_);

  virtual ~Factor_PointOnPlane() {}

  /**
   * @brief Error residual and Jacobian calculation
   */
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
};

} // namespace ov_plane

#endif // OV_PLANE_CERES_POINTONPLANE_H