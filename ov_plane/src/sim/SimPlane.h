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

#ifndef OV_PLANE_PLANE_H
#define OV_PLANE_PLANE_H

#include <Eigen/Eigen>

namespace ov_plane {

/**
 * @brief Plane helper class
 *
 * This class can be constructed using the 4 boundary points of a plane.
 * A 3d ray can be intersected with this plane and its intersection can be found.
 */
class SimPlane {

public:
  /**
   * @brief Default constructor
   * @param _plane_id ID of this plane
   * @param _pt_top_left Top left point
   * @param _pt_top_right Top right point
   * @param _pt_bottom_left Bottom left point (line start point in floorplan)
   * @param _pt_bottom_right Bottom right point (line end point in floorplan)
   */
  SimPlane(size_t _plane_id, Eigen::Vector3d &_pt_top_left, Eigen::Vector3d &_pt_top_right, Eigen::Vector3d &_pt_bottom_left,
           Eigen::Vector3d &_pt_bottom_right)
      : plane_id(_plane_id), pt_top_left(_pt_top_left), pt_top_right(_pt_top_right), pt_bottom_left(_pt_bottom_left),
        pt_bottom_right(_pt_bottom_right) {
    Eigen::Vector3d V1 = pt_top_right - pt_top_left;
    Eigen::Vector3d V2 = pt_bottom_left - pt_top_left;
    Eigen::Vector3d N = V1.cross(V2);
    A = N(0);
    B = N(1);
    C = N(2);
    D = -(A * pt_top_left(0) + B * pt_top_left(1) + C * pt_top_left(2));
  }

  /**
   * @brief This will try to intersect the given ray and this plane.
   *
   * This assumes that the ray and the plane are in the same frame of reference.
   * This will return true if it is a hit, and false otherwise.
   * Given a plane in the form Ax+By+Cz+D=0 we can first solve for the intersection.
   * R(t) = R0 + Rd*t
   * A(x0 + xd*t) + B(y0 + yd*t) + (z0 + zd*t) + D = 0
   * We can inverse the above function to get the distance along this ray bearing
   * t = -(A*x0 + B*y0 + C*z0 + D) / (A*xd + B*yd + C*zd)
   *
   * @param ray Bearing ray in the frame this plane is represented in [ray_origin, ray_bearing]
   * @param point_intersection Scalar distance along the ray that makes the point lie on this plane
   * @return True if we found an intersection
   */
  bool calculate_intersection(const Eigen::Matrix<double, 6, 1> &ray, double &point_intersection) const {

    // Intersect the ray with our plane
    point_intersection = -(A * ray(0) + B * ray(1) + C * ray(2) + D) / (A * ray(3) + B * ray(4) + C * ray(5));

    // Calculate the actual intersection 3d point
    Eigen::Vector3d pt_inter = ray.head(3) + point_intersection * ray.tail(3);

    // Check the result
    Eigen::Vector3d V1 = pt_top_right - pt_top_left;
    V1.normalize();
    Eigen::Vector3d V2 = pt_bottom_left - pt_top_left;
    V2.normalize();
    Eigen::Vector3d V3 = pt_top_right - pt_bottom_right;
    V3.normalize();
    Eigen::Vector3d V4 = pt_bottom_left - pt_bottom_right;
    V4.normalize();
    Eigen::Vector3d U1 = pt_inter - pt_top_left;
    U1.normalize();
    Eigen::Vector3d U2 = pt_inter - pt_bottom_right;
    U2.normalize();

    return (point_intersection > 0 && U1.dot(V1) > 0 && U1.dot(V2) > 0 && U2.dot(V3) > 0 && U2.dot(V4) > 0);
  }

  /**
   * @brief Recovers the closest point representation of this plane
   * @return Closest-point 3d (same frame as features)
   */
  Eigen::Vector3d cp() const {
    Eigen::Vector3d V1 = pt_top_right - pt_top_left;
    Eigen::Vector3d V2 = pt_bottom_left - pt_top_left;
    Eigen::Vector3d N = V1.cross(V2);
    Eigen::Vector3d n_PhiinG = N.normalized();
    return -D / N.norm() * n_PhiinG;
  }

  // Our id of the plane
  size_t plane_id;

  // Our top-left plane point
  Eigen::Vector3d pt_top_left;

  // Our top-right plane point
  Eigen::Vector3d pt_top_right;

  // Our top-bottom plane point
  Eigen::Vector3d pt_bottom_left;

  // Our top-bottom plane point
  Eigen::Vector3d pt_bottom_right;

  // Plane parameters for a general plane (Ax + By + Cz + D = 0)
  double A, B, C, D;
};

} // namespace ov_plane

#endif // OV_PLANE_PLANE_H