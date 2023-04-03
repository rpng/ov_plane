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

#ifndef OV_PLANE_TRACKPLANE_OPTIONS_H
#define OV_PLANE_TRACKPLANE_OPTIONS_H

#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

namespace ov_plane {

/**
 * @brief Struct which stores all our TrackPlane options
 */
struct TrackPlaneOptions {

  /// If we should extract and track plane features
  bool track_planes = false;

  /// MAX triangle line length allowed to be considered (long ones are bad meshed)
  int max_tri_side_px = 200;

  /// How many triangular planar normals we should average over
  int max_norm_count = 5;

  /// MAX triangle norm difference to mark a vertex as invalid
  double max_norm_avg_max = 25.00;

  /// MAX triangle norm variance to mark a vertex as invalid
  double max_norm_avg_var = 25.00;

  /// MAX vertex norm difference to be considered the same plane
  double max_norm_deg = 25.00;

  /// MAX point-on-plane distance for two points to be considered the same plane
  double max_dist_between_z = 0.10;

  /// MAX distance between vertex to be considered the same plane
  int max_pairwise_px = 100;

  /// MIN number of norms to do a pairwise match to be considered the same plane
  int min_norms = 3;

  /// If we should try to do pairwise matching on vertex that have already been matched
  bool check_old_feats = true;

  /// MIN number of points for a plane to be considered large enough for an update (spatial filter)
  int filter_num_feat = 4;

  /// Z-test threshold to see if a feature lies on a detected plane (checked for all features in that plane cluster)
  double filter_z_thresh = 1.2;

  // feature triangulation
  int feat_init_min_obs = 4;
  double min_dist = 0.10;
  double max_dist = 60;
  double max_cond_number = 8000;

  /// Nice print function of what parameters we have loaded
  void print(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("use_plane_constraint", track_planes);
      // matching
      parser->parse_config("plane_max_tri_side_px", max_tri_side_px);
      parser->parse_config("plane_max_norm_count", max_norm_count);
      parser->parse_config("plane_max_norm_avg_max", max_norm_avg_max);
      parser->parse_config("plane_max_norm_avg_var", max_norm_avg_var);
      parser->parse_config("plane_max_norm_deg", max_norm_deg);
      parser->parse_config("plane_max_dist_between_z", max_dist_between_z);
      parser->parse_config("plane_max_pairwise_px", max_pairwise_px);
      parser->parse_config("plane_min_norms", min_norms);
      parser->parse_config("plane_check_old_feats", check_old_feats);
      // spatial filter
      parser->parse_config("plane_filter_num_feat", filter_num_feat);
      parser->parse_config("plane_filter_z_thresh", filter_z_thresh);
      // feature triangulation
      parser->parse_config("plane_feat_min_obs", feat_init_min_obs, false);
      parser->parse_config("plane_min_dist", min_dist, false);
      parser->parse_config("plane_max_dist", max_dist, false);
      parser->parse_config("plane_max_cond_number", max_cond_number, false);
    }
    PRINT_DEBUG("TRACK PLANE OPTIONS:\n");
    PRINT_DEBUG("  - track_planes (use_plane_constraint): %d\n", track_planes);
    // matching
    PRINT_DEBUG("  - plane_max_tri_side_px: %d\n", max_tri_side_px);
    PRINT_DEBUG("  - plane_max_norm_count: %d\n", max_norm_count);
    PRINT_DEBUG("  - plane_max_norm_avg_max: %.2f\n", max_norm_avg_max);
    PRINT_DEBUG("  - plane_max_norm_avg_var: %.2f\n", max_norm_avg_var);
    PRINT_DEBUG("  - plane_max_norm_deg: %.2f\n", max_norm_deg);
    PRINT_DEBUG("  - plane_max_dist_between_z: %.2f\n", max_dist_between_z);
    PRINT_DEBUG("  - plane_max_pairwise_px: %d\n", max_pairwise_px);
    PRINT_DEBUG("  - plane_min_norms: %d\n", min_norms);
    PRINT_DEBUG("  - plane_check_old_feats: %d\n", check_old_feats);
    // spatial filter
    PRINT_DEBUG("  - plane_filter_num_feat: %d\n", filter_num_feat);
    PRINT_DEBUG("  - plane_filter_z_thresh: %.2f\n", filter_z_thresh);
    // feature triangulation
    PRINT_DEBUG("  - plane_feat_min_obs: %d\n", feat_init_min_obs);
    PRINT_DEBUG("  - min_dist: %.3f\n", min_dist);
    PRINT_DEBUG("  - max_dist: %.3f\n", max_dist);
    PRINT_DEBUG("  - max_cond_number: %.3f\n", max_cond_number);
  }
};

} // namespace ov_plane

#endif // OV_PLANE_TRACKPLANE_OPTIONS_H