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

#ifndef OV_CORE_TRACK_PLANE_H
#define OV_CORE_TRACK_PLANE_H

#include "feat/FeatureInitializerOptions.h"
#include "track/TrackBase.h"

#include "TrackPlaneOptions.h"

#include "CDT.h"
#include "ikd_tree.h"

namespace ov_plane {

/**
 * @brief Plane + KLT tracking of features.
 *
 * This will do the same tracking as teh TrackKLT but will try to find
 * enviromential planes that features share.
 */
class TrackPlane : public ov_core::TrackBase {
public:
  /**
   * @brief This is a statistics object
   */
  struct PlaneTrackingInfo {

    /// For all active planes, what is the active number of features seen
    double avg_feat_per_plane = 0.0;

    /// For this frame, how many planes have been extracted
    int plane_per_frame = 0;

    /// For all planes in this image, on average, how long have they been tracked?
    double avg_track_length = 0.0;

    /// For all planes in this image the variance how long have they been tracked
    double std_track_length = 0.0;

    /// What is the max track length so far seen
    int max_track_length = 0;

    /// Time to triangulate active features
    double tracking_time_triangulation = 0.0;

    /// Time to perform Delaunay triangulation
    double tracking_time_delaunay = 0.0;

    /// Time to perform plane normal matching and merging
    double tracking_time_matching = 0.0;

    /// How much time did it take to perform plane tracking
    double tracking_time = 0.0;
  };

  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   * @param fast_threshold FAST detection threshold
   * @param gridx size of grid in the x-direction / u-direction
   * @param gridy size of grid in the y-direction / v-direction
   * @param minpxdist features need to be at least this number pixels away from each other
   * @param track_options Options for plane extraction and triangulation
   */
  explicit TrackPlane(std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                      HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist, TrackPlaneOptions track_options)
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist), options(track_options), currplaneid(0) {}

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const ov_core::CameraData &message) override;

  /**
   * @brief We override the display equation so we can show the tags we extract.
   * @param img_out image to which we will overlayed features on
   * @param r1,g1,b1 first color to draw in
   * @param r2,g2,b2 second color to draw in
   * @param overlay Text overlay to replace to normal "cam0" in the top left of screen
   */
  void display_active(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::string overlay = "") override;

  /**
   * @brief Custom function that will display planes and their IDs
   * @param img_out image to which we will overlayed features on
   * @param r1,g1,b1 first color to draw in
   * @param r2,g2,b2 second color to draw in
   * @param highlighted unique ids which we wish to highlight (e.g. slam feats)
   * @param overlay Text overlay to replace to normal "cam0" in the top left of screen
   */
  void display_history_plane(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::vector<size_t> highlighted = {},
                             std::string overlay = "");

  // Latest state estimates (in global frame, q_GtoI and p_IinG)
  // TODO: make this a function!
  std::map<double, Eigen::MatrixXd> hist_state;

  // Latest extrinsic calibration from each camera to the IMU
  // TODO: make this a function!
  std::map<size_t, Eigen::MatrixXd> hist_calib;

  /// Get features we have successfully triangulated
  std::map<size_t, Eigen::Vector3d> get_features() {
    std::lock_guard<std::mutex> lckv(mtx_hist_vars);
    return hist_feat_inG;
  }

  /// Get feature id to plane id map
  std::map<size_t, size_t> get_feature2plane() {
    std::lock_guard<std::mutex> lckv(mtx_hist_vars);
    return hist_feat_to_plane;
  }

  /// Return history of planes which have been merged to new ones
  std::map<size_t, std::set<size_t>> get_plane2oldplane() {
    std::lock_guard<std::mutex> lckv(mtx_hist_vars);
    return hist_plane_to_oldplanes;
  }

  /// Return statistics info of tracking
  void get_tracking_info(PlaneTrackingInfo &track_info);

protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */
  void feed_monocular(const ov_core::CameraData &message, size_t msg_id);

  /**
   * @brief Detects planes from the previous image.
   * @param cam_id Camera id
   *
   * Hopefully, for all features in the last image, we have their 3d positions.
   * We can perform [Delaunay Triangulation](https://github.com/artem-ogre/CDT) to find spacial points.
   * From there calculate for each point its normal!
   */
  void perform_plane_detection_monocular(size_t cam_id);

  /**
   * @brief Computes average norm from a given norm vector
   * @param norms Vector of norms to average
   * @return Average norm (return a zero norm on failure to compute)
   */
  Eigen::Vector3d avg_norm(const std::vector<Eigen::Vector3d> &norms) const;

  /**
   * @brief Detects new features in the current image
   * @param img0pyr image we will detect features on (first level of pyramid)
   * @param mask0 mask which has what ROI we do not want features in
   * @param pts0 vector of currently extracted keypoints in this image
   * @param ids0 vector of feature ids for each currently extracted keypoint
   *
   * Given an image and its currently extracted features, this will try to add new features if needed.
   * Will try to always have the "max_features" being tracked through KLT at each timestep.
   * Passed images should already be grayscaled.
   */
  void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                   std::vector<size_t> &ids0);

  /**
   * @brief KLT track between two images, and do RANSAC afterwards
   * @param img0pyr starting image pyramid
   * @param img1pyr image pyramid we want to track too
   * @param pts0 starting points
   * @param pts1 points we have tracked
   * @param id0 id of the first camera
   * @param id1 id of the second camera
   * @param mask_out what points had valid tracks
   *
   * This will track features from the first image into the second image.
   * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
   * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
   */
  void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                        std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);

  /**
   * @brief Remove any features that are not seen from the current frame
   * @param ids vector of feature ids
   */
  void remove_feats(std::vector<size_t> &ids);

  // Parameters for our FAST grid detector
  int threshold;
  int grid_x;
  int grid_y;

  // Minimum pixel distance to be "far away enough" to be a different extracted feature
  int min_px_dist;

  // Options for our plane extraction
  TrackPlaneOptions options;

  /// Master ID for this tracker (atomic to allow for multi-threading)
  std::atomic<size_t> currplaneid;

  // How many pyramid levels to track
  int pyr_levels = 5;
  cv::Size win_size = cv::Size(15, 15);

  // Last set of image pyramids
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
  std::map<size_t, cv::Mat> img_curr;
  std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;

  // time we last got the img_last file
  std::map<size_t, double> time_last;

  // Previous image triangulations from CDT
  std::map<size_t, cv::Mat> tri_img_last, tri_mask_last;
  std::map<size_t, double> tri_time_last;
  std::map<size_t, std::vector<CDT::Triangle>> tri_cdt_tri;
  std::map<size_t, std::vector<Eigen::Vector3d>> tri_cdt_tri_norms;
  std::map<size_t, std::vector<CDT::V2d<float>>> tri_cdt_verts;
  std::map<size_t, std::vector<size_t>> tri_cdt_vert_ids;
  std::map<size_t, std::vector<cv::KeyPoint>> tri_cdt_vert_pts;

  /// Master classification of features nd their planes ids
  std::map<size_t, size_t> hist_feat_to_plane;

  /// Master list of planes which have been merged into others
  std::map<size_t, std::set<size_t>> hist_plane_to_oldplanes;

  // Triangulated 3d position of each features and their linear systems
  std::mutex mtx_hist_vars;
  std::map<size_t, Eigen::Vector3d> hist_feat_inG;
  std::map<size_t, Eigen::Matrix3d> hist_feat_linsys_A;
  std::map<size_t, Eigen::Vector3d> hist_feat_linsys_b;
  std::map<size_t, int> hist_feat_linsys_count;
  std::map<size_t, std::vector<Eigen::Vector3d>> hist_feat_norms_inG;

  // Record the tracking info
  double _tracking_time_total = 0.0;
  double _tracking_time_triangulation = 0.0;
  double _tracking_time_delaunay = 0.0;
  double _tracking_time_matching = 0.0;
  std::map<size_t, size_t> _track_length;

  // ikd-tree for plane tracking
  shared_ptr<KD_TREE<ikdTree_PointType>> ikd_tree = make_shared<KD_TREE<ikdTree_PointType>>(0.5, 0.6, 0.01);
};

} // namespace ov_plane

#endif /* OV_CORE_TRACK_PLANE_H */
