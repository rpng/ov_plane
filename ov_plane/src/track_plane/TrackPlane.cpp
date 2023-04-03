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

#include "TrackPlane.h"

#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "track/Grider_FAST.h"
#include "utils/helper.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"

using namespace ov_core;
using namespace ov_plane;

void TrackPlane::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // Preprocessing steps that we do not parallelize
  // NOTE: DO NOT PARALLELIZE THESE!
  // NOTE: These seem to be much slower if you parallelize them...
  rT1 = boost::posix_time::microsec_clock::local_time();
  size_t num_images = message.images.size();
  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {

    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize
    cv::Mat img;
    if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images.at(msg_id), img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      double eq_clip_limit = 10.0;
      cv::Size eq_win_size = cv::Size(8, 8);
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
      clahe->apply(message.images.at(msg_id), img);
    } else {
      img = message.images.at(msg_id);
    }

    // Extract image pyramid
    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);

    // Save!
    img_curr[cam_id] = img;
    img_pyramid_curr[cam_id] = imgpyr;
  }

  // Either call our stereo or monocular version
  // If we are doing binocular tracking, then we should parallize our tracking
  if (num_images == 1) {
    feed_monocular(message, 0);
  } else {
    PRINT_ERROR(RED "ONLY MONO PLANE TRACKING SUPPORTED!\n" RESET);
    std::exit(EXIT_FAILURE);
  }
}

void TrackPlane::display_active(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::string overlay) {

  // Cache the images to prevent other threads from editing while we viz (which can be slow)
  std::map<size_t, cv::Mat> img_last_cache, img_mask_last_cache;
  std::map<size_t, double> tri_time_last_cache;
  std::map<size_t, std::vector<CDT::Triangle>> tri_cdt_tri_cache;
  std::map<size_t, std::vector<Eigen::Vector3d>> tri_cdt_tri_norms_cache;
  std::map<size_t, std::vector<CDT::V2d<float>>> tri_cdt_verts_cache;
  std::map<size_t, std::vector<size_t>> tri_cdt_vert_ids_cache;
  std::map<size_t, std::vector<Eigen::Vector3d>> hist_feat_norms_inG_cache;
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last_cache = tri_img_last;
    img_mask_last_cache = tri_mask_last;
    tri_time_last_cache = tri_time_last;
    tri_cdt_tri_cache = tri_cdt_tri;
    tri_cdt_tri_norms_cache = tri_cdt_tri_norms;
    tri_cdt_verts_cache = tri_cdt_verts;
    tri_cdt_vert_ids_cache = tri_cdt_vert_ids;
  }
  {
    std::lock_guard<std::mutex> lckv(mtx_hist_vars);
    hist_feat_norms_inG_cache = hist_feat_norms_inG;
  }

  // Get the largest width and height
  int max_width = -1;
  int max_height = -1;
  for (auto const &pair : img_last_cache) {
    if (max_width < pair.second.cols)
      max_width = pair.second.cols;
    if (max_height < pair.second.rows)
      max_height = pair.second.rows;
  }

  // Return if we didn't have a last image
  if (img_last_cache.empty() || max_width == -1 || max_height == -1)
    return;

  // If the image is "small" thus we should use smaller display codes
  bool is_small = (std::min(max_width, max_height) < 400);

  // If the image is "new" then draw the images from scratch
  // Otherwise, we grab the subset of the main image and draw on top of it
  bool image_new = ((int)img_last_cache.size() * max_width != img_out.cols || max_height != img_out.rows);

  // If new, then resize the current image
  if (image_new)
    img_out = cv::Mat(max_height, (int)img_last_cache.size() * max_width, CV_8UC3, cv::Scalar(0, 0, 0));

  // Converts a unit vector to a "unit cube" rbg
  // Normalize it, move to [0,1] range, then scale to [0,255]
  auto unit2rgb = [](const Eigen::Vector3d &norm) -> Eigen::Vector3d {
    if (norm.norm() < 1e-8)
      return Eigen::Vector3d::Zero();
    Eigen::Vector3d normed = norm / norm.norm();
    Eigen::Vector3d norm01 = 0.5 * (normed + Eigen::Vector3d::Ones());
    return 255.0 * norm01;
  };

  // Loop through each image, and draw
  int index_cam = 0;
  for (auto const &pair : img_last_cache) {
    // select the subset of the image
    cv::Mat img_temp;
    if (image_new)
      cv::cvtColor(img_last_cache[pair.first], img_temp, cv::COLOR_GRAY2RGB);
    else
      img_temp = img_out(cv::Rect(max_width * index_cam, 0, max_width, max_height));

    // Plot the planes via coloring
    cv::Mat mask_norms = cv::Mat::zeros(img_temp.rows, img_temp.cols, CV_8UC3);
    for (size_t t = 0; t < tri_cdt_tri_cache[pair.first].size(); t++) {
      if (tri_cdt_tri_norms_cache[pair.first].at(t).norm() <= 0)
        continue;
      auto tri = tri_cdt_tri_cache[pair.first][t];
      auto v1 = tri_cdt_verts_cache[pair.first][tri.vertices[0]];
      auto v2 = tri_cdt_verts_cache[pair.first][tri.vertices[1]];
      auto v3 = tri_cdt_verts_cache[pair.first][tri.vertices[2]];
      std::vector<cv::Point> contour;
      contour.emplace_back((int)v1.x, (int)v1.y);
      contour.emplace_back((int)v2.x, (int)v2.y);
      contour.emplace_back((int)v3.x, (int)v3.y);
      Eigen::Vector3d color = unit2rgb(tri_cdt_tri_norms_cache[pair.first].at(t));
      cv::fillConvexPoly(mask_norms, contour, cv::Scalar((int)color(0), (int)color(1), (int)color(2)));
    }
    cv::addWeighted(mask_norms, 0.6, img_temp, 1.0, 0.0, img_temp);

    // Debug, draw on our image!
    for (auto const &tri : tri_cdt_tri_cache[pair.first]) {

      // assert that we don't have any invalid..
      assert(tri.vertices[0] != CDT::noVertex);
      assert(tri.vertices[1] != CDT::noVertex);
      assert(tri.vertices[2] != CDT::noVertex);

      // display the lines for these features
      auto v1 = tri_cdt_verts_cache[pair.first][tri.vertices[0]];
      auto v2 = tri_cdt_verts_cache[pair.first][tri.vertices[1]];
      auto v3 = tri_cdt_verts_cache[pair.first][tri.vertices[2]];
      cv::Point2f pt1(v1.x, v1.y);
      cv::Point2f pt2(v2.x, v2.y);
      cv::Point2f pt3(v3.x, v3.y);
      cv::line(img_temp, pt1, pt2, cv::Scalar(255, 0, 0));
      cv::line(img_temp, pt2, pt3, cv::Scalar(255, 0, 0));
      cv::line(img_temp, pt1, pt3, cv::Scalar(255, 0, 0));

      // If we have hte normal of this line, then we should try to display it
      size_t id1 = tri_cdt_vert_ids_cache[pair.first][tri.vertices[0]];
      size_t id2 = tri_cdt_vert_ids_cache[pair.first][tri.vertices[1]];
      size_t id3 = tri_cdt_vert_ids_cache[pair.first][tri.vertices[2]];
      if (hist_feat_norms_inG_cache.find(id1) != hist_feat_norms_inG_cache.end()) {
        Eigen::Vector3d norm = avg_norm(hist_feat_norms_inG_cache.at(id1));
        if (norm.norm() > 0.0) {
          Eigen::Vector3d color = unit2rgb(norm);
          cv::circle(img_temp, pt1, (is_small) ? 3 : 4, cv::Scalar((int)color(0), (int)color(1), (int)color(2)), cv::FILLED);
        }
      }
      if (hist_feat_norms_inG_cache.find(id2) != hist_feat_norms_inG_cache.end()) {
        Eigen::Vector3d norm = avg_norm(hist_feat_norms_inG_cache.at(id2));
        if (norm.norm() > 0.0) {
          Eigen::Vector3d color = unit2rgb(norm);
          cv::circle(img_temp, pt2, (is_small) ? 3 : 4, cv::Scalar((int)color(0), (int)color(1), (int)color(2)), cv::FILLED);
        }
      }
      if (hist_feat_norms_inG_cache.find(id3) != hist_feat_norms_inG_cache.end()) {
        Eigen::Vector3d norm = avg_norm(hist_feat_norms_inG_cache.at(id3));
        if (norm.norm() > 0.0) {
          Eigen::Vector3d color = unit2rgb(norm);
          cv::circle(img_temp, pt3, (is_small) ? 3 : 4, cv::Scalar((int)color(0), (int)color(1), (int)color(2)), cv::FILLED);
        }
      }
    }

    // Draw what camera this is
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    if (overlay == "") {
      cv::putText(img_temp, "CAM:" + std::to_string((int)pair.first), txtpt, cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0,
                  cv::Scalar(0, 255, 0), 3);
    } else {
      cv::putText(img_temp, overlay, txtpt, cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0, cv::Scalar(0, 0, 255), 3);
    }
    // Overlay the mask
    cv::Mat mask = cv::Mat::zeros(img_mask_last_cache[pair.first].rows, img_mask_last_cache[pair.first].cols, CV_8UC3);
    mask.setTo(cv::Scalar(0, 0, 255), img_mask_last_cache[pair.first]);
    cv::addWeighted(mask, 0.1, img_temp, 1.0, 0.0, img_temp);
    // Replace the output image
    img_temp.copyTo(img_out(cv::Rect(max_width * index_cam, 0, img_last_cache[pair.first].cols, img_last_cache[pair.first].rows)));
    index_cam++;
  }
}

void TrackPlane::display_history_plane(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::vector<size_t> highlighted,
                                       std::string overlay) {

  // Cache the images to prevent other threads from editing while we viz (which can be slow)
  std::map<size_t, cv::Mat> img_last_cache, img_mask_last_cache;
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> pts_last_cache;
  std::unordered_map<size_t, std::vector<size_t>> ids_last_cache;
  std::map<size_t, size_t> hist_feat_to_plane_cache;
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last_cache = img_last;
    img_mask_last_cache = img_mask_last;
    pts_last_cache = pts_last;
    ids_last_cache = ids_last;
  }
  {
    std::lock_guard<std::mutex> lckv(mtx_hist_vars);
    hist_feat_to_plane_cache = hist_feat_to_plane;
  }

  // Count how many features are on the plane
  std::map<size_t, size_t> plane2featct;
  for (auto const &tmp : hist_feat_to_plane_cache) {
    plane2featct[tmp.second]++;
  }

  // Get the largest width and height
  int max_width = -1;
  int max_height = -1;
  for (auto const &pair : img_last_cache) {
    if (max_width < pair.second.cols)
      max_width = pair.second.cols;
    if (max_height < pair.second.rows)
      max_height = pair.second.rows;
  }

  // Return if we didn't have a last image
  if (img_last_cache.empty() || max_width == -1 || max_height == -1)
    return;

  // If the image is "small" thus we should use smaller display codes
  bool is_small = (std::min(max_width, max_height) < 400);

  // If the image is "new" then draw the images from scratch
  // Otherwise, we grab the subset of the main image and draw on top of it
  bool image_new = ((int)img_last_cache.size() * max_width != img_out.cols || max_height != img_out.rows);

  // Generate static plane ids
  std::map<size_t, Eigen::Vector3d> plane2color;
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  for (size_t i = 0; i < currplaneid; i++) {
    std::mt19937_64 rng(i);
    Eigen::Vector3d color = Eigen::Vector3d::Zero();
    while (color.norm() < 0.8)
      color << unif(rng), unif(rng), unif(rng);
    plane2color[i] = 255.0f * color;
  }

  // If new, then resize the current image
  if (image_new)
    img_out = cv::Mat(max_height, (int)img_last_cache.size() * max_width, CV_8UC3, cv::Scalar(0, 0, 0));

  // Loop through each image, and draw
  int index_cam = 0;
  for (auto const &pair : img_last_cache) {
    // select the subset of the image
    cv::Mat img_temp;
    if (image_new)
      cv::cvtColor(img_last_cache[pair.first], img_temp, cv::COLOR_GRAY2RGB);
    else
      img_temp = img_out(cv::Rect(max_width * index_cam, 0, max_width, max_height));

    // For each plane, get active features we have for them
    std::map<size_t, std::vector<cv::Point2f>> plane_points;
    std::map<size_t, std::vector<size_t>> plane_pointids;
    for (auto const &featpair : hist_feat_to_plane_cache) {
      auto it0 = std::find(ids_last_cache[pair.first].begin(), ids_last_cache[pair.first].end(), featpair.first);
      if (it0 == ids_last_cache[pair.first].end())
        continue;
      auto idx0 = std::distance(ids_last_cache[pair.first].begin(), it0);
      plane_points[featpair.second].push_back(pts_last_cache[pair.first].at(idx0).pt);
      plane_pointids[featpair.second].push_back(ids_last_cache[pair.first].at(idx0));
    }

    // Now for each draw its convex hull
    for (auto const &plane : plane_points) {
      if (plane.second.size() < 3)
        continue;
      std::vector<cv::Point2f> hull;
      cv::convexHull(plane.second, hull, false, true);
      if (hull.size() < 3)
        continue;
      Eigen::Vector3d color = plane2color[plane.first];
      cv::Scalar color_cv((int)color(2), (int)color(1), (int)color(0)); // rbg -> bgr
      cv::Point2f pt_avg(0.0f, 0.0f);
      for (auto const &pt : hull) {
        pt_avg.x += pt.x;
        pt_avg.y += pt.y;
      }
      pt_avg.x /= (float)hull.size();
      pt_avg.y /= (float)hull.size();
      int thick = (plane2featct.at(plane.first) < 4) ? 1 : 3;
      cv::putText(img_temp, std::to_string(plane.first), pt_avg, cv::FONT_HERSHEY_SIMPLEX, 0.6, color_cv, 1, cv::LINE_AA);
      for (size_t i = 1; i < hull.size(); i++)
        cv::line(img_temp, hull.at(i - 1), hull.at(i), color_cv, thick);
      cv::line(img_temp, hull.at(0), hull.at(hull.size() - 1), color_cv, thick);
      // std::vector<std::vector<cv::Point2f>> tmp{hull};// not sure why this crashes!!
      // cv::drawContours(img_temp, tmp, -1, color_cv, 2);// not sure why this crashes!!!
      for (size_t i = 0; i < plane_points[plane.first].size(); i++) {
        // Draw this point
        cv::Point2f pt = plane_points[plane.first].at(i);
        size_t id = plane_pointids[plane.first].at(i);
        cv::circle(img_temp, pt, (is_small) ? 1 : 2, color_cv, cv::FILLED);
        // If a highlighted point, then put a nice box around it
        if (std::find(highlighted.begin(), highlighted.end(), id) != highlighted.end()) {
          cv::Point2f pt_l_top = cv::Point2f(pt.x - ((is_small) ? 3 : 5), pt.y - ((is_small) ? 3 : 5));
          cv::Point2f pt_l_bot = cv::Point2f(pt.x + ((is_small) ? 3 : 5), pt.y + ((is_small) ? 3 : 5));
          cv::rectangle(img_temp, pt_l_top, pt_l_bot, cv::Scalar(0, 255, 0), 1);
          cv::circle(img_temp, pt, (is_small) ? 1 : 2, cv::Scalar(0, 255, 0), cv::FILLED);
        }
      }
    }

    // Draw what camera this is
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    if (overlay == "") {
      cv::putText(img_temp, "CAM:" + std::to_string((int)pair.first), txtpt, cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0,
                  cv::Scalar(0, 255, 0), 3);
    } else {
      cv::putText(img_temp, overlay, txtpt, cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0, cv::Scalar(0, 0, 255), 3);
    }
    // Overlay the mask
    cv::Mat mask = cv::Mat::zeros(img_mask_last_cache[pair.first].rows, img_mask_last_cache[pair.first].cols, CV_8UC3);
    mask.setTo(cv::Scalar(0, 0, 255), img_mask_last_cache[pair.first]);
    cv::addWeighted(mask, 0.1, img_temp, 1.0, 0.0, img_temp);
    // Replace the output image
    img_temp.copyTo(img_out(cv::Rect(max_width * index_cam, 0, img_last_cache[pair.first].cols, img_last_cache[pair.first].rows)));
    index_cam++;
  }
}

void TrackPlane::get_tracking_info(PlaneTrackingInfo &track_info) {

  // Lock out mutex
  std::lock_guard<std::mutex> lckv(mtx_hist_vars);

  // Group features by plane id, count number of features on each plane
  std::map<size_t, size_t> plane_to_feat;
  for (auto const &feat : hist_feat_to_plane) {
    if (plane_to_feat.find(feat.second) == plane_to_feat.end()) {
      plane_to_feat[feat.second] = 1;
    } else {
      plane_to_feat[feat.second]++;
    }
  }

  // Calculate average number of features on each plane
  track_info.avg_feat_per_plane = 0.0;
  for (auto const &plane : plane_to_feat) {
    track_info.avg_feat_per_plane += (double)plane.second;
  }
  if (!plane_to_feat.empty())
    track_info.avg_feat_per_plane /= (double)plane_to_feat.size();
  track_info.plane_per_frame = (int)plane_to_feat.size();

  // Update track length
  for (auto const &plane : plane_to_feat) {
    // Deal with plane merging
    for (const auto &plane_id : hist_plane_to_oldplanes) {
      for (const auto &plane_id_old : plane_id.second) {
        if (_track_length.find(plane_id_old) == _track_length.end()) {
          // If we have already merged once, just break the loop
          break;
        } else {
          // Because the track length of the remaining plane after merge should already have the largest track length
          // So we only need to keep that one and remove the rest
          // TODO: Will this have any issue?
          _track_length.erase(plane_id_old);
        }
      }
    }
    if (_track_length.find(plane.first) == _track_length.end()) {
      _track_length[plane.first] = 1;
    } else {
      _track_length.at(plane.first)++;
    }
  }

  // Calculate average track length (In fact only the final average track length is meanful)
  track_info.avg_track_length = 0.0;
  track_info.max_track_length = 0;
  for (auto const &track_length : _track_length) {
    track_info.avg_track_length += (double)track_length.second;
    if ((int)track_length.second > track_info.max_track_length) {
      track_info.max_track_length = (int)track_length.second;
    }
  }
  if (!_track_length.empty())
    track_info.avg_track_length /= (double)_track_length.size();

  // Calculate standard deviation of track length
  double sum = 0;
  for (auto const &track_length : _track_length) {
    sum += pow(track_length.second - track_info.avg_track_length, 2);
  }
  if (_track_length.size() > 1)
    track_info.std_track_length = sqrt(sum / (_track_length.size() - 1));
  else
    track_info.std_track_length = 0;

  // Record the timing info
  track_info.tracking_time_triangulation = _tracking_time_triangulation;
  track_info.tracking_time_delaunay = _tracking_time_delaunay;
  track_info.tracking_time_matching = _tracking_time_matching;
  track_info.tracking_time = _tracking_time_total;
}

void TrackPlane::feed_monocular(const CameraData &message, size_t msg_id) {

  // Lock this data feed for this camera
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Get our image objects for this image
  cv::Mat img = img_curr.at(cam_id);
  std::vector<cv::Mat> imgpyr = img_pyramid_curr.at(cam_id);
  cv::Mat mask = message.masks.at(msg_id);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id].empty()) {
    // Detect new features
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;
    perform_detection_monocular(imgpyr, mask, good_left, good_ids_left);
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    time_last[cam_id] = message.timestamp;
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    return;
  }

  // First perform plane detection in the previous image
  if (options.track_planes) {
    _tracking_time_total = 0.0;
    _tracking_time_triangulation = 0.0;
    _tracking_time_delaunay = 0.0;
    _tracking_time_matching = 0.0;
    perform_plane_detection_monocular(cam_id);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  auto pts_left_old = pts_last[cam_id];
  auto ids_left_old = ids_last[cam_id];
  perform_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old);
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

  // Lets track temporally
  perform_matching(img_pyramid_last[cam_id], imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll);
  assert(pts_left_new.size() == ids_left_old.size());
  rT5 = boost::posix_time::microsec_clock::local_time();

  // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    time_last[cam_id] = message.timestamp;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id].clear();
    ids_last[cam_id].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
        (int)pts_left_new.at(i).pt.y >= img.rows)
      continue;
    // Check if it is in the mask
    // NOTE: mask has max value of 255 (white) if it should be
    if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
      continue;
    // If it is a good track, and also tracked from left to right
    if (mask_ll[i]) {
      good_left.push_back(pts_left_new[i]);
      good_ids_left.push_back(ids_left_old[i]);
    }
  }

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
  }

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    time_last[cam_id] = message.timestamp;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
  }
  rT6 = boost::posix_time::microsec_clock::local_time();

  // Timing information
  PRINT_DEBUG("[TIME]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_DEBUG("[TIME]: %.4f seconds for plane matching\n", (rT3 - rT2).total_microseconds() * 1e-6);
  PRINT_DEBUG("[TIME]: %.4f seconds for detection\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_DEBUG("[TIME]: %.4f seconds for temporal klt\n", (rT5 - rT4).total_microseconds() * 1e-6);
  PRINT_DEBUG("[TIME]: %.4f seconds for feature DB update (%d features)\n", (rT6 - rT5).total_microseconds() * 1e-6, (int)good_left.size());
  PRINT_DEBUG("[TIME]: %.4f seconds for total\n", (rT6 - rT1).total_microseconds() * 1e-6);
}

void TrackPlane::perform_plane_detection_monocular(size_t cam_id) {

  // Lock the system
  auto rTP1 = boost::posix_time::microsec_clock::local_time();
  std::lock_guard<std::mutex> lckv1(mtx_last_vars);
  std::lock_guard<std::mutex> lckv2(mtx_hist_vars);

  // Get data from the previous image!
  double time;
  cv::Mat img, mask;
  std::vector<cv::KeyPoint> pts_left;
  std::vector<size_t> ids_left;
  {
    // std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img = img_last[cam_id].clone();
    time = time_last[cam_id];
    mask = img_mask_last[cam_id].clone();
    pts_left = pts_last[cam_id];
    ids_left = ids_last[cam_id];
  }
  if (pts_left.empty() || ids_left.empty())
    return;
  assert(pts_left.size() == ids_left.size());

  // Return if we do not have the current state estimate
  if (hist_state.find(time) == hist_state.end() || hist_calib.find(cam_id) == hist_calib.end())
    return;

  // Remove any old features that are not seen in this frame
  // NOTE: this won't work if we are doing loop-closure...
  remove_feats(ids_left);

  // IMU historical clone
  Eigen::MatrixXd state = hist_state.at(time);
  Eigen::Matrix3d R_GtoI = quat_2_Rot(state.block(0, 0, 4, 1));
  Eigen::Vector3d p_IinG = state.block(4, 0, 3, 1);

  // Calibration
  Eigen::MatrixXd calib = hist_calib.at(cam_id);
  Eigen::Matrix3d R_ItoC = quat_2_Rot(calib.block(0, 0, 4, 1));
  Eigen::Vector3d p_IinC = calib.block(4, 0, 3, 1);

  // Convert current CAMERA position relative to global
  Eigen::Matrix3d R_GtoCi = R_ItoC * R_GtoI;
  Eigen::Vector3d p_CiinG = p_IinG - R_GtoCi.transpose() * p_IinC;

  // For all features in the current frame, lets update their linear triangulation systems
  // Sum for each feature and if it has enough measurements, recover it
  std::map<size_t, Eigen::Vector3d> hist_feat_inG_new;
  std::map<size_t, Eigen::Matrix3d> hist_feat_linsys_A_new;
  std::map<size_t, Eigen::Vector3d> hist_feat_linsys_b_new;
  std::map<size_t, int> hist_feat_linsys_count_new;
  for (size_t i = 0; i < pts_left.size(); i++) {

    // Get the UV coordinate normal
    cv::Point2f pt_n = camera_calib.at(cam_id)->undistort_cv(pts_left.at(i).pt);
    Eigen::Matrix<double, 3, 1> b_i;
    b_i << pt_n.x, pt_n.y, 1;
    b_i = R_GtoCi.transpose() * b_i;
    b_i = b_i / b_i.norm();
    Eigen::Matrix3d Bperp = skew_x(b_i);

    // Append to our linear system
    size_t featid = ids_left.at(i);
    Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
    Eigen::Vector3d bi = Ai * p_CiinG;
    if (hist_feat_linsys_A.find(featid) == hist_feat_linsys_A.end()) {
      hist_feat_linsys_A_new.insert({featid, Ai});
      hist_feat_linsys_b_new.insert({featid, bi});
      hist_feat_linsys_count_new.insert({featid, 1});
    } else {
      hist_feat_linsys_A_new[featid] = Ai + hist_feat_linsys_A[featid];
      hist_feat_linsys_b_new[featid] = bi + hist_feat_linsys_b[featid];
      hist_feat_linsys_count_new[featid] = 1 + hist_feat_linsys_count[featid];
    }

    // For this feature, recover its 3d position if we have enough observations!
    int num_obs = hist_feat_linsys_count_new.at(featid);
    // if (num_obs >= options.feat_init_min_obs && (num_obs % 4 == 0 || num_obs < 5)) {
    if (num_obs >= options.feat_init_min_obs) {

      // Recover feature estimate
      Eigen::Matrix3d A = hist_feat_linsys_A_new[featid];
      Eigen::Vector3d b = hist_feat_linsys_b_new[featid];
      Eigen::MatrixXd p_FinG = A.colPivHouseholderQr().solve(b);
      Eigen::MatrixXd p_FinCi = R_GtoCi * (p_FinG - p_CiinG);

      // Check A and p_FinCi
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
      Eigen::MatrixXd singularValues;
      singularValues.resize(svd.singularValues().rows(), 1);
      singularValues = svd.singularValues();
      double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

      // If we have a bad condition number, or it is too close
      // Then set the flag for bad (i.e. set z-axis to nan)
      if (std::abs(condA) <= options.max_cond_number && p_FinCi(2, 0) >= options.min_dist && p_FinCi(2, 0) <= options.max_dist &&
          !std::isnan(p_FinCi.norm())) {
        hist_feat_inG_new[featid] = p_FinG;
      }
    }
  }

  // Update history maps
  {
    // std::lock_guard<std::mutex> lckv(mtx_hist_vars);
    for (auto const &pair : hist_feat_linsys_A_new)
      hist_feat_linsys_A[pair.first] = pair.second;
    for (auto const &pair : hist_feat_linsys_b_new)
      hist_feat_linsys_b[pair.first] = pair.second;
    for (auto const &pair : hist_feat_linsys_count_new)
      hist_feat_linsys_count[pair.first] = pair.second;
    for (auto const &pair : hist_feat_inG_new)
      hist_feat_inG[pair.first] = pair.second;
  }

  // Remove any points that do NOT have a 3d estimate yet...
  size_t sz_orig = pts_left.size();
  auto it0 = pts_left.begin();
  auto it1 = ids_left.begin();
  while (it0 != pts_left.end()) {
    if (hist_feat_inG.find(*it1) == hist_feat_inG.end()) {
      it0 = pts_left.erase(it0);
      it1 = ids_left.erase(it1);
    } else {
      it0++;
      it1++;
    }
  }
  auto rTP2 = boost::posix_time::microsec_clock::local_time();

  //====================================================================
  // Delaunay Triangulation
  //====================================================================

  // Now lets perform our Delaunay Triangulation
  // https://github.com/artem-ogre/CDT
  std::vector<CDT::V2d<float>> tri_verts;
  for (auto const &pt : pts_left)
    tri_verts.emplace_back(CDT::V2d<float>::make(pt.pt.x, pt.pt.y));
  CDT::Triangulation<float> cdt(CDT::VertexInsertionOrder::Enum::AsProvided);
  cdt.insertVertices(tri_verts);
  // cdt.insertEdges(tri_edges);
  cdt.eraseSuperTriangle();
  auto rTP3 = boost::posix_time::microsec_clock::local_time();
  PRINT_DEBUG(BOLDCYAN "[PLANE]: %zu vertices | %zu triangles (%zu tracks, %.3f ms)\n" RESET, cdt.vertices.size(), cdt.triangles.size(),
              sz_orig, (rTP3 - rTP2).total_microseconds() * 1e-3);

  // Now for each vertex, lets calculate its normal given the current extracted "planes"
  // Loop through all triangulates, and assign each vertex the same normal
  // Then average each vertex's normal, along with the previous history....
  std::map<size_t, std::set<size_t>> feat_to_close_feat;
  std::vector<Eigen::Vector3d> tri_normals_inG; // (0,0,0) if invalid
  for (auto const &tri : cdt.triangles) {

    // assert that we don't have any invalid..
    assert(tri.vertices[0] != CDT::noVertex);
    assert(tri.vertices[1] != CDT::noVertex);
    assert(tri.vertices[2] != CDT::noVertex);

    // Assert we have the 3d position for all
    size_t id1 = ids_left[tri.vertices[0]];
    size_t id2 = ids_left[tri.vertices[1]];
    size_t id3 = ids_left[tri.vertices[2]];
    assert(hist_feat_inG.find(id1) != hist_feat_inG.end() && hist_feat_inG.find(id2) != hist_feat_inG.end() &&
           hist_feat_inG.find(id3) != hist_feat_inG.end());
    feat_to_close_feat[id1].insert(id2);
    feat_to_close_feat[id1].insert(id3);
    feat_to_close_feat[id2].insert(id1);
    feat_to_close_feat[id2].insert(id3);
    feat_to_close_feat[id3].insert(id1);
    feat_to_close_feat[id3].insert(id2);

    // See if this triangle has an edge that is too large
    double len01 = cv::norm((pts_left[tri.vertices[0]].pt - pts_left[tri.vertices[1]].pt));
    double len12 = cv::norm((pts_left[tri.vertices[1]].pt - pts_left[tri.vertices[2]].pt));
    double len20 = cv::norm((pts_left[tri.vertices[2]].pt - pts_left[tri.vertices[0]].pt));
    if (len01 > options.max_tri_side_px || len12 > options.max_tri_side_px || len20 > options.max_tri_side_px) {
      tri_normals_inG.emplace_back(Eigen::Vector3d::Zero());
      continue;
    }

    // Cross product first two to get the normal in G
    Eigen::Vector3d diff1 = hist_feat_inG.at(id2) - hist_feat_inG.at(id1);
    diff1 /= diff1.norm();
    Eigen::Vector3d diff2 = hist_feat_inG.at(id3) - hist_feat_inG.at(id1);
    diff2 /= diff2.norm();
    Eigen::Vector3d norm = diff1.cross(diff2);
    norm /= norm.norm();

    // To check the sign, we enforce that from the camera frame
    // We get a positive distance to it....
    Eigen::Vector3d p_FinCi = R_GtoCi * (hist_feat_inG.at(id1) - p_CiinG);
    Eigen::Vector3d norm_inCi = R_GtoCi * norm;
    if (norm_inCi.dot(p_FinCi) < 0)
      norm *= -1.0;
    tri_normals_inG.emplace_back(norm);

    // Record this normal for each feature
    {
      // std::lock_guard<std::mutex> lckv(mtx_hist_vars);

      // Append, newest norm!
      hist_feat_norms_inG[id1].push_back(norm);
      hist_feat_norms_inG[id2].push_back(norm);
      hist_feat_norms_inG[id3].push_back(norm);

      // Delete old ones if too many
      if ((int)hist_feat_norms_inG[id1].size() > options.max_norm_count)
        hist_feat_norms_inG[id1].erase(hist_feat_norms_inG[id1].begin(), hist_feat_norms_inG[id1].end() - options.max_norm_count);
      if ((int)hist_feat_norms_inG[id2].size() > options.max_norm_count)
        hist_feat_norms_inG[id2].erase(hist_feat_norms_inG[id2].begin(), hist_feat_norms_inG[id2].end() - options.max_norm_count);
      if ((int)hist_feat_norms_inG[id3].size() > options.max_norm_count)
        hist_feat_norms_inG[id3].erase(hist_feat_norms_inG[id3].begin(), hist_feat_norms_inG[id3].end() - options.max_norm_count);
    }
  }

  //====================================================================
  // Find common norm matches between *features* to find plane matches
  //====================================================================

  // Calculate the average norm for each vertex
  std::map<size_t, Eigen::Vector3d> hist_feat_norms_sum_inG;
  for (auto const &featpair : hist_feat_norms_inG) {
    hist_feat_norms_sum_inG[featpair.first] = avg_norm(featpair.second);
    // hist_feat_norms_sum_inG[featpair.first] = featpair.second.at(featpair.second.size() - 1);
  }

  // Map from feature id to 2d point
  std::map<size_t, cv::KeyPoint> pts_left_map;
  for (size_t i = 0; i < pts_left.size(); i++) {
    pts_left_map[ids_left.at(i)] = pts_left.at(i);
  }

  // Find common norms between *features* to find plane matches
  // NOTE: We will threshold based on the angle between the two norms
  if (!cdt.triangles.empty()) {
    std::set<size_t> done_verts_ids;
    for (auto const &featpair : hist_feat_norms_inG) {

      // Get current feature
      size_t featid = featpair.first;
      std::vector<Eigen::Vector3d> norms = featpair.second;
      Eigen::Vector3d norm = hist_feat_norms_sum_inG.at(featid);

      // Skip if invalid norm
      if ((int)norms.size() < options.min_norms)
        continue;
      if (norm.norm() <= 0)
        continue;

      // Recover the CP of this plane (assume current point lies on a plane)
      Eigen::Vector3d p_FinG = hist_feat_inG.at(featid);
      // Eigen::Vector3d p_FinCi = R_GtoCi * (p_FinG - p_CiinG);
      double d = p_FinG.dot(norm);
      // Eigen::Vector3d cp_inG = d * norm;
      // Eigen::Matrix3d R_GtoPi;
      // ov_init::InitializerHelper::gram_schmidt(cp_inG, R_GtoPi);

      // Skip if this feature has already been matched to a plane
      if (!options.check_old_feats && hist_feat_to_plane.find(featid) != hist_feat_to_plane.end())
        continue;

      // Skip if triangulation of this point failed....
      if (feat_to_close_feat.find(featid) == feat_to_close_feat.end())
        continue;

      // Find all features that this feature connects to in the current window
      // Then we should compare to their norms and see if any "match" the current
      std::vector<size_t> matches;
      for (auto const &featid_close : feat_to_close_feat.at(featid)) {
        if (hist_feat_norms_inG.find(featid_close) == hist_feat_norms_inG.end())
          continue;
        std::vector<Eigen::Vector3d> norms2 = hist_feat_norms_inG.at(featid_close);
        Eigen::Vector3d norm2 = hist_feat_norms_sum_inG.at(featid_close);
        if ((int)norms2.size() < options.min_norms)
          continue;
        if (norm2.norm() <= 0)
          continue;
        if (done_verts_ids.find(featid_close) != done_verts_ids.end()) // skip features already matched to neighbors
          continue;

        // Pairwise 2d image distance
        double len01 = cv::norm((pts_left_map.at(featid).pt - pts_left_map.at(featid_close).pt));
        if (len01 > options.max_pairwise_px)
          continue;

        // Compute the projection of the feature onto the plane
        Eigen::Vector3d p_F2inG = hist_feat_inG.at(featid_close);
        // Eigen::Vector3d p_F2inCi = R_GtoCi * (p_F2inG - p_CiinG);
        //  Eigen::Vector3d p_FinPi = R_GtoPi * (p_F2inG - cp_inG);
        double plane_dist = p_F2inG.dot(norm) - d;
        // double camxy_dist = (p_FinCi - p_F2inCi).norm() / p_FinCi.norm();

        // Compute distance between the two normals
        double dist = norm.dot(norm2);
        double angle = (180.0 / M_PI) * acos(dist);

        // Append the feature if it passes our thresholds
        // if (!std::isnan(angle) && angle < options.max_norm_deg) {
        //  PRINT_ERROR(RED "[PLANE]: %zu -> %zu is %.2f, %.2f, %.2f in plane (%.2f deg, %.2f camxy, %.2f z)\n" RESET, featid, featid_close,
        //              p_FinPi(0), p_FinPi(1), p_FinPi(2), angle, camxy_dist, std::abs(plane_dist));
        //}
        if (!std::isnan(angle) && angle < options.max_norm_deg && std::abs(plane_dist) < options.max_dist_between_z) {
          matches.emplace_back(featid_close);
        }
      }

      // If there are no features that match this feature near by, then just skip this feature
      // E.g. this is a non-planar feature that we should reject!!
      if (matches.empty())
        continue;
      // PRINT_ERROR(RED "[PLANE]: feature %zu matched to %zu other features (%zu norms)!!\n" RESET, featid, matches.size(), norms.size());

      // Then see if any of these features we match to are already classified as a plane
      // If they are, then we should merge them all to have the smallest plane id
      int min_planeid = -1;
      if (hist_feat_to_plane.find(featid) != hist_feat_to_plane.end()) {
        min_planeid = (int)hist_feat_to_plane.at(featid);
      }
      for (auto const &featid_close : matches) {
        if (hist_feat_to_plane.find(featid_close) == hist_feat_to_plane.end())
          continue;
        size_t planeid = hist_feat_to_plane.at(featid_close);
        if (min_planeid == -1) {
          min_planeid = (int)planeid;
        } else {
          min_planeid = std::min(min_planeid, (int)planeid);
        }
      }
      if (min_planeid != -1) {

        // Lock data
        // std::lock_guard<std::mutex> lckv(mtx_hist_vars);

        // This function will change all features with the "old id"
        // To the new min plane id. Also change their connected and update plane history
        auto update_plane_ids = [&](int min_planeid, size_t oldplaneid) {
          // skip if nothing to do..
          if ((size_t)min_planeid == oldplaneid)
            return;
          // update features to point to this new plane
          for (auto const &featpair_close : hist_feat_to_plane) {
            if (featpair_close.second == oldplaneid) {
              hist_feat_to_plane[featpair_close.first] = min_planeid;
            }
          }
          // update merged plane history map
          hist_plane_to_oldplanes[min_planeid].insert(oldplaneid);
          if (hist_plane_to_oldplanes.find(oldplaneid) != hist_plane_to_oldplanes.end()) {
            for (auto const &tmpoldplaneid : hist_plane_to_oldplanes.at(oldplaneid)) {
              hist_plane_to_oldplanes[min_planeid].insert(tmpoldplaneid);
            }
            hist_plane_to_oldplanes.erase(oldplaneid);
          }
        };

        // MATCHES: Loop through all old planes and merge them into the new ID
        // MATCHES: Skip features that have not yet been added to the match map
        for (auto const &tmpfeatid : matches) {
          if (hist_feat_to_plane.find(tmpfeatid) == hist_feat_to_plane.end())
            continue;
          size_t oldplaneid = hist_feat_to_plane.at(tmpfeatid);
          update_plane_ids(min_planeid, oldplaneid);
        }

        // CURRENT: See if the current feature is changing its id
        // CURRENT: If so, then we should update it and its points
        if (hist_feat_to_plane.find(featid) != hist_feat_to_plane.end()) {
          size_t oldplaneid = hist_feat_to_plane.at(featid);
          update_plane_ids(min_planeid, oldplaneid);
        }

        // Debug print contents
        // for (auto const &tmp1 : hist_plane_to_oldplanes) {
        //  std::cout << "plane " << tmp1.first << " -> ";
        //  for (auto const &tmp2 : tmp1.second) {
        //    std::cout << tmp2 << ", ";
        //  }
        //  std::cout << std::endl;
        //}

        // Update the neighbors of the current feature
        // NOTE: this also creates an entry in the map if it does not exists
        for (auto const &featid_close : matches)
          hist_feat_to_plane[featid_close] = min_planeid;
        hist_feat_to_plane[featid] = min_planeid;
        done_verts_ids.insert(featid);
      }

      // If none of the features have a plane, then we will assign a new plane id and add them all to it!
      if (min_planeid == -1) {
        size_t temp = ++currplaneid;
        // std::lock_guard<std::mutex> lckv(mtx_hist_vars);
        for (auto const &featid_close : matches)
          hist_feat_to_plane[featid_close] = temp;
        hist_feat_to_plane[featid] = temp;
      }
    }
  }
  auto rTP4 = boost::posix_time::microsec_clock::local_time();

  //====================================================================
  // Spacial filtering of plane points
  //====================================================================

  // Make sure our option is valid
  assert(options.filter_num_feat > 1);

  // First build plane to feature global id (only active features)
  std::map<size_t, std::vector<size_t>> plane_to_feat;
  for (auto const &idpair : hist_feat_to_plane) {
    if (std::find(ids_left.begin(), ids_left.end(), idpair.first) != ids_left.end()) {
      size_t feat_id = idpair.first;
      size_t plane_id = idpair.second;
      plane_to_feat[plane_id].push_back(feat_id);
    }
  }

  // Spatial filter features for all planes
  // Calculate the N closest distance average for each feature
  // And for all the features in this plane, calculate the mean of this closet distance
  // then apply statistical filtering (which is similar to the consistency check z-test)
  for (auto const &idpairs : plane_to_feat) {

    // Make sure we have enough points for this plane
    std::vector<size_t> feat_ids = idpairs.second;
    if ((int)feat_ids.size() <= options.filter_num_feat)
      continue;

    // Create a vector of points (IKD tree input form)
    KD_TREE<ikdTree_PointType>::PointVector vec_points;
    vec_points.reserve(feat_ids.size());
    for (unsigned long &feat_id : feat_ids) {
      ikdTree_PointType p;
      p.x = (float)hist_feat_inG.at(feat_id)[0];
      p.y = (float)hist_feat_inG.at(feat_id)[1];
      p.z = (float)hist_feat_inG.at(feat_id)[2];
      p.id = feat_id;
      vec_points.push_back(p);
    }

    // Clean IKD-Tree and build with a new pointcloud
    // NOTE: it seems that creating the ikd each time can takes >10ms
    // NOTE: thus we go through this process of clearing the old one and re-using
    if (ikd_tree->initialized())
      ikd_tree->reset();
    ikd_tree->Build(vec_points);

    // Get average neighbor distance of each point using ikd-tree
    Eigen::ArrayXd N_closest_avg_distances = Eigen::ArrayXd::Zero(feat_ids.size(), 1);
    size_t cnt = 0;
    for (auto point : vec_points) {
      KD_TREE<ikdTree_PointType>::PointVector vec_points_local;
      vector<float> p_dist;
      ikd_tree->Nearest_Search(point, options.filter_num_feat + 1, vec_points_local, p_dist);
      assert((int)p_dist.size() == options.filter_num_feat + 1);
      p_dist.erase(p_dist.begin()); // erase the first element, because it is the distance to itself.
      N_closest_avg_distances(cnt) = std::accumulate(p_dist.begin(), p_dist.end(), 0.0) / (double)p_dist.size();
      cnt++;
    }

    // Now apply Z test to filter outliers if they are above the z-threshold value
    // If a feature fails, we just remove it from any plane association
    // https://en.wikipedia.org/wiki/Z-test
    double N_closest_avg_distances_avg = N_closest_avg_distances.mean();
    double N_closest_avg_distances_std =
        std::sqrt((N_closest_avg_distances - N_closest_avg_distances_avg).square().sum() / ((double)N_closest_avg_distances.size() - 1));
    int current_array_id = 0;
    for (auto const &feat_id : feat_ids) {
      double sigma = std::abs(N_closest_avg_distances(current_array_id) - N_closest_avg_distances_avg);
      sigma /= N_closest_avg_distances_std;
      // PRINT_ALL("[PLANE]: filter feature %zu -> has sigma = %.2f with %.2f threshold\n", feat_id, sigma, options.filter_z_thresh);
      if (sigma > options.filter_z_thresh) {
        hist_feat_to_plane.erase(feat_id);
      }
      current_array_id++;
    }
  }

  //====================================================================
  // Remove any planes that are not with an active feature set...
  //====================================================================

  {
    // create a list of planes which are currently being actively tracked
    // this can be found by grouping planes which are seen from the active ids of the system
    std::map<size_t, size_t> hist_feat_to_plane_tmp;
    std::map<size_t, std::set<size_t>> hist_plane_to_oldplanes_tmp;
    for (auto const &id : ids_left) {
      if (hist_feat_to_plane.find(id) != hist_feat_to_plane.end()) {
        hist_feat_to_plane_tmp.insert({id, hist_feat_to_plane.at(id)});
      }
    }
    // now loop through and remove any planes that have a very small number of features
    // this can happen if features go out of view, or we just had a single (or two) pairwise matches
    std::map<size_t, size_t> plane2featct;
    for (auto const &tmp : hist_feat_to_plane_tmp) {
      plane2featct[tmp.second]++;
    }
    hist_feat_to_plane_tmp.clear();
    for (auto const &id : ids_left) {
      if (hist_feat_to_plane.find(id) != hist_feat_to_plane.end() && plane2featct.at(hist_feat_to_plane.at(id)) > 3) {
        hist_feat_to_plane_tmp.insert({id, hist_feat_to_plane.at(id)});
      }
    }
    // now update the history of plane merges for those that are active and large!
    for (auto const &idpair : hist_feat_to_plane_tmp) {
      if (hist_plane_to_oldplanes.find(idpair.second) != hist_plane_to_oldplanes.end()) {
        hist_plane_to_oldplanes_tmp.insert({idpair.second, hist_plane_to_oldplanes.at(idpair.second)});
      }
    }
    // std::lock_guard<std::mutex> lckv(mtx_hist_vars);
    hist_feat_to_plane = hist_feat_to_plane_tmp;
    hist_plane_to_oldplanes = hist_plane_to_oldplanes_tmp;
  }

  // Finally save the display
  {
    // std::lock_guard<std::mutex> lckv(mtx_last_vars);
    tri_img_last[cam_id] = img;
    tri_time_last[cam_id] = time;
    tri_mask_last[cam_id] = mask;
    tri_cdt_tri[cam_id] = cdt.triangles;
    tri_cdt_tri_norms[cam_id] = tri_normals_inG;
    tri_cdt_verts[cam_id] = tri_verts;
    tri_cdt_vert_ids[cam_id] = ids_left;
    tri_cdt_vert_pts[cam_id] = pts_left;
  }

  // Debug print
  auto rTP5 = boost::posix_time::microsec_clock::local_time();
  PRINT_ALL("[PLANE]: %.4f sec feature triangulation\n", (rTP2 - rTP1).total_microseconds() * 1e-6);
  PRINT_ALL("[PLANE]: %.4f sec for CDT triangulation\n", (rTP3 - rTP2).total_microseconds() * 1e-6);
  PRINT_ALL("[PLANE]: %.4f sec norm / plane merging\n", (rTP4 - rTP3).total_microseconds() * 1e-6);
  PRINT_ALL("[PLANE]: %.4f seconds for spatial filter\n", (rTP5 - rTP4).total_microseconds() * 1e-6);
  PRINT_ALL("[PLANE]: %.4f seconds for total\n", (rTP5 - rTP1).total_microseconds() * 1e-6);
  _tracking_time_triangulation = (rTP2 - rTP1).total_microseconds() * 1e-6;
  _tracking_time_delaunay = (rTP3 - rTP2).total_microseconds() * 1e-6;
  _tracking_time_matching = (rTP5 - rTP3).total_microseconds() * 1e-6;
  _tracking_time_total = (rTP5 - rTP1).total_microseconds() * 1e-6;
}

Eigen::Vector3d TrackPlane::avg_norm(const std::vector<Eigen::Vector3d> &norms) const {

  // TODO: should this be rotation averaging?
  // TODO: norms can be flipped frame-to-frame, need to handle this...
  // TODO: https://web.archive.org/web/20130531101356/http://www.emeyex.com/site/tuts/VertexNormals.pdf

  // Return if nothing to do
  if (norms.empty())
    return Eigen::Vector3d::Zero();

  // Compute the mean norm
  int count = 0;
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  for (auto const &norm : norms) {
    if (norm.norm() <= 0)
      continue;
    sum += norm;
    count++;
  }
  sum /= sum.norm(); // re-normalize

  // Return if we don't have enough to compute a variance
  if (count < 2)
    return Eigen::Vector3d::Zero();

  // Compute our sample norm variance
  // Eigen::Matrix3d var = Eigen::Matrix3d::Zero();
  double max_deg = 0.0;
  double var_deg = 0.0;
  for (auto const &norm : norms) {
    if (norm.norm() <= 0)
      continue;
    // Eigen::Vector3d diff = (norm - sum);
    // var += diff * diff.transpose();
    double diff_deg = 180.0 / M_PI * acos(norm.dot(sum));
    var_deg += (diff_deg - 0.0) * (diff_deg - 0.0);
    max_deg = std::max(max_deg, diff_deg);
  }
  // var /= (count - 1);
  var_deg /= (count - 1);

  // Debug print
  // std::cout << "average = " << sum.transpose() << std::endl;
  // std::cout << "var = " << var.diagonal().cwiseSqrt().transpose() << std::endl;
  // std::cout << "var_deg = " << std::sqrt(var_deg) << " | max_deg = " << max_deg << std::endl;
  if (std::sqrt(var_deg) > options.max_norm_avg_var || max_deg > options.max_norm_avg_max)
    return Eigen::Vector3d::Zero();
  return sum;
}

void TrackPlane::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                             std::vector<size_t> &ids0) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less than grid_px_size points away then existing features
  cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                      (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x);
    int y_grid = std::floor(kpt.pt.y / size_y);
    if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  // If we don't need any features, just return
  double min_feat_percent = 0.95;
  int num_featsneeded = num_features - (int)pts0.size();
  if (num_featsneeded < num_features * (1.0 - min_feat_percent))
    return;

  // Flip the mask
  // cv::Mat mask0_updated_inverse;
  // cv::bitwise_not(mask0_updated, mask0_updated_inverse);
  //  std::vector<cv::Point2f> pts0_ext_f;
  //  std::vector<cv::KeyPoint> pts0_ext;
  //  cv::goodFeaturesToTrack(img0pyr.at(0), pts0_ext_f, num_featsneeded, 0.01, min_px_dist, mask0_updated_inverse);
  //  for (auto const &pt : pts0_ext_f) {
  //    cv::KeyPoint kpt;
  //    kpt.pt = pt;
  //    pts0_ext.push_back(kpt);
  //  }
  std::vector<cv::KeyPoint> pts0_ext;
  Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // Now, reject features that are close a current feature
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : pts0_ext) {
    // Check that it is in bounds
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
      continue;
    // See if there is a point at this location
    if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
      continue;
    // Else lets add it!
    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Loop through and record only ones that are valid
  // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
  // NOTE: this is due to the fact that we select update features based on feat id
  // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
  // NOTE: not sure how to remove... maybe a better way?
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // update the uv coordinates
    kpts0_new.at(i).pt = pts0_new.at(i);
    // append the new uv coordinate
    pts0.push_back(kpts0_new.at(i));
    // move id foward and append this new point
    size_t temp = ++currid;
    ids0.push_back(temp);
  }
}

void TrackPlane::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr,
                                  std::vector<cv::KeyPoint> &kpts0, std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1,
                                  std::vector<uchar> &mask_out) {

  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // If we don't have enough points for ransac just return empty
  // We set the mask to be all zeros since all points failed RANSAC
  if (pts0.size() < 10) {
    for (size_t i = 0; i < pts0.size(); i++)
      mask_out.push_back((uchar)0);
    return;
  }

  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

  // Normalize these points, so we can then do ransac
  // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
  std::vector<cv::Point2f> pts0_n, pts1_n;
  for (size_t i = 0; i < pts0.size(); i++) {
    pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
    pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
  }

  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
  std::vector<uchar> mask_rsc;
  double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
  double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

  // Loop through and record only ones that are valid
  for (size_t i = 0; i < mask_klt.size(); i++) {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
    mask_out.push_back(mask);
  }

  // Copy back the updated positions
  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}

void TrackPlane::remove_feats(std::vector<size_t> &ids) {

  // Remove any old features that are not seen in this frame
  // std::lock_guard<std::mutex> lckv(mtx_hist_vars);
  std::map<size_t, Eigen::Vector3d> tmp_hist_feat_inG;
  std::map<size_t, Eigen::Matrix3d> tmp_hist_feat_linsys_A;
  std::map<size_t, Eigen::Vector3d> tmp_hist_feat_linsys_b;
  std::map<size_t, int> tmp_hist_feat_linsys_count;
  std::map<size_t, std::vector<Eigen::Vector3d>> tmp_hist_feat_norms_inG;
  for (auto const &id : ids) {
    if (hist_feat_inG.find(id) != hist_feat_inG.end())
      tmp_hist_feat_inG[id] = hist_feat_inG[id];
    if (hist_feat_linsys_A.find(id) != hist_feat_linsys_A.end())
      tmp_hist_feat_linsys_A[id] = hist_feat_linsys_A[id];
    if (hist_feat_linsys_b.find(id) != hist_feat_linsys_b.end())
      tmp_hist_feat_linsys_b[id] = hist_feat_linsys_b[id];
    if (hist_feat_linsys_count.find(id) != hist_feat_linsys_count.end())
      tmp_hist_feat_linsys_count[id] = hist_feat_linsys_count[id];
    if (hist_feat_norms_inG.find(id) != hist_feat_norms_inG.end())
      tmp_hist_feat_norms_inG[id] = hist_feat_norms_inG[id];
  }
  hist_feat_inG = tmp_hist_feat_inG;
  hist_feat_linsys_A = tmp_hist_feat_linsys_A;
  hist_feat_linsys_b = tmp_hist_feat_linsys_b;
  hist_feat_linsys_count = tmp_hist_feat_linsys_count;
  hist_feat_norms_inG = tmp_hist_feat_norms_inG;
}
