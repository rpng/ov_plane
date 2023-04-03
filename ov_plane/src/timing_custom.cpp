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

#include <Eigen/Eigen>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <string>

#include "utils/Loader.h"
#include "utils/Statistics.h"
#include "utils/colors.h"
#include "utils/print.h"

int main(int argc, char **argv) {

  // Verbosity setting
  ov_core::Printer::setPrintLevel("INFO");

  // Ensure we have a path
  if (argc < 2) {
    PRINT_ERROR(RED "ERROR: Please specify a timing file\n" RESET);
    PRINT_ERROR(RED "ERROR: ./timing_custom <file_time.txt>\n" RESET);
    PRINT_ERROR(RED "ERROR: rosrun ov_eval timing_custom <file_times1.txt>\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Skip zeros for feat/plane
  // feat/plane,num plane,track length(avg),track length(std),track length(max),
  // num constraint updates,state planes,
  // triangulation,delaunay,matching,total
  std::vector<bool> skip_zeros = {true, false, false, false, false, false, false, false, false, false, false};
  std::vector<bool> print_last = {false, false, true, true, true, false, false, false, false, false, false};
  std::vector<double> scale = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0, 1000.0, 1000.0, 1000.0};
  std::vector<bool> skip_print = {false, false, false, false, false, true, false, true, true, true, false};

  // Read in all our trajectories from file
  std::vector<std::string> names;
  std::vector<std::string> names_cols;
  std::vector<std::vector<std::pair<double, double>>> print_vals;
  for (int z = 1; z < argc; z++) {

    // Parse the name of this timing
    boost::filesystem::path path(argv[z]);
    std::string name = path.stem().string();
    PRINT_INFO("[TIME]: loading data for %s\n", name.c_str());

    // Load it!!
    std::vector<std::string> names_temp;
    std::vector<double> times;
    std::vector<Eigen::VectorXd> timing_values;
    ov_eval::Loader::load_timing_flamegraph(argv[z], names_temp, times, timing_values);
    PRINT_DEBUG("[TIME]: loaded %d timestamps from file (%d categories)!!\n", (int)times.size(), (int)names_temp.size());

    // Our categories
    std::vector<ov_eval::Statistics> stats;
    for (size_t i = 0; i < names_temp.size(); i++)
      stats.push_back(ov_eval::Statistics());

    // Loop through each and report the average timing information
    for (size_t i = 0; i < times.size(); i++) {
      for (size_t c = 0; c < names_temp.size(); c++) {
        assert((int)skip_zeros.size() == timing_values.at(i).rows());
        if (skip_zeros.at(c) && timing_values.at(i)(c) == 0.0) {
          // std::cout << timing_values.at(i).transpose() << std::endl;
          continue;
        }
        stats.at(c).timestamps.push_back(times.at(i));
        stats.at(c).values.push_back(scale.at(c) * timing_values.at(i)(c));
      }
    }

    // Now print the statistic for this run
    names_cols.clear();
    std::vector<std::pair<double, double>> values;
    for (size_t i = 0; i < names_temp.size(); i++) {
      if (print_last.at(i)) {
        PRINT_INFO("last_value = %.4f (%s)\n", stats.at(i).values.at(stats.at(i).values.size() - 1), names_temp.at(i).c_str());
        if (names_temp.at(i) == "track length(avg)" && !skip_print.at(i)) {
          names_cols.push_back(names_temp.at(i));
          values.push_back(
              {stats.at(i).values.at(stats.at(i).values.size() - 1), stats.at(i + 1).values.at(stats.at(i + 1).values.size() - 1)});
        }
      } else {
        stats.at(i).calculate();
        PRINT_INFO("mean_time = %.4f | std = %.4f | 99th = %.4f  | max = %.4f (%s)\n", stats.at(i).mean, stats.at(i).std,
                   stats.at(i).ninetynine, stats.at(i).max, names_temp.at(i).c_str());
        if (!skip_print.at(i)) {
          names_cols.push_back(names_temp.at(i));
          values.push_back({stats.at(i).mean, stats.at(i).std});
        }
      }
    }

    // Append the total stats to the big vector
    if (stats.empty()) {
      PRINT_ERROR(RED "[TIME]: unable to load any data.....\n" RESET);
    }

    // Save data
    names.push_back(name);
    print_vals.push_back(values);
  }

  // feat/plane,num plane,track length(avg),track length(std),track length(max),
  // num constraint updates,state planes,
  // triangulation,delaunay,matching,total
  PRINT_INFO("============================================\n");
  PRINT_INFO("TIMING LATEX TABLE\n");
  PRINT_INFO("============================================\n");
  for (size_t i = 0; i < names_cols.size(); i++) {
    std::string name = names_cols.at(i);
    boost::replace_all(name, "_", "\\_");
    PRINT_INFO(" & \\textbf{%s}", name.c_str());
  }
  PRINT_INFO(" \\\\\\hline\n");
  for (int i = (int)names.size() - 1; i >= 0; i--) {
    std::string algoname = names.at(i);
    boost::replace_all(algoname, "_", "\\_");
    PRINT_INFO(algoname.c_str());
    for (auto &vals : print_vals.at(i)) {
      PRINT_INFO(" & %.1f $\\pm$ %.1f", vals.first, vals.second);
    }
    PRINT_INFO(" \\\\\n");
  }

  // Done!
  return EXIT_SUCCESS;
}