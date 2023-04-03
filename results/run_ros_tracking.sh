#!/usr/bin/env bash

# Source our workspace directory to load ENV variables
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/../../../devel/setup.bash

#=============================================================
#=============================================================
#=============================================================

config=(
  "euroc_mav"
  "euroc_mav"
  "euroc_mav"
  "euroc_mav"
  "euroc_mav"
  "euroc_mav"
  "rpng_plane"
  "rpng_plane"
  "rpng_plane"
  "rpng_plane"
  "rpng_plane"
  "rpng_plane"
  "rpng_plane"
  "rpng_plane"
)


# dataset locations
bagnames=(
  "V1_01_easy"
  "V1_02_medium"
  "V1_03_difficult"
  "V2_01_easy"
  "V2_02_medium"
  "V2_03_difficult"
  "table_01"
  "table_02"
  "table_03"
  "table_04"
  "table_05"
  "table_06"
  "table_07"
  "table_08"
)

# how far we should start into the dataset
# this can be used to skip the initial sections
bagstarttimes=(
  "0"
  "0"
  "0"
  "0"
  "0"
  "0"
  "0"
  "0"
  "0"
  "0"
  "0"
  "0"
  "0"
  "0"
)

do_slam_plane="true"
sigma_c="0.010"
max_slam="30"



# location to save log files into
bag_path="/home/patrick/datasets/"
base_path="/home/patrick/workspace/catkin_ws_plane/src/ov_plane/results/"
save_path3="$base_path/exp_track_stats/trackings"


#=============================================================
#=============================================================
#=============================================================

# Loop through all datasets
for i in "${!bagnames[@]}"; do


# start timing
start_time="$(date -u +%s)"
filename_track="$save_path3/${config[i]}_${bagnames[i]}.txt"


# run our ROS launch file (note we send console output to terminator)
roslaunch ov_plane serial.launch \
  verbosity:="WARNING" \
  max_cameras:="1" \
  use_stereo:="false" \
  config:="${config[i]}" \
  object:="" \
  dataset:="${bagnames[i]}" \
  bag:="$bag_path/${config[i]}/${bagnames[i]}.bag" \
  bag_start:="${bagstarttimes[i]}" \
  use_plane_constraint:="true" \
  use_plane_constraint_msckf:="true" \
  use_plane_constraint_slamu:="true" \
  use_plane_constraint_slamd:="true" \
  use_plane_slam_feats:="$do_slam_plane" \
  use_refine_plane_feat:="true" \
  sigma_constraint:="$sigma_c" \
  const_init_multi:="1.00" \
  const_init_chi2:="1.00" \
  num_pts:="200" \
  num_slam:="${max_slam[c]}" \
  dobag:="true" \
  bag_rate:="2" \
  dosave:="false" \
  dotime:="false" \
  dotrackinfo:="true" \
  path_track:="$filename_track" \
  dolivetraj:="true" &> /dev/null



# print out the time elapsed
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "BASH: ${config[i]}_${bagnames[i]} - took $elapsed seconds";

done


