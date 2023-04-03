#!/usr/bin/env bash

# Source our workspace directory to load ENV variables
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/../../../devel/setup.bash

#=============================================================
#=============================================================
#=============================================================

# dataset locations
bagnames=(
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
)

do_msckf_plane=(
    "false"
    "true"
)

do_slam_plane=(
    "false"
    "true"
)

sigma_c=(
#    "0.001"
    "0.010"
#    "0.030"
)

max_slam=(
    "00"
    "15"
#    "30"
)

# location to save log files into
bag_path="/home/patrick/datasets/"
base_path="/home/patrick/workspace/catkin_ws_plane/src/ov_plane/results/"
save_path1="$base_path/exp_rpng_plane/algorithms"
save_path2="$base_path/exp_rpng_plane/timings"
save_path3="$base_path/exp_rpng_plane/trackings"
sufix=""


#=============================================================
#=============================================================
#=============================================================

# Loop through all datasets
for i in "${!bagnames[@]}"; do

# Loop through all modes
for a in "${!do_msckf_plane[@]}"; do
for b in "${!sigma_c[@]}"; do
for c in "${!max_slam[@]}"; do
for d in "${!do_slam_plane[@]}"; do

# Monte Carlo runs for this dataset
# If you want more runs, change the below loop
for j in {00..00}; do

# for none loop we only need to run the first one
if [[ "${do_msckf_plane[a]}" == "false" && "$b" != "0" ]]; then
  break
fi
if [[ "${do_msckf_plane[a]}" == "false" && "${do_slam_plane[d]}" == "true" && "$d" != "0" ]]; then
  break
fi

# start timing
start_time="$(date -u +%s)"
if [ "${max_slam[c]}" == "00" ]; then
  foldername="M-PT"
else
  foldername="MS-PT(${max_slam[c]})"
fi
if [[ "${do_msckf_plane[a]}" == "true" && "${do_slam_plane[d]}" == "false"  ]]; then
  foldername="${foldername} - M-PL(${sigma_c[b]})"
fi
if [[ "${do_msckf_plane[a]}" == "true" && "${do_slam_plane[d]}" == "true"  ]]; then
  foldername="${foldername} - MS-PL(${sigma_c[b]})"
fi
foldername="${foldername}${sufix}"
filename_est="$save_path1/$foldername/${bagnames[i]}/${j}_estimate.txt"
filename_time="$save_path2/$foldername/${bagnames[i]}/${j}_timing.txt"
filename_track="$save_path3/$foldername/${bagnames[i]}/${j}_tracking.txt"


# run our ROS launch file (note we send console output to terminator)
roslaunch ov_plane serial.launch \
  verbosity:="WARNING" \
  max_cameras:="1" \
  use_stereo:="false" \
  config:="rpng_plane" \
  object:="" \
  dataset:="${bagnames[i]}" \
  bag:="$bag_path/rpng_plane/${bagnames[i]}.bag" \
  bag_start:="${bagstarttimes[i]}" \
  use_plane_constraint:="${do_msckf_plane[a]}" \
  use_plane_constraint_msckf:="${do_msckf_plane[a]}" \
  use_plane_constraint_slamu:="${do_msckf_plane[a]}" \
  use_plane_constraint_slamd:="${do_msckf_plane[a]}" \
  use_plane_slam_feats:="${do_slam_plane[d]}" \
  use_refine_plane_feat:="true" \
  sigma_constraint:="${sigma_c[b]}" \
  const_init_multi:="1.00" \
  const_init_chi2:="1.00" \
  num_pts:="200" \
  num_slam:="${max_slam[c]}" \
  dobag:="true" \
  bag_rate:="2" \
  dosave:="true" \
  path_est:="$filename_est" \
  dotime:="true" \
  path_time:="$filename_time" \
  dotrackinfo:="true" \
  path_track:="$filename_track" \
  dolivetraj:="true" &> /dev/null


# print out the time elapsed
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "BASH: $foldername - ${bagnames[i]} - run $j took $elapsed seconds";

done

done
done
done
done

done

