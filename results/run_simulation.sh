#!/usr/bin/env bash

# Source our workspace directory to load ENV variables
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/../../../devel/setup.bash

#=============================================================
#=============================================================
#=============================================================


# dataset locations
datasets=(
  "udel_arl_short"
#  "udel_room_05"
#  "udel_room_05_short"
)

# how far we should start into the dataset
# this can be used to skip the initial sections
do_msckf_plane=(
    "false"
    "true"
)

do_slam_plane=(
    "false"
    "true"
)


sigma_c=(
    "0.001"
#    "0.01"
#    "0.05"
#    "0.10"
)

max_slam=(
    "00"
    "15"
#    "30"
)

# location to save log files into
base_path="/home/patrick/workspace/catkin_ws_plane/src/ov_plane/results"
save_path1="$base_path/sim_general/algorithms"
save_path_gt="$base_path/sim_general/truths"
save_path2="$base_path/sim_general/timings"
sufix=""


#=============================================================
#=============================================================
#=============================================================

# Loop through all datasets
for i in "${!datasets[@]}"; do
for a in "${!do_msckf_plane[@]}"; do
for b in "${!sigma_c[@]}"; do
for c in "${!max_slam[@]}"; do
for d in "${!do_slam_plane[@]}"; do

# groundtruth file save location
# ONLY SAVE FOR NONE SO WE DON'T KEEP OVERWRITTING IT!
if [[ "${do_msckf_plane[a]}" == "false" ]]; then
  filename_gt="$save_path_gt/${datasets[i]}.txt"
else
  filename_gt="/tmp/groundtruth.txt"
fi

# Monte Carlo runs for this dataset
# If you want more runs, change the below loop
for j in {00..19}; do

# for none loop we only need to run the first one
if [[ "${do_msckf_plane[a]}" == "false" && "$b" != "0" ]]; then
  break
fi
if [[ "${do_msckf_plane[a]}" == "false" && "${do_slam_plane[d]}" == "true" ]]; then
  break
fi

# start timing
start_time="$(date -u +%s)"
foldername="${max_slam[c]}slam"
if [ "${do_msckf_plane[a]}" == "true" ]
then
  foldername="${max_slam[c]}slam_${sigma_c[b]}plane"
fi
if [[ "${do_msckf_plane[a]}" == "true" && "${do_slam_plane[d]}" == "true" ]]
then
  foldername="${foldername}_slam"
fi

foldername="${foldername}${sufix}"
filename_est="$save_path1/$foldername/${datasets[i]}/${j}_estimate.txt"
filename_time="$save_path2/$foldername/${datasets[i]}/${j}_timing.txt"

roslaunch ov_plane simulation.launch \
  verbosity:="WARNING" \
  seed:="$((10#$j + 5))" \
  dataset:="${datasets[i]}.txt" \
  use_plane_constraint:="${do_msckf_plane[a]}" \
  use_plane_constraint_msckf:="${do_msckf_plane[a]}" \
  use_plane_constraint_slamu:="${do_msckf_plane[a]}" \
  use_plane_constraint_slamd:="${do_msckf_plane[a]}" \
  use_plane_slam_feats:="${do_slam_plane[d]}" \
  use_refine_plane_feat:="true" \
  use_groundtruths:="false" \
  sigma_constraint:="${sigma_c[b]}" \
  const_init_multi:="1.00" \
  num_pts:="0" \
  num_pts_plane:="150" \
  num_slam:="${max_slam[c]}" \
  dosave_pose:="true" \
  path_est:="$filename_est" \
  path_gt:="$filename_gt" \
  dotime:="true" \
  path_time:="$filename_time" &> /dev/null

# print out the time elapsed
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "BASH: $foldername - ${datasets[i]} - run $j took $elapsed seconds";

done
done
done
done
done
done


