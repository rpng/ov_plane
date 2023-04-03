
# Source our workspace directory to load ENV variables
source /home/patrick/workspace/catkin_ws_plane/devel/setup.bash


# get directory
BASEDIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo "$BASEDIR"


for d in $BASEDIR/trackings/* ; do
    files=($d/$dataset/*)
    echo $d
    first_run=${files[0]}
    new_run="$export_folder/$d"
    cmd_files="$new_run $cmd_files"
done

rosrun ov_plane timing_custom $cmd_files > "$BASEDIR/output_tracking.txt"

