cmake_minimum_required(VERSION 3.3)
project(ov_plane)

# Include libraries (if we don't have opencv 4, then fallback to opencv 3)
# The OpenCV version needs to match the one used by cv_bridge otherwise you will get a segmentation fault!
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif ()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(Ceres REQUIRED)
message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION} " | CERES: " ${Ceres_VERSION})

# We need c++14 for ROS2, thus just require it for everybody
# NOTE: To future self, hope this isn't an issue...
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -fno-omit-frame-pointer")

# Find ROS build system
find_package(catkin REQUIRED COMPONENTS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs image_transport cv_bridge ov_core ov_init ov_eval)

# Describe ROS project
catkin_package(
        CATKIN_DEPENDS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs image_transport cv_bridge ov_core ov_init ov_eval
        INCLUDE_DIRS src/
        LIBRARIES ov_plane_lib
)
add_definitions(-DROS_AVAILABLE=1)

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CERES_LIBRARIES}
        ${catkin_LIBRARIES}
)

##################################################
# Make the shared library
##################################################

# thirdparty ikd library!
file(GLOB_RECURSE source_ikd ../thirdparty/ikd/ikd_tree.cpp)
add_library(thirdparty_ikd SHARED ${source_ikd})
target_link_libraries(thirdparty_ikd ${thirdparty_libraries})
set_target_properties(thirdparty_ikd PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(thirdparty_ikd PUBLIC ../thirdparty/ikd/)
list(APPEND thirdparty_libraries thirdparty_ikd)

# thirdparty render library!
file(GLOB_RECURSE source_render ../thirdparty/render/render_model.cpp)
add_library(thirdparty_render SHARED ${source_render})
target_link_libraries(thirdparty_render ${thirdparty_libraries})
set_target_properties(thirdparty_render PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(thirdparty_render PUBLIC ../thirdparty/render/)
list(APPEND thirdparty_libraries thirdparty_render)

# thirdparty cdt library!
file(GLOB_RECURSE source_cdt ../thirdparty/cdt/CDT.hpp ../thirdparty/cdt/CDTUtils.hpp ../thirdparty/cdt/remove_at.hpp ../thirdparty/cdt/Triangulation.hpp)
add_library(thirdparty_cdt SHARED ${source_cdt})
target_link_libraries(thirdparty_cdt ${thirdparty_libraries})
set_target_properties(thirdparty_cdt PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(thirdparty_cdt PUBLIC ../thirdparty/cdt/)
list(APPEND thirdparty_libraries thirdparty_cdt)

# main source code now
list(APPEND LIBRARY_SOURCES
        src/ceres/Factor_PointOnPlane.cpp
        src/core/VioManager.cpp
        src/core/VioManagerHelper.cpp
        src/sim/Simulator.cpp
        src/state/State.cpp
        src/state/StateHelper.cpp
        src/state/Propagator.cpp
        src/track_plane/TrackPlane.cpp
        src/track_plane/PlaneFitting.cpp
        src/update/UpdaterHelper.cpp
        src/update/UpdaterMSCKF.cpp
        src/update/UpdaterPlane.cpp
        src/update/UpdaterSLAM.cpp
        src/update/UpdaterZeroVelocity.cpp
        src/ros/ROS1Visualizer.cpp
        src/ros/ROSVisualizerHelper.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_plane_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(ov_plane_lib ${thirdparty_libraries})
target_include_directories(ov_plane_lib PUBLIC src/)
install(TARGETS ov_plane_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

##################################################
# Make binary files!
##################################################


add_executable(ros1_serial_msckf src/ros1_serial_msckf.cpp)
target_link_libraries(ros1_serial_msckf ov_plane_lib ${thirdparty_libraries})
install(TARGETS ros1_serial_msckf
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(run_simulation src/run_simulation.cpp)
target_link_libraries(run_simulation ov_plane_lib ${thirdparty_libraries})
install(TARGETS run_simulation
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


add_executable(timing_custom src/timing_custom.cpp)
target_link_libraries(timing_custom ov_plane_lib ${thirdparty_libraries})
install(TARGETS timing_custom
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

##################################################
# Launch files!
##################################################

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)









