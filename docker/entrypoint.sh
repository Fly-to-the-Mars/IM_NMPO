#!/usr/bin/env bash
set -e

source_ros_environment() {
  source /opt/ros/noetic/setup.bash

  if [ -f "${IM_NMPO_WS:-/root/im_nmpo_ws}/devel/setup.bash" ]; then
    source "${IM_NMPO_WS:-/root/im_nmpo_ws}/devel/setup.bash"
  fi
}

source_ros_environment

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_IP="${ROS_IP:-127.0.0.1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd "${IM_NMPO_WS:-/root/im_nmpo_ws}"
exec "$@"
