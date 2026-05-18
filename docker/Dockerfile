FROM ros:noetic-ros-base

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1
ENV IM_NMPO_WS=/root/im_nmpo_ws

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-tk \
    python3-yaml \
    ros-noetic-geometry-msgs \
    ros-noetic-mavlink \
    ros-noetic-nav-msgs \
    ros-noetic-rviz \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-tf \
    ros-noetic-visualization-msgs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir casadi==3.6.5

WORKDIR ${IM_NMPO_WS}
RUN mkdir -p src

COPY . ${IM_NMPO_WS}/src/IM_NMPO

RUN source /opt/ros/noetic/setup.bash \
    && catkin_make \
    && chmod +x ${IM_NMPO_WS}/src/IM_NMPO/docker/entrypoint.sh

ENTRYPOINT ["/root/im_nmpo_ws/src/IM_NMPO/docker/entrypoint.sh"]
CMD ["bash"]
