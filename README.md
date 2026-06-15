<h1 align="center">
Robust Optimal Agile Flight of Quadrotors: An Internal Model-Based Nonlinear Model Predictive Optimization Approach
</h1>

<p align="center">
  <img src="IM-NMPO/fig/Framework_review.png" width="70%" alt="IM-NMPO framework">
</p>

This repository contains the ROS simulation code for reproducing the agile
quadrotor tracking experiments of the IM-NMPO method.

The default reproduction compares:

- `ctrl_flag:=1`: proposed IM-NMPO controller
- `ctrl_flag:=2`: NMPC baseline

under the high-frequency periodic input disturbance setting
`disturbance_case:=hf_periodic`.

## Project Overview

```text
IM-NMPO/
|-- IM-NMPO/                  # im_nmpo ROS package
|   |-- launch/               # launch files
|   |-- script/               # controllers, trajectory publisher, plotting
|   `-- fig/                  # framework and result figures
|-- px4_bridge/               # quadrotor simulator and bridge messages
`-- docker/                   # Docker environment
```

## Installation

Two execution paths are provided. The Docker path reproduces the experiments
with the packaged environment. The local path uses an existing ROS Noetic
installation.

### Option 1: Docker

```bash
git clone https://github.com/Fly-to-the-Mars/IM_NMPO.git
cd IM_NMPO
```

Download and load the prebuilt image:

```bash
wget -O im-nmpo_noetic.gz \
  https://github.com/Fly-to-the-Mars/IM_NMPO/releases/latest/download/im-nmpo_noetic.gz

wget -O im-nmpo_noetic.gz.sha256 \
  https://github.com/Fly-to-the-Mars/IM_NMPO/releases/latest/download/im-nmpo_noetic.gz.sha256

sha256sum -c im-nmpo_noetic.gz.sha256
docker load -i im-nmpo_noetic.gz
```

Alternatively, build the image from source:

```bash
docker build -t im-nmpo:noetic -f docker/Dockerfile .
```

### Option 2: Local ROS Noetic

Install ROS Noetic on Ubuntu 20.04, then create a catkin workspace:

```bash
sudo apt update
sudo apt install -y \
  python3-pip python3-numpy python3-scipy python3-matplotlib python3-yaml \
  ros-noetic-geometry-msgs ros-noetic-mavlink ros-noetic-nav-msgs \
  ros-noetic-rviz ros-noetic-sensor-msgs ros-noetic-std-msgs \
  ros-noetic-tf ros-noetic-visualization-msgs

pip3 install casadi==3.6.5

mkdir -p ~/IM_NMPO_ws/src
cd ~/IM_NMPO_ws/src
git clone https://github.com/Fly-to-the-Mars/IM_NMPO.git

cd ~/IM_NMPO_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Reproduce The Default Simulation

The default setting uses angular-rate commands and a high-frequency periodic
matched input disturbance.

### Docker

Run the proposed IM-NMPO controller:

```bash
mkdir -p docker_results/im_nmpo

timeout --foreground -s INT 45s docker run --rm --net=host \
  -v "$PWD/docker_results/im_nmpo:/root/im_nmpo_ws/results" \
  im-nmpo:noetic \
  roslaunch im_nmpo robust_im_nmpo.launch \
    ctrl_flag:=1 command_mode:=rate \
    disturbance_case:=hf_periodic \
    disturbance_application:=input disturbance_time_base:=relative \
    use_rviz:=false plot_trajectory:=true \
    plot_output_dir:=/root/im_nmpo_ws/results \
  || test $? -eq 124
```

Run the NMPC baseline:

```bash
mkdir -p docker_results/nmpc

timeout --foreground -s INT 45s docker run --rm --net=host \
  -v "$PWD/docker_results/nmpc:/root/im_nmpo_ws/results" \
  im-nmpo:noetic \
  roslaunch im_nmpo robust_im_nmpo.launch \
    ctrl_flag:=2 command_mode:=rate \
    disturbance_case:=hf_periodic \
    disturbance_application:=input disturbance_time_base:=relative \
    use_rviz:=false plot_trajectory:=true \
    plot_output_dir:=/root/im_nmpo_ws/results \
  || test $? -eq 124
```

The Docker plots are written to:

```bash
find docker_results -name 'trajectory_plot_*.png' -print
```

### Local ROS

```bash
source ~/IM_NMPO_ws/devel/setup.bash
```

Run the proposed IM-NMPO controller:

```bash
roslaunch im_nmpo robust_im_nmpo.launch
```

Run the NMPC baseline:

```bash
roslaunch im_nmpo robust_im_nmpo.launch ctrl_flag:=2
```

For terminal-only execution, add `use_rviz:=false`.

The local plots are written to:

```bash
find ~/IM_NMPO_ws/src/IM_NMPO/IM-NMPO/fig/sim_results \
  -name 'trajectory_plot_*.png' -print
```

## Disturbance Cases

Replace `disturbance_case:=hf_periodic` with one of:

```text
none
constant
lf_periodic
hf_periodic
ou
pink_noise
```

Example:

```bash
roslaunch im_nmpo robust_im_nmpo.launch \
  ctrl_flag:=1 \
  command_mode:=rate \
  disturbance_case:=ou \
  imc_frequency_adaptation:=true
```

## Launch Parameters

| Parameter | Description |
| --- | --- |
| `ctrl_flag:=1` | proposed IM-NMPO controller |
| `ctrl_flag:=2` | NMPC baseline |
| `command_mode:=rate` | angular-rate command interface |
| `disturbance_case:=hf_periodic` | default disturbance scenario |
| `disturbance_application:=input` | matched input disturbance |
| `disturbance_time_base:=relative` | disturbance timing relative to trajectory start |
| `imc_frequency_adaptation:=true` | online frequency adaptation for non-harmonic disturbances |
| `use_rviz:=false` | headless execution |
| `plot_trajectory:=true` | save trajectory plots |
| `plot_output_dir:=...` | output directory for trajectory plots |

## Example Results

Trajectory tracking under periodic external disturbances:

<p align="center">
  <img src="IM-NMPO/fig/tracking_peridicdisturbance.png" width="90%" alt="Trajectory tracking under periodic disturbances">
  <br>
  <em>Proposed IM-NMPO controller (left) and NMPC baseline (right).</em>
</p>

Trajectory tracking under constant external disturbances:

<p align="center">
  <img src="IM-NMPO/fig/tracking_constantdisturbance.png" width="90%" alt="Trajectory tracking under constant disturbances">
  <br>
  <em>Proposed IM-NMPO controller (left) and NMPC baseline (right).</em>
</p>
