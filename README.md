# Internal Model-Based Nonlinear Model Predictive Optimization for Robust Agile Quadrotor Flight

<p align="center">
  <img src="IM-NMPO/fig/Framework_review.png" width="60%" alt="IM-NMPO Framework">
</p>

<em>Figure 1: (A) IM-NMPO framework. (B) Simulation and real-world experiments encompassing various disturbances, including unknown payloads, unknown persistent fan-induced and time-varying gusts, across quadrotors with different wheelbases (450mm, 330mm, and 250mm).</em>

## Simulation Reproduction

The Docker image contains the ROS Noetic environment and Python dependencies
used for the simulation experiments.

The commands below reproduce the default comparison between the proposed
IM-NMPO controller and the NMPC baseline under high-frequency periodic input
disturbances.

## 1. Clone The Repository

```bash
git clone https://github.com/Fly-to-the-Mars/IM_NMPO.git
cd IM_NMPO
```

Run all remaining commands from this repository root.

## 2. Load The Docker Image

```bash
wget -O im-nmpo_noetic.gz \
  https://github.com/Fly-to-the-Mars/IM_NMPO/releases/latest/download/im-nmpo_noetic.gz

wget -O im-nmpo_noetic.gz.sha256 \
  https://github.com/Fly-to-the-Mars/IM_NMPO/releases/latest/download/im-nmpo_noetic.gz.sha256
```

```bash
sha256sum -c im-nmpo_noetic.gz.sha256
docker load -i im-nmpo_noetic.gz
docker image ls im-nmpo
```

## 3. Run IM-NMPO

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

## 4. Run The NMPC Baseline

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

## 5. Check The Results

```bash
find docker_results -name 'trajectory_plot_*.png' -print
```

```text
docker_results/im_nmpo/trajectory_plot_<timestamp>.png
docker_results/nmpc/trajectory_plot_<timestamp>.png
```

## 6. Run Other Disturbance Cases

To reproduce another disturbance scenario, replace
`disturbance_case:=hf_periodic` in the commands above.

Available disturbance cases:

```text
none
constant
lf_periodic
hf_periodic
ou
pink_noise
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
| `use_rviz:=false` | headless execution |
| `plot_trajectory:=true` | save the trajectory plot |
| `plot_output_dir:=/root/im_nmpo_ws/results` | plot output directory inside Docker |

## Example Tracking Results

Trajectory tracking subject to periodic external disturbances:

<p align="center">
  <img src="IM-NMPO/fig/tracking_peridicdisturbance.png" width="90%" alt="The proposed IM-NMPO (Left) VS. the NMPC Baseline (Right)">
  <br>
  <em>Figure 2: The proposed IM-NMPO (left) and the NMPC baseline (right).</em>
</p>

Trajectory tracking subject to constant external disturbances:

<p align="center">
  <img src="IM-NMPO/fig/tracking_constantdisturbance.png" width="90%" alt="The proposed IM-NMPO (Left) VS. the NMPC Baseline (Right)">
  <br>
  <em>Figure 3: The proposed IM-NMPO (left) and the NMPC baseline (right).</em>
</p>
