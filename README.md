# Robust Optimal Planning and Control for Agile Flight of Quadrotors: An Internal Model-Based Nonlinear Model Predictive Optimization Approach

<p align="center">
  <img src="IM-NMPO/fig/Framework_review.png" width="60%" alt="IM-NMPO Framework">
</p>

<em>Figure 1: (A) IM-NMPO framework. (B) Simulation and real-world experiments encompassing various disturbances, including unknown payloads, unknown persistent fan-induced and time-varying gusts, across quadrotors with different wheelbases (450mm, 330mm, and 250mm).</em>

## Environment

Tested on Ubuntu 20.04 with ROS Noetic.

Required Python package:

```bash
pip3 install casadi==3.6.5
```

## Build From Source

Clone the repository into a catkin workspace and build it:

```bash
mkdir -p ~/im_nmpo_ws/src
cd ~/im_nmpo_ws/src
git clone <repository-url> IM_NMPO
cd ~/im_nmpo_ws
catkin_make
source devel/setup.bash
```

## Run A Comparison

The launch file exposes the same command interface for both controllers:

- `ctrl_flag:=1`: IM-NMPO
- `ctrl_flag:=2`: NMPC
- `command_mode:=rate`: angular-rate command interface
- `disturbance_case:=none`: no disturbance
- `disturbance_case:=constant`: constant torque disturbance
- `disturbance_case:=lf_periodic`: low-frequency periodic torque disturbance
- `disturbance_case:=hf_periodic`: high-frequency periodic torque disturbance
- `disturbance_case:=ou`: Ornstein-Uhlenbeck stochastic torque disturbance
- `disturbance_case:=pink_noise`: pink-noise torque disturbance

Run IM-NMPO under the high-frequency periodic disturbance:

```bash
roslaunch im_nmpo robust_im_nmpo.launch \
  ctrl_flag:=1 command_mode:=rate \
  disturbance_case:=hf_periodic \
  disturbance_application:=input disturbance_time_base:=relative \
  use_rviz:=true plot_trajectory:=true
```

Run the NMPC baseline under the same condition:

```bash
roslaunch im_nmpo robust_im_nmpo.launch \
  ctrl_flag:=2 command_mode:=rate \
  disturbance_case:=hf_periodic \
  disturbance_application:=input disturbance_time_base:=relative \
  use_rviz:=true plot_trajectory:=true
```

Stop each launch with `Ctrl-C` after the simulated trajectory finishes. A
trajectory plot is saved to:

```text
IM-NMPO/fig/sim_results/trajectory_plot_<timestamp>.png
```

For headless runs, disable RViz and keep trajectory plotting enabled:

```bash
roslaunch im_nmpo robust_im_nmpo.launch \
  ctrl_flag:=1 command_mode:=rate \
  disturbance_case:=lf_periodic \
  disturbance_application:=input disturbance_time_base:=relative \
  use_rviz:=false plot_trajectory:=true
```

For OU and pink-noise disturbances, IM-NMPO can update the effective internal
model frequency online from recent rotational tracking-error data:

```bash
roslaunch im_nmpo robust_im_nmpo.launch \
  ctrl_flag:=1 command_mode:=rate \
  disturbance_case:=ou \
  imc_frequency_adaptation:=true \
  disturbance_application:=input disturbance_time_base:=relative \
  use_rviz:=false plot_trajectory:=true
```

To compare another disturbance case, keep the controller arguments unchanged
and replace only `disturbance_case`.

## Docker

The repository includes a Docker environment with ROS Noetic and all required
simulation dependencies.

### Load An Image Archive

If an image archive is provided with a release, load it locally:

```bash
sha256sum -c im-nmpo_noetic_<date>.tar.gz.sha256
docker load -i im-nmpo_noetic_<date>.tar.gz
```

The image is tagged as:

```text
im-nmpo:noetic
```

Run IM-NMPO in the container and write plots to a host directory:

```bash
mkdir -p docker_results

docker run --rm -it --net=host \
  -v "$PWD/docker_results:/root/im_nmpo_ws/results" \
  im-nmpo:noetic \
  roslaunch im_nmpo robust_im_nmpo.launch \
    ctrl_flag:=1 command_mode:=rate \
    disturbance_case:=hf_periodic \
    disturbance_application:=input disturbance_time_base:=relative \
    use_rviz:=false plot_trajectory:=true \
    plot_output_dir:=/root/im_nmpo_ws/results
```

Run the NMPC baseline by changing `ctrl_flag:=2`:

```bash
docker run --rm -it --net=host \
  -v "$PWD/docker_results:/root/im_nmpo_ws/results" \
  im-nmpo:noetic \
  roslaunch im_nmpo robust_im_nmpo.launch \
    ctrl_flag:=2 command_mode:=rate \
    disturbance_case:=hf_periodic \
    disturbance_application:=input disturbance_time_base:=relative \
    use_rviz:=false plot_trajectory:=true \
    plot_output_dir:=/root/im_nmpo_ws/results
```

The saved trajectory plots appear in `docker_results`.

### Build The Image

From the repository root:

```bash
docker build -t im-nmpo:noetic -f docker/Dockerfile .
```

### Export The Image

After building the image, create a compressed archive and checksum:

```bash
./docker/export_image.sh im-nmpo:noetic docker_release
```

This creates:

```text
docker_release/im-nmpo_noetic_<date>.tar.gz
docker_release/im-nmpo_noetic_<date>.tar.gz.sha256
```

## Launch Parameters

Commonly used arguments in `robust_im_nmpo.launch`:

- `plot_trajectory:=true`: save a trajectory tracking plot at shutdown.
- `plot_output_dir:=<path>`: choose where trajectory plots are saved.
- `use_rviz:=true`: start RViz visualization.
- `disturbance_scale:=<value>`: scale the simulated disturbance magnitude.
- `disturbance_seed:=<int>`: set the random seed for stochastic disturbances.
- `imc_wsin_x`, `imc_wsin_y`, `imc_wsin_z`: set the internal-model frequencies
  for the three rotational axes.

The disturbance source is implemented in `px4_bridge/script/q_sim.py`, and the
controller logic is implemented in `IM-NMPO/script/track.py` and
`IM-NMPO/script/robust_agile_fly/internal_model.py`.

## Example Tracking Results

The following examples illustrate the visual comparison produced by the
simulation.

For periodic torque disturbances, `px4_bridge/script/q_sim.py` injects matched
input disturbances into the rotational channels:

```python
u[1] += a1 * math.sin(b1 * t + c1)
u[2] += a2 * math.sin(b2 * t + c2)
u[3] += a3 * math.sin(b3 * t + c3)
```

Trajectory tracking subject to periodic external disturbances:

<p align="center">
  <img src="IM-NMPO/fig/tracking_peridicdisturbance.png" width="90%" alt="The proposed IM-NMPO (Left) VS. the NMPC Baseline (Right)">
  <br>
  <em>Figure 2: The proposed IM-NMPO (left) and the NMPC baseline (right).</em>
</p>
 
For constant torque disturbances, the same simulator applies:

```python
u[1] += 1
u[2] += 1
u[3] += 1
```

Trajectory tracking subject to constant external disturbances:

<p align="center">
  <img src="IM-NMPO/fig/tracking_constantdisturbance.png" width="90%" alt="The proposed IM-NMPO (Left) VS. the NMPC Baseline (Right)">
  <br>
  <em>Figure 3: The proposed IM-NMPO (left) and the NMPC baseline (right).</em>
</p>
