# Internal Model-Based Nonlinear Model Predictive Optimization for Robust Agile Quadrotor Flight

<p align="center">
  <img src="IM-NMPO/fig/Framework_review.png" width="60%" alt="IM-NMPO Framework">
</p>

<em>Figure 1: (A) IM-NMPO framework. (B) Simulation and real-world experiments encompassing various disturbances, including unknown payloads, unknown persistent fan-induced and time-varying gusts, across quadrotors with different wheelbases (450mm, 330mm, and 250mm).</em>

## Reproduce With Docker


Docker is the recommended path. ROS Noetic and the required Python packages are
included in the Docker image.

This repository reproduces the simulation comparison between:

- `ctrl_flag:=1`: the proposed IM-NMPO controller.
- `ctrl_flag:=2`: the NMPC baseline.

Default reproduced case:

- command interface: angular-rate command, `command_mode:=rate`
- disturbance: high-frequency periodic input disturbance, `disturbance_case:=hf_periodic`
- output: trajectory plots saved in `docker_results`

## 1. Clone The Repository

```bash
git clone https://github.com/Fly-to-the-Mars/IM_NMPO.git
cd IM_NMPO
```

Run all remaining commands from this repository root.

## 2. Prepare The Docker Image

Choose one of the following two options.

### Option A: Use The Release Image Archive

Use this option if the GitHub release provides the prebuilt Docker image archive.
It is faster than building the image locally.

Download the Docker image archive:

```bash
wget -O im-nmpo_noetic.gz \
  https://github.com/Fly-to-the-Mars/IM_NMPO/releases/latest/download/im-nmpo_noetic.gz

wget -O im-nmpo_noetic.gz.sha256 \
  https://github.com/Fly-to-the-Mars/IM_NMPO/releases/latest/download/im-nmpo_noetic.gz.sha256
```

Then run:

```bash
sha256sum -c im-nmpo_noetic.gz.sha256
docker load -i im-nmpo_noetic.gz
docker image ls im-nmpo
```

Expected result:

```text
im-nmpo_noetic.gz: OK
Loaded image: im-nmpo:noetic
```

### Option B: Build The Docker Image Locally

Use this option if no image archive is available, or if you prefer to build from
the Dockerfile.

```bash
docker build -t im-nmpo:noetic -f docker/Dockerfile .
```

This installs ROS Noetic, Python dependencies, and builds the catkin workspace
inside the image. A successful build ends with an image tagged as:

```text
im-nmpo:noetic
```

## 3. Run IM-NMPO

Run the proposed IM-NMPO controller:

```bash
mkdir -p docker_results

timeout --foreground -s INT 45s docker run --rm --net=host \
  -v "$PWD/docker_results:/root/im_nmpo_ws/results" \
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

Run the NMPC baseline under the same disturbance:

```bash
timeout --foreground -s INT 45s docker run --rm --net=host \
  -v "$PWD/docker_results:/root/im_nmpo_ws/results" \
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

After both runs, list the saved plots:

```bash
ls -lh docker_results
```

Expected result:

```text
trajectory_plot_1781011412.png
trajectory_plot_1781011475.png
```

There should be one plot from IM-NMPO and one plot from the NMPC baseline. Your
filenames will have different timestamps.

## 6. Run Other Disturbance Cases

To reproduce a different disturbance, keep the command unchanged and replace
only `disturbance_case:=hf_periodic`.

Available disturbance cases:

```text
none
constant
lf_periodic
hf_periodic
ou
pink_noise
```

For example, to run IM-NMPO under an Ornstein-Uhlenbeck stochastic disturbance
with online internal-model frequency adaptation:

```bash
timeout --foreground -s INT 45s docker run --rm --net=host \
  -v "$PWD/docker_results:/root/im_nmpo_ws/results" \
  im-nmpo:noetic \
  roslaunch im_nmpo robust_im_nmpo.launch \
    ctrl_flag:=1 command_mode:=rate \
    disturbance_case:=ou \
    imc_frequency_adaptation:=true \
    disturbance_application:=input disturbance_time_base:=relative \
    use_rviz:=false plot_trajectory:=true \
    plot_output_dir:=/root/im_nmpo_ws/results \
  || test $? -eq 124
```

## Optional: Build And Run Without Docker

This path is only for users who already have Ubuntu 20.04 and ROS Noetic
installed locally.

Install the Python dependency:

```bash
pip3 install casadi==3.6.5
```

Build in a catkin workspace:

```bash
mkdir -p ~/im_nmpo_ws/src
cd ~/im_nmpo_ws/src
git clone https://github.com/Fly-to-the-Mars/IM_NMPO.git
cd ~/im_nmpo_ws
catkin_make
source devel/setup.bash
```

Run IM-NMPO without RViz:

```bash
roslaunch im_nmpo robust_im_nmpo.launch \
  ctrl_flag:=1 command_mode:=rate \
  disturbance_case:=hf_periodic \
  disturbance_application:=input disturbance_time_base:=relative \
  use_rviz:=false plot_trajectory:=true
```

Wait 20-45 s, then press `Ctrl-C`. The plot is saved to:

```text
IM-NMPO/fig/sim_results/trajectory_plot_1781011412.png
```

Run the NMPC baseline by changing `ctrl_flag:=1` to `ctrl_flag:=2`.

## Important Launch Parameters

- `ctrl_flag:=1`: proposed IM-NMPO controller.
- `ctrl_flag:=2`: NMPC baseline.
- `command_mode:=rate`: angular-rate command interface.
- `disturbance_case:=hf_periodic`: disturbance scenario used in the default
  comparison.
- `plot_trajectory:=true`: save the trajectory plot at shutdown.
- `plot_output_dir:=/root/im_nmpo_ws/results`: output directory for saved plots
  inside Docker.
- `use_rviz:=false`: headless mode, recommended for Docker.
- `imc_frequency_adaptation:=true`: update internal-model frequencies online
  for stochastic disturbances.

The disturbance source is implemented in `px4_bridge/script/q_sim.py`. The
controller logic is implemented in `IM-NMPO/script/track.py` and
`IM-NMPO/script/robust_agile_fly/internal_model.py`.

## Troubleshooting

If Docker says permission denied:

```bash
sudo docker --version
```

If `sudo docker --version` works, either prepend `sudo` to the Docker commands
or configure Docker so your user can run Docker without `sudo`.

If `sha256sum -c ...` fails, make sure both downloaded release files are in the
same directory and that you are running the command from that directory.

If Docker reports that port `11311` is already in use, stop any existing
`roscore` or `roslaunch` process on the host, then rerun the command.

If no plot appears, check that the command includes:

```text
plot_trajectory:=true
plot_output_dir:=/root/im_nmpo_ws/results
-v "$PWD/docker_results:/root/im_nmpo_ws/results"
```

## Maintainer: Export The Docker Image

The Docker image archive is not committed to git. Upload these two files as
GitHub Release assets:

```text
docker_release/im-nmpo_noetic.gz
docker_release/im-nmpo_noetic.gz.sha256
```

To regenerate them after rebuilding `im-nmpo:noetic`, run:

```bash
./docker/export_image.sh im-nmpo:noetic docker_release
mv docker_release/im-nmpo_noetic_*.tar.gz docker_release/im-nmpo_noetic.gz
(cd docker_release && sha256sum im-nmpo_noetic.gz > im-nmpo_noetic.gz.sha256)
rm -f docker_release/im-nmpo_noetic_*.tar.gz.sha256
```

Then upload the two files above to the GitHub Release page.

## Example Tracking Results

The following figures illustrate the visual comparison produced by the
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
