# Robust Optimal Planning and Control for Agile Flight of Quadrotors: An Internal Model-Based Nonlinear Model Predictive Optimization Approach

<div align="center">
  <img src="IM-NMPO/fig/Framework_review.png" width="60%" alt="IM-NMPO Framework">
  <br>
  <em>Figure 1: (A) IM-NMPO framework. (B) Simulation and real-world experiments encompassing various disturbances, including unknown payloads, unknown persistent fan-induced and time-varying gusts, across quadrotors with different wheelbases (450mm, 330mm, and 250mm).</em>
</div>

## Instructions
1. Install ros noetic
2. Setup the ros workspace: `~/IM_NMPO_ws`
3. Install CasADi python package: `pip3 install casadi`
4. Clone this repository into `~/IM_NMPO_ws/src`
5. Compile: run `catkin_make` in `~/IM_NMPO_ws`
6. Run the example with `roslaunch im_nmpo robust_im_nmpo.launch`

This will execute the robust agile tracking of IM-NMPO under external periodic or constant disturbances, with performance comparison against the NMPC baseline.

 
