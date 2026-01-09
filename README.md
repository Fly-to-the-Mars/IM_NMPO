# Robust Optimal Planning and Control for Agile Flight of Quadrotors: An Internal Model-Based Nonlinear Model Predictive Optimization Approach

![Framework](IM-NMPO/fig/Framework_review.png)

## Instructions
1. Install ros noetic
2. Setup the ros workspace: `~/IM_NMPO_ws`
3. Install CasADi python package: `pip3 install casadi`
4. Clone this repository into `~/IM_NMPO_ws/src`
6. Compile: run `catkin_make` in `~/IM_NMPO_ws`
7. Run the example with `roslaunch im_nmpo robust_im_nmpo.launch`

This will run the time-optimal planning and tracking in the simulation.


