# Robust Optimal Planning and Control for Agile Flight of Quadrotors: An Internal Model-Based Nonlinear Model Predictive Optimization Approach

<p align="center">
  <img src="IM-NMPO/fig/Framework_review.png" width="60%" alt="IM-NMPO Framework">
</p>

<em>Figure 1: (A) IM-NMPO framework. (B) Simulation and real-world experiments encompassing various disturbances, including unknown payloads, unknown persistent fan-induced and time-varying gusts, across quadrotors with different wheelbases (450mm, 330mm, and 250mm).</em>

## Instructions
1. Install ros noetic
2. Setup the ros workspace: `~/IM_NMPO_ws`
3. Install CasADi python package: `pip3 install casadi==3.6.5`
4. Clone this repository into `~/IM_NMPO_ws/src`
5. Compile: run `catkin_make` in `~/IM_NMPO_ws`
6. Run the example with `roslaunch im_nmpo robust_im_nmpo.launch` for the proposed IM-NMPO
7. Run the example with `roslaunch im_nmpo robust_im_nmpo.launch ctrl_flag:=2` for the compared NMPC


This will execute the robust agile tracking of IM-NMPO under external periodic or constant disturbances, with performance comparison against the NMPC baseline.

For the periodic torque disturbances: (IM_NMPO_ws\px4_bridge\script\q_sim.py)

        wind_speed_x = a1 * math.sin(b1 * t + c1)
        wind_speed_y = a2 * math.sin(b2 * t + c2)
        wind_speed_z = a3 * math.sin(b3 * t + c3)

        u[1] += wind_speed_x 
        u[2] += wind_speed_y 
        u[3] += wind_speed_z 
 
For the constant torque disturbances: (IM_NMPO_ws\px4_bridge\script\q_sim.py)

        u[1] += 1 
        u[2] += 1 
        u[3] += 1 

Tested Environments:
        * Ubuntu 20.04 + ROS1 Noetic