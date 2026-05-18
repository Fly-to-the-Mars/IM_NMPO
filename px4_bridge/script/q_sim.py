#! /usr/bin/python3.8

import numpy as np
import casadi as ca
import time,math
import ast

# ROS
import rospy
import tf
from std_msgs.msg import Float32
from px4_bridge.msg import ThrustRates
from nav_msgs.msg import Odometry

from quadrotor import QuadrotorModel, QuadrotorSim

# 
import os, sys
BASEPATH = os.path.abspath(__file__).split('script', 1)[0]
# sys.path += [BASEPATH]

class LP_delay():
    def __init__(self, wc, ts):
        self._alpha = 1/(wc*ts+1)
        self._yk_ = 0
    
    def update(self, xk):
        yk = self._alpha*self._yk_ + (1-self._alpha)*xk
        self._yk_ = yk
        return yk

#######################################################################################################
rospy.init_node("q_sim")
rospy.loginfo("ROS: Hello")

def parse_vector_param(param_name, default_value):
    value = rospy.get_param(param_name, default_value)
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return np.array(value, dtype=float).flatten()
    except Exception as e:
        rospy.logwarn("Failed to parse %s=%s: %s, using %s", param_name, value, e, default_value)
        return np.array(default_value, dtype=float).flatten()

thrust_rates_msg = ThrustRates()
has_control_command = False
def thrust_rates_cb(msg:ThrustRates):
    global thrust_rates_msg, has_control_command
    thrust_rates_msg = msg
    has_control_command = True
    pass

thrust_rates_sub = rospy.Subscriber("~thrust_rates", ThrustRates, callback=thrust_rates_cb, queue_size=1, tcp_nodelay=True)
# wind_pub = rospy.Publisher('wind_speed', Float32, queue_size=1)
command_mode = rospy.get_param("~command_mode", "torque")
disturbance_case = rospy.get_param("~disturbance_case", "hf_periodic")
disturbance_application = rospy.get_param("~disturbance_application", "input")
disturbance_time_base = rospy.get_param("~disturbance_time_base", "absolute")
disturbance_scale = float(rospy.get_param("~disturbance_scale", 1.0))
disturbance_constant = parse_vector_param("~disturbance_constant", [1.0, 1.0, 1.0])
disturbance_amplitude = parse_vector_param("~disturbance_amplitude", [1.0, 2.0, 0.6])
disturbance_frequency = parse_vector_param("~disturbance_frequency", [15.0, 15.0, 16.0])
disturbance_phase = parse_vector_param("~disturbance_phase", [0.1, 0.2, 0.3])
disturbance_seed = int(rospy.get_param("~disturbance_seed", 7))
ou_theta = parse_vector_param("~ou_theta", [1.0, 1.2, 1.4])
ou_sigma = parse_vector_param("~ou_sigma", [1.0, 1.1, 0.6])
ou_mu = parse_vector_param("~ou_mu", [0.0, 0.0, 0.0])
pink_amplitude = parse_vector_param("~pink_amplitude", [1.0, 2.0, 0.6])
pink_sources = int(rospy.get_param("~pink_sources", 10))
pink_theta_min = float(rospy.get_param("~pink_theta_min", 0.15))
pink_theta_max = float(rospy.get_param("~pink_theta_max", 18.0))
disturbance_clip = parse_vector_param("~disturbance_clip", [2.0, 3.0, 1.2])
sim_start_time = rospy.Time.now().to_sec()
rng = np.random.default_rng(disturbance_seed)
ou_state = np.zeros(3)
pink_state = np.zeros((3, pink_sources))
pink_theta = np.geomspace(pink_theta_min, pink_theta_max, pink_sources)
last_noise_time = None
rospy.loginfo("Command mode: %s", command_mode)
rospy.loginfo("Disturbance case: %s", disturbance_case)
rospy.loginfo("Disturbance application: %s", disturbance_application)
rospy.loginfo("Disturbance time base: %s", disturbance_time_base)
rospy.loginfo("Disturbance scale: %.3f", disturbance_scale)
rospy.loginfo("Disturbance seed: %d", disturbance_seed)
rospy.loginfo("Disturbance constant: %s", disturbance_constant.tolist())
rospy.loginfo("Disturbance amplitude: %s", disturbance_amplitude.tolist())
rospy.loginfo("Disturbance frequency: %s", disturbance_frequency.tolist())
rospy.loginfo("Disturbance phase: %s", disturbance_phase.tolist())

quad = QuadrotorModel(BASEPATH+"config/quad_real.yaml")
thrust_rates_msg.thrust = quad._m * quad._G / (4.0 * quad._T_max)
q_sim = QuadrotorSim(quad)
q_sim.set_pos(np.array([-4,-3,-2]))
tf_br = tf.TransformBroadcaster(queue_size=1)
odom_pub = rospy.Publisher("~odom", Odometry, tcp_nodelay=True, queue_size=1)

delay = LP_delay(30, 0.01)
delay_wx = LP_delay(20, 0.01)
delay_wy = LP_delay(20, 0.01)
delay_wz = LP_delay(20, 0.01)

cnt = 0

def _noise_dt(t):
    global last_noise_time
    if last_noise_time is None:
        dt = 0.01
    else:
        dt = max(0.001, min(0.05, t - last_noise_time))
    last_noise_time = t
    return dt

def _clip_disturbance(d):
    return np.clip(d, -disturbance_clip, disturbance_clip)

def get_disturbance(t):
    global ou_state, pink_state
    case = disturbance_case.strip().lower()
    if case in ("none", "zero", "off", "no_disturbance"):
        return np.zeros(3)

    if case in ("constant", "const"):
        return disturbance_scale * disturbance_constant

    if case in ("ou", "ornstein_uhlenbeck", "ornstein-uhlenbeck"):
        dt = _noise_dt(t)
        ou_state += (
            ou_theta * (ou_mu - ou_state) * dt
            + ou_sigma * np.sqrt(dt) * rng.standard_normal(3)
        )
        return disturbance_scale * _clip_disturbance(ou_state)

    if case in ("pink", "pink_noise", "pink-noise", "one_over_f"):
        dt = _noise_dt(t)
        sqrt_dt = np.sqrt(dt)
        for k, theta in enumerate(pink_theta):
            pink_state[:, k] += (
                -theta * pink_state[:, k] * dt
                + np.sqrt(2.0 * theta) * sqrt_dt * rng.standard_normal(3)
            )
        pink = np.sum(pink_state / np.sqrt(pink_theta)[None, :], axis=1)
        pink = pink / np.sqrt(pink_sources)
        return disturbance_scale * _clip_disturbance(pink_amplitude * pink)

    if case in ("lf_periodic", "low_frequency", "low"):
        frequency = np.array([0.1, 0.2, 0.2])
    elif case in ("hf_periodic", "high_frequency", "high"):
        frequency = np.array([15.0, 15.0, 16.0])
    elif case in ("periodic", "custom_periodic", "sin", "sine"):
        frequency = disturbance_frequency
    else:
        rospy.logwarn_throttle(
            2.0,
            "Unknown disturbance_case=%s, using custom periodic disturbance",
            disturbance_case
        )
        frequency = disturbance_frequency

    return disturbance_scale * disturbance_amplitude * np.sin(frequency * t + disturbance_phase)

def sim_run(e):
    # rospy.loginfo("run")
    global thrust_rates_msg, cnt
    now = rospy.Time.now().to_sec()
    if disturbance_time_base.strip().lower() in ("absolute", "ros", "wall"):
        t = now
    else:
        t = now - sim_start_time

    if not has_control_command:
        disturbance = np.zeros(3)
    else:
        disturbance = get_disturbance(t)


    if thrust_rates_msg != None:

        u = np.zeros(4)
        u[0] = delay.update((thrust_rates_msg.thrust)*quad._T_max)  
        if command_mode == "torque":
            u[1] = thrust_rates_msg.wx
            u[2] = thrust_rates_msg.wy
            u[3] = thrust_rates_msg.wz
        else:
            u[1] = delay_wx.update(thrust_rates_msg.wx)
            u[2] = delay_wy.update(thrust_rates_msg.wy)
            u[3] = delay_wz.update(thrust_rates_msg.wz)

        application = disturbance_application.strip().lower()
        external_tau = np.zeros(3)
        if application in ("input", "command", "command_input", "matched"):
            # Matched input disturbance. In torque mode this is tau + d(t),
            # which matches the paper's rotational disturbance model. In rate
            # mode it preserves the legacy NMPC baseline where the disturbance
            # perturbs the angular-rate command channel before the low-level PID.
            u[1:4] += disturbance
        elif application in ("external", "external_torque", "plant"):
            # Physical torque applied directly to the rigid-body dynamics.
            # Useful for stress tests, but it is not the legacy NMPC baseline.
            external_tau = disturbance
        else:
            rospy.logwarn_throttle(
                2.0,
                "Unknown disturbance_application=%s, using input disturbance",
                disturbance_application
            )
            u[1:4] += disturbance
        
        q_sim.step10ms(u, command_mode, external_tau)

        q_state = q_sim.get_state()
        odom_msg = Odometry()
        odom_msg.header.frame_id="world"
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.pose.pose.position.x = q_state[0]
        odom_msg.pose.pose.position.y = q_state[1]
        odom_msg.pose.pose.position.z = q_state[2]
        odom_msg.twist.twist.linear.x = q_state[3]
        odom_msg.twist.twist.linear.y = q_state[4]
        odom_msg.twist.twist.linear.z = q_state[5]
        odom_msg.pose.pose.orientation.w = q_state[6]
        odom_msg.pose.pose.orientation.x = q_state[7]
        odom_msg.pose.pose.orientation.y = q_state[8]
        odom_msg.pose.pose.orientation.z = q_state[9]
        odom_msg.twist.twist.angular.x = q_state[10]
        odom_msg.twist.twist.angular.y = q_state[11]
        odom_msg.twist.twist.angular.z = q_state[12]
        odom_pub.publish(odom_msg)

        if cnt%3 == 0:
            tf_br.sendTransform((q_state[1],q_state[0],-q_state[2]), (q_state[8],q_state[7],-q_state[9],q_state[6]), rospy.Time.now(), "quad_body", "world")
    pass
    

timer = rospy.Timer(rospy.Duration(0.01), sim_run, oneshot=False, reset=False)

rospy.spin()
rospy.loginfo("ROS: Goodby")
