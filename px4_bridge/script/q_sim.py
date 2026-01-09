#! /usr/bin/python3.8

import numpy as np
import casadi as ca
import time,math

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

thrust_rates_msg =ThrustRates()
def thrust_rates_cb(msg:ThrustRates):
    global thrust_rates_msg
    thrust_rates_msg = msg
    pass

thrust_rates_sub = rospy.Subscriber("~thrust_rates", ThrustRates, callback=thrust_rates_cb, queue_size=1, tcp_nodelay=True)
# wind_pub = rospy.Publisher('wind_speed', Float32, queue_size=1)

quad = QuadrotorModel(BASEPATH+"config/quad_real.yaml")
q_sim = QuadrotorSim(quad)
q_sim.set_pos(np.array([-4,-3,-2]))
tf_br = tf.TransformBroadcaster(queue_size=1)
odom_pub = rospy.Publisher("~odom", Odometry, tcp_nodelay=True, queue_size=1)

delay = LP_delay(30, 0.01)
delay_wx = LP_delay(20, 0.01)
delay_wy = LP_delay(20, 0.01)
delay_wz = LP_delay(20, 0.01)

cnt = 0

def sim_run(e):
    # rospy.loginfo("run")
    global thrust_rates_msg, cnt
    t = rospy.Time.now().to_sec()

    # 风速分量的正弦函数系数
    a1 = 1.0
    b1 = 0.1
    c1 = 0.1

    a2 = 2.0
    b2 = 0.2
    c2 = 0.2

    a3 = 0.6
    b3 = 0.2
    c3 = 0.3

    # 计算风速分量
    wind_speed_x = a1 * math.sin(b1 * t + c1)
    wind_speed_y = a2 * math.sin(b2 * t + c2)
    wind_speed_z = a3 * math.sin(b3 * t + c3)


    if thrust_rates_msg != None:

        u = np.zeros(4)
        u[0] = delay.update((thrust_rates_msg.thrust)*quad._T_max)  
        u[1] = delay_wx.update(thrust_rates_msg.wx)
        u[2] = delay_wy.update(thrust_rates_msg.wy)
        u[3] = delay_wz.update(thrust_rates_msg.wz)

        u[1] += wind_speed_x  # 在wx上叠加风速
        u[2] += wind_speed_y  # 在wy上叠加风速
        u[3] += wind_speed_z  # 在wz上叠加风速
        q_sim.step10ms(u)

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
