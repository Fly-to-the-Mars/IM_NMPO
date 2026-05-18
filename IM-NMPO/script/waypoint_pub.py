#! /usr/bin/python3.8

import rospy
import numpy as np

from im_nmpo.msg import TrackTraj
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import Header, ColorRGBA

import os, sys
BASEPATH = os.path.abspath(__file__).split('script/', 1)[0]+'script/robust_agile_fly/'
sys.path += [BASEPATH]

from gates.gates import Gates

rospy.init_node("gates_sim")
gates_pub = rospy.Publisher("~gates", TrackTraj, tcp_nodelay=True, queue_size=1)
gates_marker_pub = rospy.Publisher("/plan/gates_marker", Marker, queue_size=1)
ground_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)

def create_blue_background_marker():
    marker = Marker()
    
    marker.header = Header(
        frame_id="world",  
        stamp=rospy.Time.now()
    )
    marker.ns = "background"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose = Pose(
        position=Point(x=0, y=2, z=0),
        orientation=Quaternion(x=0, y=0, z=0, w=1)
    )
    marker.scale = Vector3(x=70, y=70, z=0.001)
    marker.color = ColorRGBA(r=0.73, g=1, b=1, a=0.3)
    marker.lifetime = rospy.Duration(0)
    return marker

gates = Gates(BASEPATH+"gates/gates_n7.yaml")


def timer_cb(event):
    gates_traj = TrackTraj()
    gates_marker = Marker()
    # 设置 Marker 基本属性
    gates_marker.header.frame_id = "world"
    gates_marker.action = Marker.ADD
    gates_marker.type = Marker.LINE_LIST  # 使用 LINE_LIST 类型
    gates_marker.scale.x = 0.12  # 设置线条宽度
    gates_marker.color = ColorRGBA(0, 0, 0, 1)  # 红色并设置为不透明
    # 创建方框的顶点
    half_size = 0.5  # 方框边长的一半
    
    for i in range(gates._N):
        pos = Point()
        pos.x = gates._pos[i][0]
        pos.y = gates._pos[i][1]
        pos.z = gates._pos[i][2]
        gates_traj.position.append(pos)

        pos = Point()

        pos.x = gates._pos[i][1]  # gates._pos[i][1] 对应新的 x
        pos.y = gates._pos[i][0]  # gates._pos[i][0] 对应新的 y
        pos.z = -gates._pos[i][2]  # -gates._pos[i][2] 对应新的 z
        # 生成方框的8个顶点
        if i == 1 or i ==2 or i==6 or i==7: 
            points = [
                # 底面
                Point(pos.x - half_size, pos.y , pos.z- half_size),
                Point(pos.x + half_size, pos.y , pos.z- half_size),
                Point(pos.x + half_size, pos.y , pos.z- half_size),
                Point(pos.x + half_size, pos.y , pos.z+ half_size),
                Point(pos.x + half_size, pos.y , pos.z+ half_size),
                Point(pos.x - half_size, pos.y , pos.z+ half_size),
                Point(pos.x - half_size, pos.y , pos.z+ half_size),
                Point(pos.x - half_size, pos.y , pos.z- half_size)
            ]
            gates_marker.points.extend(points)
        elif i ==3 or i==4 or i==5 or i==0:
            points = [
                # 底面
                Point(pos.x , pos.y- half_size , pos.z- half_size),
                Point(pos.x , pos.y+ half_size , pos.z- half_size),
                Point(pos.x , pos.y+ half_size , pos.z- half_size),
                Point(pos.x , pos.y+ half_size , pos.z+ half_size),
                Point(pos.x , pos.y+ half_size , pos.z+ half_size),
                Point(pos.x , pos.y- half_size , pos.z+ half_size),
                Point(pos.x , pos.y- half_size , pos.z+ half_size),
                Point(pos.x , pos.y- half_size , pos.z- half_size)
            ]   
            # 将点添加到 gates_marker 中
            gates_marker.points.extend(points)

    # 发布轨迹和 Marker
    gates_pub.publish(gates_traj)
    gates_marker_pub.publish(gates_marker)
    ground_marker = create_blue_background_marker()
    ground_pub.publish(ground_marker)


    
rospy.Timer(rospy.Duration(0.1), timer_cb)

rospy.spin()
