#include "Px4Bridge.hpp"

#include<iostream>
#include<string>

#include<ros/ros.h>
#include<sensor_msgs/Imu.h>
#include<nav_msgs/Odometry.h>
#include<geometry_msgs/Point.h>
#include"px4_bridge/ThrustRates.h"
#include<tf/transform_broadcaster.h>

Px4Bridge quad;

ros::Publisher _imu_pub;
ros::Publisher _state_pub;
ros::Publisher _dynamic_gate_pub;
ros::Subscriber _slam_odom_sub;
ros::Subscriber _thrust_rates_sub;
tf::TransformBroadcaster* tf_br;

void rcv_imu_callback(float w[3], float a[3])
{
    sensor_msgs::Imu s_imu;
    s_imu.header.stamp = ros::Time::now();
    s_imu.angular_velocity.x = w[0];
    s_imu.angular_velocity.y = w[1];
    s_imu.angular_velocity.z = w[2];
    s_imu.linear_acceleration.x = a[0];
    s_imu.linear_acceleration.y = a[1];
    s_imu.linear_acceleration.z = a[2];

    quad._angular_rate[0] = w[0];
    quad._angular_rate[1] = w[1];
    quad._angular_rate[2] = w[2];
    _imu_pub.publish(s_imu);
}

void rcv_state_callback(float state[13])
{
    nav_msgs::Odometry odom;
    odom.header.stamp = ros::Time::now();
    odom.pose.pose.position.x = state[0];
    odom.pose.pose.position.y = state[1];
    odom.pose.pose.position.z = state[2];
    odom.twist.twist.linear.x = state[3];
    odom.twist.twist.linear.y = state[4];
    odom.twist.twist.linear.z = state[5];
    odom.pose.pose.orientation.w = state[6];
    odom.pose.pose.orientation.x = state[7];
    odom.pose.pose.orientation.y = state[8];
    odom.pose.pose.orientation.z = state[9];
    odom.twist.twist.angular.x = state[10];
    odom.twist.twist.angular.y = state[11];
    odom.twist.twist.angular.z = state[12];
    _state_pub.publish(odom);

    tf::Transform transform;
    transform.setOrigin( tf::Vector3(state[1], state[0], -state[2]) );
    transform.setRotation( tf::Quaternion( state[8], state[7], -state[9], state[6] ));
    tf_br->sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "quad_body"));
}

void rcv_dynamic_gate_callback(float pos[3], float quat[4])
{
    geometry_msgs::Point gate;
    gate.x = pos[0];
    gate.y = pos[1];
    gate.z = pos[2];
    _dynamic_gate_pub.publish(gate);
}

void rcv_slam_odom_cb(nav_msgs::Odometry msg)
{
    quad.send_odom_data(msg.pose.pose.position.x, -msg.pose.pose.position.y, -msg.pose.pose.position.z,
                        -msg.pose.pose.orientation.x, msg.pose.pose.orientation.w, -msg.pose.pose.orientation.z, msg.pose.pose.orientation.y);
}

void rcv_thrust_rates_cb(px4_bridge::ThrustRates msg)
{
    // quad.send_control_u(-msg.az/20, msg.wx, msg.wy, msg.wz);
    quad.send_control_u(msg.thrust, msg.wx, msg.wy, msg.wz);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "px4");
    ros::NodeHandle n("~");

    _imu_pub = n.advertise<sensor_msgs::Imu>("imu", 1);
    _state_pub = n.advertise<nav_msgs::Odometry>("odom", 1);
    _dynamic_gate_pub = n.advertise<geometry_msgs::Point>("/dynamic_gate", 1);
    _slam_odom_sub = n.subscribe("slam_odom", 1, rcv_slam_odom_cb, ros::TransportHints().tcpNoDelay());
    _thrust_rates_sub = n.subscribe("thrust_rates", 1, rcv_thrust_rates_cb, ros::TransportHints().tcpNoDelay());
    // _slam_odom_sub = n.subscribe("slam_odom", 1, rcv_slam_odom_cb);
    // _thrust_rates_sub = n.subscribe("thrust_rates", 1, rcv_thrust_rates_cb);
    tf_br = new tf::TransformBroadcaster();

    quad.registe_rcv_state_callback(rcv_state_callback);
    quad.registe_rcv_sensor_imu_callback(rcv_imu_callback);
    quad.registe_rcv_dynamic_object_callback(rcv_dynamic_gate_callback);

    quad.set_thread_rt(90);
    if(quad.setup_port("/dev/ttyACM0") == -1)
    {
        exit(-1);
    }

    quad.setup_optitrack("192.168.1.200");
    quad.add_fordwarding("drone0", 8976, "192.168.1.40", 14550);
    // quad.add_fordwarding("127.0.0.1", 8976, "127.0.0.1", 14550);
    quad.core_start();

    ros::spin();
    // ros::Rate loop_rate(100);
    // while(ros::ok())
    // {
    //     ros::spinOnce();
    //     loop_rate.sleep();
    // }

    return 0;
}
