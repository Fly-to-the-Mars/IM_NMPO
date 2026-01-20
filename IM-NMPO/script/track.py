#!/usr/bin/env python3

import numpy as np
import rospy
import sys
import os
import atexit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from collections import deque
import ast  # 用于安全地解析字符串为Python对象

# ROS messages
from im_nmpo.msg import TrackTraj
from px4_bridge.msg import ThrustRates
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

# Import local modules
BASEPATH = os.path.abspath(__file__).split('script', 1)[0] + 'script/robust_agile_fly/'
sys.path.append(BASEPATH)

from quadrotor import QuadrotorModel_nominal, QuadrotorModel
from tracker import TrackerPos_1, TrackerPos_2
from trajectory import Trajectory
from internal_model import InnerModelCompensator, AdaptiveFeedbackCompensator


@dataclass
class QuadrotorState:
    """Quadrotor state data structure"""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: np.ndarray  # [qw, qx, qy, qz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    
    @classmethod
    def from_odometry(cls, msg: Odometry) -> 'QuadrotorState':
        """Create state from Odometry message"""
        return cls(
            position=np.array([msg.pose.pose.position.x, 
                              msg.pose.pose.position.y, 
                              msg.pose.pose.position.z]),
            velocity=np.array([msg.twist.twist.linear.x,
                              msg.twist.twist.linear.y,
                              msg.twist.twist.linear.z]),
            orientation=np.array([msg.pose.pose.orientation.w,
                                 msg.pose.pose.orientation.x,
                                 msg.pose.pose.orientation.y,
                                 msg.pose.pose.orientation.z]),
            angular_velocity=np.array([msg.twist.twist.angular.x,
                                      msg.twist.twist.angular.y,
                                      msg.twist.twist.angular.z])
        )
    
    def to_state_vector(self) -> np.ndarray:
        """Convert to state vector"""
        return np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
            self.angular_velocity
        ])


class TrajectoryTrackerNode:
    """Trajectory tracking node"""
    
    def __init__(self):
        rospy.init_node("trajectory_tracker")
        
        # Parameter configuration with better error handling
        self.ctrl_flag = self._get_param_with_default("~ctrl_flag", 1)  # Default to controller 1
        self.publish_rate = self._get_param_with_default("~publish_rate", 50)  # Hz
        self.tracking_horizon = self._get_param_with_default("~tracking_horizon", 10)
        self.plot_trajectory = self._get_param_with_default("~plot_trajectory", True)
        
        # Get IMC parameters safely
        imc_wsin_str = self._get_param_with_default("~imc_wsin", "[0.1, 0.2, 0.2]")
        imc_wsin = self._safe_parse_list_param(imc_wsin_str, [0.1, 0.2, 0.2])
        
        # Log current configuration
        rospy.loginfo(f"Controller configuration: ctrl_flag={self.ctrl_flag}")
        rospy.loginfo(f"IMC parameters: wsin={imc_wsin}")
        
        # Model initialization
        self.quad_1 = QuadrotorModel_nominal(BASEPATH + 'quad/quad_real.yaml')
        self.quad_2 = QuadrotorModel(BASEPATH + 'quad/quad_real.yaml')
        
        # Controller initialization
        self._init_controller()
        
        # Compensator initialization
        self._init_compensators(imc_wsin)
        
        # State variables
        self.current_state: Optional[QuadrotorState] = None
        self.trajectory = Trajectory()
        self.v_im = np.zeros(6)  # Internal model state
        self.imu_data: Optional[Imu] = None
        self.trajectory_ready = False
        
        # Performance monitoring
        self.tracking_errors = deque(maxlen=1000)
        self.computation_times = deque(maxlen=100)
        self.message_count = 0
        
        # ROS publishers/subscribers
        self._setup_ros_communications()
        
        # Trajectory data storage
        self.r_x = []  # Actual x positions
        self.r_y = []  # Actual y positions
        self.ref_traj_x = []  # Reference trajectory x positions
        self.ref_traj_y = []  # Reference trajectory y positions
        
        # Register cleanup function
        atexit.register(self._plot_trajectory)
        rospy.on_shutdown(self._plot_trajectory)
        
        rospy.loginfo("Trajectory tracking node started, using controller type: %d", self.ctrl_flag)
    
    def _get_param_with_default(self, param_name, default_value):
        """Safely get parameter with default value"""
        try:
            value = rospy.get_param(param_name, default_value)
            rospy.logdebug(f"Parameter {param_name} = {value}")
            return value
        except Exception as e:
            rospy.logwarn(f"Failed to get parameter {param_name}: {e}, using default: {default_value}")
            return default_value
    
    def _safe_parse_list_param(self, param_str, default_value):
        """Safely parse list parameter from string"""
        try:
            # Remove any extra whitespace
            param_str = param_str.strip()
            # Use ast.literal_eval for safe evaluation
            parsed_value = ast.literal_eval(param_str)
            # Ensure it's a list and convert to numpy array
            if isinstance(parsed_value, (list, tuple)):
                return np.array(parsed_value, dtype=float)
            else:
                rospy.logwarn(f"Parameter is not a list/tuple: {param_str}, using default")
                return np.array(default_value, dtype=float)
        except Exception as e:
            rospy.logwarn(f"Failed to parse parameter: {param_str}, error: {e}, using default")
            return np.array(default_value, dtype=float)
    
    def _init_controller(self):
        """Initialize controller"""
        if self.ctrl_flag == 1:
            self.tracker = TrackerPos_1(self.quad_1)
            try:
                self.tracker.load_so(BASEPATH + "generated/tracker_pos_1.so")
                rospy.loginfo("SO file loaded successfully")
            except Exception as e:
                rospy.logwarn("Failed to load SO file: %s, will use Python implementation", str(e))
        elif self.ctrl_flag == 2:
            self.tracker = TrackerPos_2(self.quad_2)
            self.tracker.define_opt()
        else:
            rospy.logerr("Unknown controller flag: %d, using default controller 1", self.ctrl_flag)
            self.ctrl_flag = 1
            self.tracker = TrackerPos_1(self.quad_1)
    
    def _init_compensators(self, imc_wsin):
        """Initialize compensators with given IMC parameters"""
        if self.ctrl_flag == 1:
            # Internal Model Compensator with configurable parameters
            self.imc_compensator = InnerModelCompensator(imc_wsin, self.quad_1._J)
            
            # Adaptive Feedback Compensator
            self.adaptive_compensator = AdaptiveFeedbackCompensator(self.quad_1._J)
            rospy.loginfo(f"Compensators initialized with wsin={imc_wsin}")
    
    def _setup_ros_communications(self):
        """Setup ROS communications"""
        # Publisher
        self.ctrl_pub = rospy.Publisher(
            "~thrust_rates", 
            ThrustRates, 
            queue_size=1, 
            tcp_nodelay=True
        )
        
        # Subscribers
        rospy.Subscriber(
            "~odom", 
            Odometry, 
            self._odom_callback,
            queue_size=1, 
            tcp_nodelay=True
        )
        
        rospy.Subscriber(
            "~imu", 
            Imu, 
            self._imu_callback,
            queue_size=1, 
            tcp_nodelay=True
        )
        
        rospy.Subscriber(
            "~track_traj", 
            TrackTraj, 
            self._trajectory_callback,
            queue_size=1
        )
        
    def _odom_callback(self, msg: Odometry):
        """Odometry callback"""
        start_time = rospy.Time.now()
        
        try:
            self.current_state = QuadrotorState.from_odometry(msg)
            self.message_count += 1
            
            # Store actual trajectory points
            self.r_x.append(msg.pose.pose.position.x)
            self.r_y.append(msg.pose.pose.position.y)
            
            # Only perform tracking when trajectory is ready
            if self.trajectory_ready and self.current_state is not None:
                control_msg = self._compute_control()
                if control_msg:
                    self.ctrl_pub.publish(control_msg)
                    
                    # Monitor performance
                    comp_time = (rospy.Time.now() - start_time).to_sec()
                    self.computation_times.append(comp_time)
                    
                    if self.message_count % 100 == 0:
                        self._log_performance()
                        
        except Exception as e:
            rospy.logerr("Error processing odometry data: %s", str(e))
    
    def _imu_callback(self, msg: Imu):
        """IMU callback"""
        self.imu_data = msg
    
    def _trajectory_callback(self, msg: TrackTraj):
        """Trajectory callback"""
        try:
            self._load_trajectory(msg)
            self.trajectory_ready = True
            rospy.loginfo("New trajectory loaded, total %d trajectory points", len(msg.dt))
            
            # Store reference trajectory for plotting
            self._store_reference_trajectory()
        except Exception as e:
            rospy.logerr("Failed to load trajectory: %s", str(e))
    
    def _load_trajectory(self, msg: TrackTraj):
        """Load trajectory from message"""
        pos_list, vel_list, quat_list, angular_list, dt_list = [], [], [], [], []
        
        n_points = len(msg.dt)
        for i in range(n_points):
            pos_list.append([msg.position[i].x, msg.position[i].y, msg.position[i].z])
            vel_list.append([msg.velocity[i].x, msg.velocity[i].y, msg.velocity[i].z])
            quat_list.append([msg.orientation[i].w, msg.orientation[i].x, 
                            msg.orientation[i].y, msg.orientation[i].z])
            angular_list.append([msg.angular[i].x, msg.angular[i].y, msg.angular[i].z])
            dt_list.append(msg.dt[i])
        
        # Add the last point
        pos_list.append([msg.position[-1].x, msg.position[-1].y, msg.position[-1].z])
        vel_list.append([msg.velocity[-1].x, msg.velocity[-1].y, msg.velocity[-1].z])
        quat_list.append([msg.orientation[-1].w, msg.orientation[-1].x,
                         msg.orientation[-1].y, msg.orientation[-1].z])
        angular_list.append([msg.angular[-1].x, msg.angular[-1].y, msg.angular[-1].z])
        
        self.trajectory.load_data(
            np.array(pos_list),
            np.array(vel_list),
            np.array(quat_list),
            np.array(angular_list),
            np.array(dt_list)
        )
    
    def _store_reference_trajectory(self):
        """Store reference trajectory for plotting"""
        if hasattr(self.trajectory, '_pos') and self.trajectory._pos is not None:
            self.ref_traj_x = self.trajectory._pos[:, 0].tolist()
            self.ref_traj_y = self.trajectory._pos[:, 1].tolist()
            rospy.loginfo("Stored reference trajectory with %d points", len(self.ref_traj_x))
    
    def _compute_control(self) -> Optional[ThrustRates]:
        """Compute control command"""
        try:
            x0 = self.current_state.to_state_vector()
            
            if self.ctrl_flag == 1:
                return self._compute_control_type1(x0)
            else:
                return self._compute_control_type2(x0)
                
        except Exception as e:
            rospy.logwarn("Error computing control command: %s", str(e))
            return None
    
    def _compute_control_type1(self, x0: np.ndarray) -> ThrustRates:
        """Type 1 controller computation"""
        # Extend state vector
        x1 = np.concatenate([x0, self.v_im])
        
        # Sample trajectory
        trjp, trjv, trjdt, ploy = self.trajectory.sample(
            self.tracker._trj_N, 
            x1[:3]
        )
        
        # Solve optimization problem
        res = self.tracker.solve(x1, ploy.reshape(-1), trjdt)
        x = res['x'].full().flatten()
        
        # Update internal model state
        self.v_im = x[13:19]
        
        # Extract control command
        control_start_idx = self.tracker._Herizon * 19
        Tt = 1 * (x[control_start_idx + 0] + 
                  x[control_start_idx + 1] + 
                  x[control_start_idx + 2] + 
                  x[control_start_idx + 3])
        
        # Nominal attitude and angular velocity
        q_nominal = np.array([float(x[6]), float(x[7]), 
                             float(x[8]), float(x[9])])
        w_nominal = np.array([float(x[10]), float(x[11]), float(x[12])])
        tau_nominal = np.array([x[10], x[11], x[12]])
        
        # Actual state
        q_actual = self.current_state.orientation
        w_actual = self.current_state.angular_velocity
        
        # Apply compensation
        imc_compensation = self.imc_compensator.get_compensation(self.v_im)
        adaptive_compensation = self.adaptive_compensator.compute_delta_tau(
            q_actual, w_actual, q_nominal, w_nominal
        )
        
        # Final torque = nominal torque + internal model compensation + adaptive compensation
        tau_final = tau_nominal + imc_compensation.flatten() + adaptive_compensation
        
        # Create control message
        control_msg = ThrustRates()
        control_msg.thrust = Tt / (4 * self.quad_1._T_max)
        control_msg.wx = float(tau_final[0])
        control_msg.wy = float(tau_final[1])
        control_msg.wz = float(tau_final[2])
        
        return control_msg
    
    def _compute_control_type2(self, x0: np.ndarray) -> ThrustRates:
        """Type 2 controller computation"""
        # Sample trajectory
        trjp, trjv, trjdt, ploy = self.trajectory.sample(
            self.tracker._trj_N, 
            x0[:3]
        )
        
        # Solve optimization problem
        res = self.tracker.solve(x0, ploy.reshape(-1), trjdt)
        x = res['x'].full().flatten()
        
        # Extract control command
        Tt = 1 * (x[self.tracker._Herizon * 13 + 0] +
                  x[self.tracker._Herizon * 13 + 1] +
                  x[self.tracker._Herizon * 13 + 2] +
                  x[self.tracker._Herizon * 13 + 3])
        
        # Create control message
        control_msg = ThrustRates()
        control_msg.thrust = Tt / (4 * self.quad_2._T_max)
        control_msg.wx = float(x[10])
        control_msg.wy = float(x[11])
        control_msg.wz = float(x[12])
        
        return control_msg
    
    def _log_performance(self):
        """Log performance metrics"""
        if len(self.computation_times) > 0:
            avg_time = np.mean(self.computation_times)
            max_time = np.max(self.computation_times)
            rospy.loginfo("Computation time - Avg: %.4fs, Max: %.4fs", avg_time, max_time)
    
    def _plot_trajectory(self):
        """Plot and save trajectory"""
        if not self.plot_trajectory:
            return
            
        try:
            if len(self.r_x) > 0 and len(self.ref_traj_x) > 0:
                # Create figure
                plt.figure(figsize=(10, 8))
                
                # Plot reference trajectory
                plt.plot(self.ref_traj_x, self.ref_traj_y, color='#1f77b4', linestyle='-', linewidth=2.5, 
                        label='Reference Trajectory', alpha=1)
                
                # Plot actual trajectory
                plt.plot(self.r_x, self.r_y, color='#ff7f0e', linestyle='-', linewidth=2, 
                        label='Actual Trajectory', alpha=0.9)
                
                # Add start and end markers
                if len(self.r_x) > 0:
                    plt.scatter(self.r_x[0], self.r_y[0], c='blue', s=100, 
                              marker='o', label='Start', zorder=5)
                    plt.scatter(self.r_x[-1], self.r_y[-1], c='orange', s=100, 
                              marker='s', label='End', zorder=5)
                
                # Plot settings
                plt.xlabel('X Position (m)', fontsize=12)
                plt.ylabel('Y Position (m)', fontsize=12)
                plt.title('Trajectory Tracking Performance', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                plt.axis('equal')
                
                # Save figure
                timestamp = rospy.Time.now().to_sec()
                save_path = f'/tmp/trajectory_plot_{timestamp:.0f}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                rospy.loginfo(f"Trajectory plot saved to: {save_path}")
                
                # Show plot
                plt.show()
                
            elif len(self.r_x) > 0:
                rospy.logwarn("Only actual trajectory available for plotting")
                plt.figure(figsize=(10, 8))
                plt.plot(self.r_x, self.r_y, 'r-', linewidth=1.5, label='Actual Trajectory')
                if len(self.r_x) > 0:
                    plt.scatter(self.r_x[0], self.r_y[0], c='blue', s=100, marker='o', label='Start')
                    plt.scatter(self.r_x[-1], self.r_y[-1], c='orange', s=100, marker='s', label='End')
                plt.xlabel('X Position (m)', fontsize=12)
                plt.ylabel('Y Position (m)', fontsize=12)
                plt.title('Actual Trajectory', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                plt.axis('equal')
                plt.show()
            else:
                rospy.logwarn("No trajectory data to plot")
                
        except Exception as e:
            rospy.logerr(f"Error plotting trajectory: {str(e)}")
    
    def run(self):
        """Run the node"""
        rate = rospy.Rate(self.publish_rate)
        
        while not rospy.is_shutdown():
            # Here you can add periodic tasks like status monitoring, parameter updates, etc.
            rate.sleep()


def main():
    """Main function"""
    try:
        node = TrajectoryTrackerNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted by user")
    except Exception as e:
        rospy.logfatal("Node runtime error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()