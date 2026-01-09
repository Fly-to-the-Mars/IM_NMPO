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
        
        # Parameter configuration
        self.ctrl_flag = rospy.get_param("~ctrl_flag", 2)  # Default to controller 1
        self.publish_rate = rospy.get_param("~publish_rate", 50)  # Hz
        self.tracking_horizon = rospy.get_param("~tracking_horizon", 10)
        
        # Model initialization
        self.quad_1 = QuadrotorModel_nominal(BASEPATH + 'quad/quad_real.yaml')
        self.quad_2 = QuadrotorModel(BASEPATH + 'quad/quad_real.yaml')
        
        # Controller initialization
        self._init_controller()
        
        # Compensator initialization
        self._init_compensators()
        
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
        
        # Register exit handler to plot errors when program is killed
        atexit.register(self._plot_errors_on_exit)
        
        rospy.loginfo("Trajectory tracking node started, using controller type: %d", self.ctrl_flag)
        
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
    
    def _init_compensators(self):
        """Initialize compensators"""
        if self.ctrl_flag == 1:
            # Internal Model Compensator parameters (configurable)
            wsin = np.array(rospy.get_param("~imc_wsin", [0.1, 0.2, 0.2]))
            self.imc_compensator = InnerModelCompensator(wsin, self.quad_1._J)
            
            # Adaptive Feedback Compensator
            self.adaptive_compensator = AdaptiveFeedbackCompensator(self.quad_1._J)
            rospy.loginfo("Compensators initialized")
    
    def _setup_ros_communications(self):
        """Setup ROS communications"""
        # Publisher
        self.ctrl_pub = rospy.Publisher(
            "~thrust_rates", 
            ThrustRates, 
            queue_size=1, 
            tcp_nodelay=True
        )
        
        # Optional debugging service
        # self.service = rospy.Service('~reset_tracker', ResetTracker, self.handle_reset)
        
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
        
        # Record tracking error
        self._record_tracking_error(x0[:3], trjp[0] if len(trjp) > 0 else None)
        
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
        
        # Record tracking error
        self._record_tracking_error(x0[:3], trjp[0] if len(trjp) > 0 else None)
        
        return control_msg
    
    def _record_tracking_error(self, actual_pos: np.ndarray, 
                              target_pos: Optional[np.ndarray]):
        """Record tracking error"""
        if target_pos is not None:
            error = np.linalg.norm(actual_pos - target_pos)
            self.tracking_errors.append(error)
    
    def _log_performance(self):
        """Log performance statistics"""
        if self.computation_times:
            avg_comp_time = np.mean(self.computation_times)
            max_comp_time = np.max(self.computation_times)
            rospy.logdebug("Average computation time: %.4f ms, Max: %.4f ms", 
                          avg_comp_time * 1000, max_comp_time * 1000)
        
        if self.tracking_errors:
            avg_error = np.mean(self.tracking_errors)
            rospy.logdebug("Average tracking error: %.4f m", avg_error)
    
    def _plot_errors_on_exit(self):
        """Plot error curves when the program is killed"""
        if len(self.tracking_errors) > 0:
            try:
                print("\n" + "="*50)
                print("Plotting tracking error statistics...")
                print("="*50)
                
                # Create figure with multiple subplots
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # Plot 1: Tracking error over time
                axes[0, 0].plot(list(self.tracking_errors), 'b-', linewidth=2)
                axes[0, 0].set_title('Tracking Error Over Time')
                axes[0, 0].set_xlabel('Sample Number')
                axes[0, 0].set_ylabel('Error (m)')
                axes[0, 0].grid(True)
                
                # Plot 2: Error histogram
                axes[0, 1].hist(list(self.tracking_errors), bins=50, alpha=0.7, color='blue', edgecolor='black')
                axes[0, 1].set_title('Tracking Error Distribution')
                axes[0, 1].set_xlabel('Error (m)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Computation time
                if len(self.computation_times) > 0:
                    axes[1, 0].plot([t * 1000 for t in self.computation_times], 'r-', linewidth=2)
                    axes[1, 0].set_title('Computation Time Over Time')
                    axes[1, 0].set_xlabel('Sample Number')
                    axes[1, 0].set_ylabel('Time (ms)')
                    axes[1, 0].grid(True)
                
                # Plot 4: Statistics
                axes[1, 1].axis('off')
                
                # Calculate statistics
                errors = list(self.tracking_errors)
                avg_error = np.mean(errors)
                max_error = np.max(errors)
                min_error = np.min(errors)
                std_error = np.std(errors)
                
                stats_text = f'Tracking Error Statistics:\n\n'
                stats_text += f'Average Error: {avg_error:.4f} m\n'
                stats_text += f'Maximum Error: {max_error:.4f} m\n'
                stats_text += f'Minimum Error: {min_error:.4f} m\n'
                stats_text += f'Standard Deviation: {std_error:.4f} m\n'
                stats_text += f'Total Samples: {len(errors)}\n\n'
                
                if len(self.computation_times) > 0:
                    comp_times = list(self.computation_times)
                    avg_comp_time = np.mean(comp_times) * 1000
                    max_comp_time = np.max(comp_times) * 1000
                    stats_text += f'Average Comp Time: {avg_comp_time:.2f} ms\n'
                    stats_text += f'Max Comp Time: {max_comp_time:.2f} ms\n'
                
                axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                               verticalalignment='top', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.suptitle(f'Trajectory Tracking Performance (Controller Type: {self.ctrl_flag})', fontsize=14)
                plt.tight_layout()
                
                # Save plot to file
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"tracking_error_plot_{timestamp}.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Plot saved to: {filename}")
                
                # Show plot
                plt.show()
                
                print("\nSummary Statistics:")
                print(f"  Average tracking error: {avg_error:.4f} m")
                print(f"  Maximum tracking error: {max_error:.4f} m")
                print(f"  Minimum tracking error: {min_error:.4f} m")
                print(f"  Error standard deviation: {std_error:.4f} m")
                print(f"  Total error samples: {len(errors)}")
                
            except Exception as e:
                print(f"Error while plotting: {str(e)}")
        else:
            print("No tracking error data to plot.")
    
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