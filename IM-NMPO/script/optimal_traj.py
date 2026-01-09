#!/usr/bin/env python3
"""

This node reads trajectory data and publishes it as ROS messages
for visualization and control. It supports both trajectory tracking (TrackTraj)
and path visualization (Path) message formats.

"""

import rospy
import csv
import os
import sys
from typing import List, Optional

# ROS message imports
from im_nmpo.msg import TrackTraj
from geometry_msgs.msg import Point, Vector3, Quaternion, PoseStamped
from nav_msgs.msg import Path

# Global configuration
NODE_NAME = "Ref_traj"
TRAJ_TOPIC = "~track_traj"
PATH_TOPIC = "Timeoptimal_path"
DEFAULT_CSV_FILE = "time_optimal_traj.csv"

# Calculate base path for file operations
BASEPATH = os.path.abspath(__file__).split('script', 1)[0] + 'script/'
sys.path.append(BASEPATH)

class TrajectoryPublisher:
    
    def __init__(self):
        """Initialize the trajectory publisher node."""
        rospy.init_node(NODE_NAME, anonymous=True)
        
        # Setup publishers with queue size and TCP_NODELAY for real-time performance
        self.traj_pub = rospy.Publisher(
            TRAJ_TOPIC, 
            TrackTraj, 
            tcp_nodelay=True, 
            queue_size=1
        )
        
        self.path_pub = rospy.Publisher(
            PATH_TOPIC,
            Path,
            queue_size=10
        )
        
        # Subscribe to trigger message
        rospy.Subscriber(
            "~gates",
            TrackTraj,
            self.trajectory_callback,
            queue_size=1
        )
        
        rospy.loginfo(f"{NODE_NAME} initialized successfully")
        rospy.loginfo(f"Waiting for trigger on topic '~gates'...")
        
    def read_trajectory(self, csv_path: str) -> Optional[List[List[float]]]:
        
        try:
            trajectory_data = []
            
            with open(csv_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                
                # Skip header row
                next(csv_reader, None)
                
                # Read all data rows
                for row in csv_reader:
                    if not row:  # Skip empty rows
                        continue
                    
                    # Convert to floats and ensure we have at least 14 columns
                    if len(row) >= 14:
                        try:
                            # Extract first 14 columns (t + 13 state variables)
                            point_data = [float(value) for value in row[:14]]
                            trajectory_data.append(point_data)
                        except ValueError as e:
                            rospy.logwarn(f"Skipping invalid row in CSV: {row}. Error: {e}")
            
            if not trajectory_data:
                rospy.logwarn(f"No valid trajectory data found in {csv_path}")
                return None
            
            rospy.loginfo(f"Successfully loaded {len(trajectory_data)} trajectory points from {csv_path}")
            return trajectory_data
            
        except Exception as e:
            rospy.logerr(f"Error reading CSV file {csv_path}: {e}")
            return None
    
    def publish_trajectory(self, trajectory_data: List[List[float]]) -> bool:
        """
        Publish trajectory data as TrackTraj message.
        
        Args:
            trajectory_data: List of trajectory points
            
        Returns:
            True if successful, False otherwise
        """
        try:
            traj_msg = TrackTraj()
            
            # Process first point
            first_point = trajectory_data[0]
            self._add_trajectory_point(traj_msg, first_point[1:])  # Skip timestamp
            
            # Process remaining points
            for i in range(1, len(trajectory_data)):
                current_point = trajectory_data[i]
                prev_point = trajectory_data[i-1]
                
                # Add point to trajectory
                self._add_trajectory_point(traj_msg, current_point[1:])  # Skip timestamp
                
                # Calculate time difference for dt
                dt = current_point[0] - prev_point[0]
                traj_msg.dt.append(dt)
            
            # Publish trajectory
            self.traj_pub.publish(traj_msg)
            rospy.loginfo(f"Published trajectory with {len(traj_msg.position)} points")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error publishing trajectory: {e}")
            return False
    
    def publish_path_visualization(self, trajectory_data: List[List[float]]) -> bool:
        """
        Publish trajectory as Path message for RViz visualization.
        
        Args:
            trajectory_data: List of trajectory points
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path_msg = Path()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = "world"
            
            for point in trajectory_data:
                # Extract position (columns 1-3 in CSV)
                p_x, p_y, p_z = point[1], point[2], point[3]
                
                # Create pose stamped message
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "world"
                pose_stamped.header.stamp = rospy.Time.now()
                
                # Apply coordinate transformation (matching original implementation)
                # Original: x[i*opt._X_dim+1], y=x[i*opt._X_dim+0], z=-x[i*opt._X_dim+2]
                pose_stamped.pose.position.x = p_y      # Swap x and y
                pose_stamped.pose.position.y = p_x      # Swap x and y
                pose_stamped.pose.position.z = -p_z     # Invert z-axis
                
                # Set orientation to identity quaternion
                pose_stamped.pose.orientation.w = 1
                pose_stamped.pose.orientation.x = 0
                pose_stamped.pose.orientation.y = 0
                pose_stamped.pose.orientation.z = 0
                
                path_msg.poses.append(pose_stamped)
            
            # Publish path
            self.path_pub.publish(path_msg)
            rospy.loginfo(f"Published visualization path with {len(path_msg.poses)} poses")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error publishing visualization path: {e}")
            return False
    
    def _add_trajectory_point(self, traj_msg: TrackTraj, state_vector: List[float]) -> None:
        """
        Add a single trajectory point to TrackTraj message.
        
        Args:
            traj_msg: TrackTraj message to modify
            state_vector: 13-element state vector [px, py, pz, vx, vy, vz, 
                                                    qw, qx, qy, qz, wx, wy, wz]
        """
        # Create position
        position = Point()
        position.x, position.y, position.z = state_vector[0], state_vector[1], state_vector[2]
        
        # Create velocity
        velocity = Vector3()
        velocity.x, velocity.y, velocity.z = state_vector[3], state_vector[4], state_vector[5]
        
        # Create orientation (quaternion)
        orientation = Quaternion()
        orientation.w, orientation.x, orientation.y, orientation.z = state_vector[6:10]
        
        # Create angular velocity
        angular = Vector3()
        angular.x, angular.y, angular.z = state_vector[10], state_vector[11], state_vector[12]
        
        # Add to message
        traj_msg.position.append(position)
        traj_msg.velocity.append(velocity)
        traj_msg.orientation.append(orientation)
        traj_msg.angular.append(angular)
    
    def trajectory_callback(self, msg: TrackTraj) -> None:
        """
        Callback triggered by incoming message on ~gates topic.
        
        Args:
            msg: Trigger message (unused, but required by ROS subscriber)
        """
        rospy.loginfo("Received trigger message, publishing trajectory from CSV...")
        
        # Construct CSV file path
        csv_path = os.path.join(BASEPATH, DEFAULT_CSV_FILE)
        rospy.loginfo(f"Loading trajectory from: {csv_path}")
        
        # Read trajectory data from CSV
        trajectory_data = self.read_trajectory(csv_path)
        if trajectory_data is None:
            return
        
        # Publish trajectory for control
        if not self.publish_trajectory(trajectory_data):
            rospy.logerr("Failed to publish trajectory")
            return
        
        # Publish path for visualization
        if not self.publish_path_visualization(trajectory_data):
            rospy.logwarn("Failed to publish visualization path (non-critical)")
        
        rospy.loginfo("Trajectory publishing completed successfully")
    
    def run(self):
        """Main execution loop."""
        rospy.spin()

def main():
    """Main entry point for the node."""
    try:
        publisher = TrajectoryPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node shutdown requested")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()