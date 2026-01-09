#!/usr/bin/env python3
"""
Time-Optimal Trajectory Planning for Quadrotors
"""

import numpy as np
import casadi as ca
import csv
import sys
import os
from typing import List, Tuple, Optional, Dict, Any


# Add base path to system path
BASEPATH = os.path.abspath(__file__).split("robust_agile_fly/", 1)[0] + "robust_agile_fly/"
sys.path += [BASEPATH]

from quadrotor import QuadrotorModel


def quaternion_multiply(q1: ca.SX, q2: ca.SX) -> ca.SX:
    """Multiply two quaternions: q_result = q1 ⊗ q2"""
    return ca.vertcat(
        q2[0, :] * q1[0, :] - q2[1, :] * q1[1, :] - q2[2, :] * q1[2, :] - q2[3, :] * q1[3, :],
        q2[0, :] * q1[1, :] + q2[1, :] * q1[0, :] - q2[2, :] * q1[3, :] + q2[3, :] * q1[2, :],
        q2[0, :] * q1[2, :] + q2[2, :] * q1[0, :] + q2[1, :] * q1[3, :] - q2[3, :] * q1[1, :],
        q2[0, :] * q1[3, :] - q2[1, :] * q1[2, :] + q2[2, :] * q1[1, :] + q2[3, :] * q1[0, :]
    )


def rotate_vector_by_quaternion(q: ca.SX, v: ca.SX) -> ca.SX:
    """Rotate a 3D vector by a quaternion: v_rotated = q ⊗ [0, v] ⊗ q_conjugate"""
    q_conj = ca.vertcat(q[0, :], -q[1, :], -q[2, :], -q[3, :])
    v_quat = ca.vertcat(0, v)
    
    result = quaternion_multiply(
        quaternion_multiply(q, v_quat),
        q_conj
    )
    
    return ca.vertcat(result[1, :], result[2, :], result[3, :])


class TimeOptimalPlanner:
    """
    Time-optimal trajectory planner for quadrotors through waypoints
    
    Attributes:
        quad: Quadrotor model
        waypoint_num: Number of waypoints
        segment_points: Number of discretization points per segment
        loop_flag: Whether the trajectory is closed-loop
        tolerance: Waypoint tolerance
    """
    
    def __init__(self, quad: QuadrotorModel, waypoint_num: int, 
                 segment_points: List[int], loop_flag: bool, tolerance: float = 0.01):
        self._quad = quad
        self._loop_flag = loop_flag
        self._tolerance = tolerance
        self._waypoint_num = waypoint_num
        self._segment_points = segment_points
        
        # Validate inputs
        if len(segment_points) != waypoint_num:
            raise ValueError(f"Expected {waypoint_num} segment point counts, got {len(segment_points)}")
        
        # Initialize dynamics
        self._dynamics = self._quad.dynamics_discrete_time()
        
        # Calculate total horizon and segment boundaries
        self._total_horizon = sum(segment_points)
        self._segment_boundaries = self._calculate_segment_boundaries(segment_points)
        
        print(f"Total discretization points: {self._total_horizon}")
        
        # Get system dimensions
        self._state_dim = self._dynamics.size1_in(0)
        self._control_dim = self._dynamics.size1_in(1)
        
        # Get bounds
        self._state_bounds = self._quad._X_lb, self._quad._X_ub
        self._control_bounds = self._quad._U_lb, self._quad._U_ub
        
        # Define symbolic variables
        self._define_symbolic_variables()
        
        # Define optimization options
        self._define_optimization_options()
        
        # Build optimization problem
        self._build_optimization_problem()
        
    def _calculate_segment_boundaries(self, segment_points: List[int]) -> List[int]:
        """Calculate cumulative segment boundaries"""
        boundaries = [0]
        for points in segment_points:
            boundaries.append(boundaries[-1] + points)
        return boundaries
    
    def _define_symbolic_variables(self):
        """Define symbolic variables for optimization"""
        # Time variables for each segment
        self._segment_times = ca.SX.sym('segment_times', self._waypoint_num)
        
        # State and control variables
        self._states = ca.SX.sym('states', self._state_dim, self._total_horizon)
        self._controls = ca.SX.sym('controls', self._control_dim, self._total_horizon)
        
        # Waypoint positions
        self._waypoint_positions = ca.SX.sym('waypoint_positions', 3, self._waypoint_num)
        
        # Initial state
        if self._loop_flag:
            self._initial_state = self._states[:, -1]
        else:
            self._initial_state = ca.SX.sym('initial_state', self._state_dim)
    
    def _define_optimization_options(self):
        """Define optimization solver options"""
        self._solver_options = {
            'ipopt.tol': 1e-5,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,
        }
        
        self._time_solver_options = {
            'ipopt.tol': 1e-2,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,
        }
        
        self._warm_start_options = {
            'ipopt.tol': 1e-2,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_frac': 1e-6,
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.warm_start_slack_bound_frac': 1e-6,
            'ipopt.warm_start_slack_bound_push': 1e-6,
            'ipopt.print_level': 0,
        }
    
    def _build_optimization_problem(self):
        """Build the optimization problem constraints and objectives"""
        # Initialize lists for optimization variables and constraints
        self._opt_vars_states = []
        self._opt_vars_controls = []
        self._opt_vars_times = []
        
        self._opt_bounds_states_lb = []
        self._opt_bounds_states_ub = []
        self._opt_bounds_controls_lb = []
        self._opt_bounds_controls_ub = []
        self._opt_bounds_times_lb = []
        self._opt_bounds_times_ub = []
        
        self._constraints_dynamics = []
        self._constraints_waypoints = []
        
        self._constraints_dynamics_lb = []
        self._constraints_dynamics_ub = []
        self._constraints_waypoints_lb = []
        self._constraints_waypoints_ub = []
        
        # Parameters
        self._parameters_initial_state = [] if self._loop_flag else [self._initial_state]
        self._parameters_segment_times = []
        self._parameters_waypoints = []
        
        # Cost function components
        self._cost_angular_velocity = 0
        self._cost_time = 0
        self._cost_waypoint_error = 0
        
        # Weight matrices
        self._weight_angular_velocity = ca.diag([0.01, 0.01, 0.01])
        self._weight_waypoint_error = ca.diag([1, 1, 1])
        
        # Build problem segment by segment
        for segment_idx in range(self._waypoint_num):
            self._add_segment_constraints(segment_idx)
        
        # Store initial guess
        self._initial_guess = self._create_initial_guess()
        
    def _add_segment_constraints(self, segment_idx: int):
        """Add constraints for a single trajectory segment"""
        start_idx = self._segment_boundaries[segment_idx]
        end_idx = self._segment_boundaries[segment_idx + 1]
        
        # Add variables for segment start
        self._opt_vars_states.append(self._states[:, start_idx])
        self._opt_bounds_states_lb.extend(self._state_bounds[0])
        self._opt_bounds_states_ub.extend(self._state_bounds[1])
        
        self._opt_vars_controls.append(self._controls[:, start_idx])
        self._opt_bounds_controls_lb.extend(self._control_bounds[0])
        self._opt_bounds_controls_ub.extend(self._control_bounds[1])
        
        self._opt_vars_times.append(self._segment_times[segment_idx])
        self._opt_bounds_times_lb.append(0)
        self._opt_bounds_times_ub.append(0.5)
        
        # Dynamics constraint for segment start
        if segment_idx == 0 and not self._loop_flag:
            # First point: constraint from initial state
            dynamics_constraint = self._states[:, 0] - self._dynamics(
                self._initial_state, self._controls[:, 0], self._segment_times[0]
            )
        else:
            # Intermediate points: constraint from previous state
            prev_idx = start_idx - 1 if start_idx > 0 else -1
            dynamics_constraint = self._states[:, start_idx] - self._dynamics(
                self._states[:, prev_idx], self._controls[:, start_idx], self._segment_times[segment_idx]
            )
        
        self._constraints_dynamics.append(dynamics_constraint)
        self._constraints_dynamics_lb.extend([-0.0] * self._state_dim)
        self._constraints_dynamics_ub.extend([0.0] * self._state_dim)
        
        # Add cost for angular velocity
        self._cost_angular_velocity += self._states[10:13, start_idx].T @ \
                                      self._weight_angular_velocity @ \
                                      self._states[10:13, start_idx]
        
        # Waypoint constraint (at segment end)
        waypoint_constraint = (self._states[:3, end_idx - 1] - 
                              self._waypoint_positions[:, segment_idx]).T @ \
                             (self._states[:3, end_idx - 1] - self._waypoint_positions[:, segment_idx])
        
        self._constraints_waypoints.append(waypoint_constraint)
        self._constraints_waypoints_lb.append(0)
        self._constraints_waypoints_ub.append(self._tolerance * self._tolerance)
        
        # Add parameters
        self._parameters_segment_times.append(self._segment_times[segment_idx])
        self._parameters_waypoints.append(self._waypoint_positions[:, segment_idx])
        
        # Add cost components
        self._cost_time += self._segment_times[segment_idx] * self._segment_points[segment_idx]
        self._cost_waypoint_error += (self._states[:3, end_idx - 1] - 
                                     self._waypoint_positions[:, segment_idx]).T @ \
                                    self._weight_waypoint_error @ \
                                    (self._states[:3, end_idx - 1] - self._waypoint_positions[:, segment_idx])
        
        # Add intermediate points in segment
        for point_idx in range(1, self._segment_points[segment_idx]):
            current_idx = start_idx + point_idx
            
            self._opt_vars_states.append(self._states[:, current_idx])
            self._opt_bounds_states_lb.extend(self._state_bounds[0])
            self._opt_bounds_states_ub.extend(self._state_bounds[1])
            
            self._opt_vars_controls.append(self._controls[:, current_idx])
            self._opt_bounds_controls_lb.extend(self._control_bounds[0])
            self._opt_bounds_controls_ub.extend(self._control_bounds[1])
            
            # Dynamics constraint for intermediate point
            dynamics_constraint = self._states[:, current_idx] - self._dynamics(
                self._states[:, current_idx - 1], self._controls[:, current_idx], 
                self._segment_times[segment_idx]
            )
            
            self._constraints_dynamics.append(dynamics_constraint)
            self._constraints_dynamics_lb.extend([-0.0] * self._state_dim)
            self._constraints_dynamics_ub.extend([0.0] * self._state_dim)
            
            # Add cost for angular velocity
            self._cost_angular_velocity += self._states[10:13, current_idx].T @ \
                                          self._weight_angular_velocity @ \
                                          self._states[10:13, current_idx]
    
    def _create_initial_guess(self) -> np.ndarray:
        """Create initial guess for optimization variables"""
        guess = np.zeros((self._state_dim + self._control_dim) * self._total_horizon)
        
        for i in range(self._total_horizon):
            guess[i * self._state_dim + 6] = 1.0  # Quaternion w component
            guess[self._total_horizon * self._state_dim + i * self._control_dim] = -9.81  # Initial thrust
        
        return guess
    
    def create_warm_start_solver(self):
        """Create solver for warm-up phase"""
        optimization_problem = {
            'f': self._cost_waypoint_error + self._cost_angular_velocity,
            'x': ca.vertcat(*(self._opt_vars_states + self._opt_vars_controls)),
            'p': ca.vertcat(*(self._parameters_initial_state + 
                             self._parameters_waypoints + 
                             self._parameters_segment_times)),
        }
        
        self._solver = ca.nlpsol('planner', 'ipopt', optimization_problem, self._solver_options)
    
    def solve_warm_start(self, initial_state: np.ndarray, 
                        waypoints: np.ndarray, 
                        segment_times: np.ndarray) -> Dict[str, Any]:
        """Solve warm-up optimization problem"""
        if self._loop_flag:
            parameters = np.concatenate([waypoints.flatten(), segment_times])
        else:
            parameters = np.concatenate([
                initial_state, 
                waypoints.flatten(), 
                segment_times
            ])
        
        solution = self._solver(
            x0=self._initial_guess,
            lbx=self._opt_bounds_states_lb + self._opt_bounds_controls_lb,
            ubx=self._opt_bounds_states_ub + self._opt_bounds_controls_ub,
            p=parameters
        )
        
        # Update initial guess and store solution
        self._initial_guess = solution['x'].full().flatten()
        self._current_solution = self._store_solution(solution, segment_times)
        
        return solution
    
    def create_time_optimal_solver(self):
        """Create solver for time-optimal phase"""
        optimization_problem = {
            'f': self._cost_time,
            'x': ca.vertcat(*(self._opt_vars_states + self._opt_vars_controls + self._opt_vars_times)),
            'p': ca.vertcat(*(self._parameters_initial_state + self._parameters_waypoints)),
            'g': ca.vertcat(*(self._constraints_dynamics + self._constraints_waypoints)),
        }
        
        self._time_solver = ca.nlpsol('time_optimal', 'ipopt', optimization_problem, self._time_solver_options)
        self._warm_start_solver = ca.nlpsol('time_optimal_warm', 'ipopt', optimization_problem, self._warm_start_options)
        
        # Initialize Lagrange multipliers
        solver_size = self._time_solver.size_in(6)[0]
        self._lambda_x = np.zeros(solver_size)
        self._lambda_g = np.zeros(self._time_solver.size_in(7)[0])
    
    def solve_time_optimal(self, initial_state: np.ndarray, 
                          waypoints: np.ndarray, 
                          warm_start: bool = False) -> Dict[str, Any]:
        """Solve time-optimal optimization problem"""
        if self._loop_flag:
            parameters = waypoints.flatten()
        else:
            parameters = np.concatenate([initial_state, waypoints.flatten()])
        
        solver = self._warm_start_solver if warm_start else self._time_solver
        
        solution = solver(
            x0=self._current_solution['full_vector'],
            lam_x0=self._lambda_x,
            lam_g0=self._lambda_g,
            lbx=(self._opt_bounds_states_lb + self._opt_bounds_controls_lb + self._opt_bounds_times_lb),
            ubx=(self._opt_bounds_states_ub + self._opt_bounds_controls_ub + self._opt_bounds_times_ub),
            lbg=(self._constraints_dynamics_lb + self._constraints_waypoints_lb),
            ubg=(self._constraints_dynamics_ub + self._constraints_waypoints_ub),
            p=parameters
        )
        
        # Update solution and Lagrange multipliers
        self._current_solution = self._store_solution(solution, solution['x'].full().flatten()[-self._waypoint_num:])
        self._lambda_x = solution["lam_x"].full().flatten()
        self._lambda_g = solution["lam_g"].full().flatten()
        
        return solution
    
    def _store_solution(self, solution: Dict[str, Any], segment_times: np.ndarray) -> Dict[str, Any]:
        """Extract and store solution in a structured format"""
        full_vector = solution['x'].full().flatten()
        
        return {
            'full_vector': full_vector,
            'segment_times': segment_times,
            'states': full_vector[:self._total_horizon * self._state_dim].reshape(-1, self._state_dim),
            'controls': full_vector[self._total_horizon * self._state_dim: 
                                   self._total_horizon * (self._state_dim + self._control_dim)].reshape(-1, self._control_dim),
        }


def save_trajectory_to_csv(solution: Dict[str, Any], planner: TimeOptimalPlanner, 
                          filename: str):
    """
    Save trajectory to CSV file
    
    Format: t, p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, w_x, w_y, w_z, u_1, u_2, u_3, u_4
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        header = ['t', 'p_x', 'p_y', 'p_z', 'v_x', 'v_y', 'v_z', 
                  'q_w', 'q_x', 'q_y', 'q_z', 'w_x', 'w_y', 'w_z',
                  'u_1', 'u_2', 'u_3', 'u_4']
        writer.writerow(header)
        
        # Extract data from solution
        states = solution['states']
        controls = solution['controls']
        segment_times = solution['segment_times']
        
        # Write initial point
        current_time = 0.0
        writer.writerow([current_time] + 
                       list(states[0, :]) + 
                       list(controls[0, :]) if controls.shape[0] > 0 else [0, 0, 0, 0])
        
        # Write remaining points
        point_idx = 0
        for segment_idx, segment_time in enumerate(segment_times):
            for _ in range(planner._segment_points[segment_idx]):
                point_idx += 1
                if point_idx >= len(states):
                    break
                
                current_time += segment_time
                control = controls[point_idx, :] if point_idx < len(controls) else [0, 0, 0, 0]
                
                writer.writerow([current_time] + 
                               list(states[point_idx, :]) + 
                               list(control))


def calculate_segment_points(gates_positions: List[List[float]], 
                           points_per_meter: float, 
                           loop: bool, 
                           initial_position: List[float] = [0, 0, 0]) -> List[int]:
    """
    Calculate number of discretization points for each segment based on distance
    
    Args:
        gates_positions: List of gate positions
        points_per_meter: Number of discretization points per meter
        loop: Whether the trajectory is closed-loop
        initial_position: Starting position
    
    Returns:
        List of point counts for each segment
    """
    if loop and gates_positions:
        initial_position = gates_positions[-1]
    
    segment_points = []
    
    # First segment: from initial position to first gate
    distance = np.linalg.norm(np.array(gates_positions[0]) - np.array(initial_position))
    segment_points.append(int(distance * points_per_meter))
    
    # Remaining segments: between gates
    for i in range(len(gates_positions) - 1):
        distance = np.linalg.norm(np.array(gates_positions[i]) - np.array(gates_positions[i + 1]))
        segment_points.append(int(distance * points_per_meter))
    
    return segment_points


# Example usage
if __name__ == "__main__":
    # Load quadrotor model
    quadrotor = QuadrotorModel(BASEPATH + 'quad/quad_real.yaml')
    
    # Example waypoints
    waypoints = np.array([
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 1.5],
        [3.0, 0.0, 2.0],
    ]).T  # Transpose to match expected shape (3 x N)
    
    # Calculate segment points
    segment_points = calculate_segment_points(
        waypoints.T.tolist(),  # Convert back to list of positions
        points_per_meter=2.0,
        loop=False
    )
    
    # Create planner
    planner = TimeOptimalPlanner(
        quad=quadrotor,
        waypoint_num=waypoints.shape[1],
        segment_points=segment_points,
        loop_flag=False,
        tolerance=0.01
    )
    
    # Create solvers
    planner.create_warm_start_solver()
    planner.create_time_optimal_solver()
    
    # Initial state (hovering at origin)
    initial_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    
    # Initial segment times guess
    initial_segment_times = np.ones(waypoints.shape[1]) * 0.1
    
    warm_solution = planner.solve_warm_start(initial_state, waypoints, initial_segment_times)
    
    # Solve time-optimal problem
    print("Solving time-optimal problem...")
    time_optimal_solution = planner.solve_time_optimal(initial_state, waypoints, warm_start=True)
    
    # Save trajectory
    save_trajectory_to_csv(planner._current_solution, planner, "optimal_trajectory.csv")
