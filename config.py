"""
Configuration file for SE380 Raceline Tracking Controller
Contains all tunable parameters for the control system
"""

import numpy as np

# Controller Configuration
CONTROLLER_CONFIG = {
    # Lookahead distance for path tracking (meters)
    'lookahead_distance': 15.0,  # Distance to look ahead on the track
    
    # Velocity Controller (C1) - PID parameters
    'velocity_controller': {
        'kp': 2.0,      # Proportional gain
        'ki': 0.1,      # Integral gain  
        'kd': 0.5,      # Derivative gain
        'integral_limit': 10.0,  # Anti-windup limit
        'target_velocity': 40.0,  # Default target velocity (m/s)
        'corner_velocity': 25.0,  # Velocity for corners (m/s)
        'straight_velocity': 60.0,  # Velocity for straights (m/s)
    },
    
    # Steering Controller (C2) - PID parameters for lateral control
    'steering_controller': {
        'kp': 2.5,      # Proportional gain for cross-track error
        'ki': 0.05,     # Integral gain
        'kd': 1.0,      # Derivative gain  
        'kp_heading': 1.5,  # Proportional gain for heading error
        'integral_limit': 0.5,  # Anti-windup limit (radians)
    },
    
    # Pure Pursuit Controller parameters (alternative/supplementary)
    'pure_pursuit': {
        'min_lookahead': 8.0,   # Minimum lookahead distance
        'max_lookahead': 25.0,  # Maximum lookahead distance
        'speed_factor': 0.3,    # Factor to scale lookahead with speed
    },
    
    # Path tracking parameters
    'path_tracking': {
        'search_radius': 50,  # Number of points to search ahead/behind
        'curvature_window': 5,  # Window size for curvature calculation
    },
    
    # Safety limits
    'safety': {
        'max_lateral_accel': 15.0,  # Maximum lateral acceleration (m/s²)
        'track_margin': 0.5,  # Safety margin from track boundaries (m)
    }
}

# State indices for clarity
STATE_X = 0
STATE_Y = 1
STATE_STEERING = 2
STATE_VELOCITY = 3
STATE_HEADING = 4

# Control input indices
CONTROL_STEERING_VEL = 0
CONTROL_ACCELERATION = 1
