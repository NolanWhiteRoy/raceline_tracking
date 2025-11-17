"""
SE380 Project: Raceline Tracking Controller
Implements a two-level control architecture:
- Upper level: Path tracking using Pure Pursuit
- Lower level: PID controllers for steering and velocity
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple

# Controller state storage
controller_state = {
    'last_idx': 0,
    'velocity_integral': 0.0,
    'velocity_prev_error': 0.0,
}


def find_closest_point_on_track(position: ArrayLike, track: ArrayLike, 
                                last_idx: int = 0) -> Tuple[int, float]:
    """
    Find the closest point on track to current position.
    """
    n_points = len(track)
    
    # Always do global search to ensure we find the true closest point
    distances = np.linalg.norm(track - position, axis=1)
    closest_idx = np.argmin(distances)
    min_dist = distances[closest_idx]
    
    return closest_idx, min_dist


def find_lookahead_point(position: ArrayLike, track: ArrayLike, 
                         start_idx: int, lookahead_dist: float) -> Tuple[ArrayLike, int]:
    """
    Find a point on the track at lookahead distance ahead.
    """
    n_points = len(track)
    accumulated = 0.0
    current_idx = start_idx
    
    for _ in range(n_points):
        next_idx = (current_idx + 1) % n_points
        segment_dist = np.linalg.norm(track[next_idx] - track[current_idx])
        
        if accumulated + segment_dist >= lookahead_dist:
            # Interpolate to get exact lookahead point
            ratio = (lookahead_dist - accumulated) / max(segment_dist, 0.001)
            ratio = np.clip(ratio, 0, 1)
            lookahead_point = track[current_idx] * (1 - ratio) + track[next_idx] * ratio
            return lookahead_point, next_idx
        
        accumulated += segment_dist
        current_idx = next_idx
    
    return track[current_idx], current_idx


def calculate_curvature(track: ArrayLike, idx: int, window: int = 5) -> float:
    """
    Estimate curvature at a point using neighboring points.
    """
    n_points = len(track)
    
    idx_prev = (idx - window) % n_points
    idx_next = (idx + window) % n_points
    
    p1 = track[idx_prev]
    p2 = track[idx]
    p3 = track[idx_next]
    
    # Calculate curvature using three points
    v1 = p2 - p1
    v2 = p3 - p2
    
    cross = np.cross(v1, v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) ** 1.5
    
    if denom > 1e-6:
        curvature = abs(2 * cross / denom)
    else:
        curvature = 0.0
    
    return curvature


def controller(state: ArrayLike, parameters: ArrayLike, racetrack) -> ArrayLike:
    """
    Main controller using Pure Pursuit for path tracking.
    Conservative approach that prioritizes staying on track.
    
    Args:
        state: [x, y, steering, velocity, heading]
        parameters: Vehicle parameters
        racetrack: Track object with centerline
    
    Returns:
        [desired_steering, desired_velocity]
    """
    # Extract state
    x, y = state[0], state[1]
    velocity = state[3]
    heading = state[4]
    position = np.array([x, y])
    
    # Use centerline as reference path
    track = racetrack.centerline
    
    # Find closest point on track
    last_idx = controller_state.get('last_idx', 0)
    closest_idx, cross_track_error = find_closest_point_on_track(position, track, last_idx)
    controller_state['last_idx'] = closest_idx
    
    # Conservative lookahead distance
    # Longer lookahead for smoother tracking
    base_lookahead = 8.0
    
    # Scale with velocity but not too much
    velocity_factor = 0.15
    lookahead = base_lookahead + velocity_factor * min(velocity, 30)
    
    # Slightly shorter lookahead when off track, but not too short
    if cross_track_error > 2.0:
        lookahead *= 0.8
    elif cross_track_error > 1.0:
        lookahead *= 0.9
    
    # Keep lookahead in reasonable range
    lookahead = np.clip(lookahead, 6.0, 15.0)
    
    # Find lookahead point
    lookahead_point, lookahead_idx = find_lookahead_point(position, track, closest_idx, lookahead)
    
    # Pure Pursuit steering calculation
    dx = lookahead_point[0] - x
    dy = lookahead_point[1] - y
    
    # Transform to vehicle frame
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    target_x = cos_h * dx + sin_h * dy
    target_y = -sin_h * dx + cos_h * dy
    
    # Calculate steering angle using standard Pure Pursuit
    L = np.sqrt(target_x**2 + target_y**2)
    if L > 0.1:
        wheelbase = 3.6
        # Standard pure pursuit formula
        desired_steering = np.arctan(2.0 * wheelbase * target_y / (L * L))
        
        # Very gentle correction for cross-track error
        if cross_track_error > 1.5:
            # Small proportional correction
            error_correction = 0.05 * (cross_track_error - 1.5) * np.sign(target_y)
            error_correction = np.clip(error_correction, -0.1, 0.1)
            desired_steering += error_correction
    else:
        desired_steering = 0.0
    
    # Limit steering angle
    desired_steering = np.clip(desired_steering, -0.9, 0.9)
    
    # Conservative speed control
    curvature = calculate_curvature(track, lookahead_idx)
    
    # Very conservative base speeds
    if curvature > 0.1:
        target_velocity = 12.0  # Very slow in tight corners
    elif curvature > 0.05:
        target_velocity = 18.0  # Slow in moderate corners
    elif curvature > 0.02:
        target_velocity = 25.0  # Moderate in gentle curves
    else:
        target_velocity = 35.0  # Conservative max speed on straights
    
    # Further reduce speed if off track
    if cross_track_error > 2.0:
        target_velocity = min(target_velocity, 15.0)
    elif cross_track_error > 1.0:
        target_velocity *= 0.85
    
    # Smooth speed transitions - don't change speed too quickly
    if velocity > 0:
        max_speed_change = 10.0  # Max 10 m/s difference from current
        target_velocity = np.clip(target_velocity, 
                                  velocity - max_speed_change, 
                                  velocity + max_speed_change)
    
    return np.array([desired_steering, target_velocity])


def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    """
    Lower-level controller for tracking desired steering and velocity.
    Conservative gains for stability.
    
    Args:
        state: Current vehicle state
        desired: [desired_steering, desired_velocity]
        parameters: Vehicle parameters
    
    Returns:
        [steering_velocity, acceleration]
    """
    assert(desired.shape == (2,))
    
    current_steering = state[2]
    current_velocity = state[3]
    
    desired_steering = desired[0]
    desired_velocity = desired[1]
    
    # Steering control with moderate gain
    steering_error = desired_steering - current_steering
    
    # Conservative steering gain
    if current_velocity < 10:
        steering_gain = 2.0  # Moderate gain at low speed
    elif current_velocity < 25:
        steering_gain = 1.8
    else:
        steering_gain = 1.5  # Lower gain at high speed for stability
    
    steering_velocity = steering_gain * steering_error
    steering_velocity = np.clip(steering_velocity, -0.4, 0.4)
    
    # Velocity control - simple proportional
    velocity_error = desired_velocity - current_velocity
    
    # Conservative acceleration
    kp = 1.5  # Lower gain for smoother acceleration
    acceleration = kp * velocity_error
    
    # Limit acceleration more conservatively
    max_accel = 15.0  # Less than the 20 m/s² limit for smoother control
    max_decel = -15.0
    acceleration = np.clip(acceleration, max_decel, max_accel)
    
    return np.array([steering_velocity, acceleration])