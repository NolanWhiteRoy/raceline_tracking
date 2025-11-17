#!/usr/bin/env python3
"""
Test the tracking performance of the controller
Shows detailed tracking metrics in text format
"""

import numpy as np
import sys
from racetrack import RaceTrack
from racecar import RaceCar
from controller import controller, lower_controller, controller_state

def test_tracking(track_file, max_steps=200):
    """Run tracking test and output results."""
    
    print("="*60)
    print("RACELINE TRACKING TEST")
    print("="*60)
    
    # Load track
    track = RaceTrack(track_file)
    car = RaceCar(track.initial_state.T)
    
    # Reset controller state
    controller_state.clear()
    controller_state.update({
        'velocity_integral': 0.0,
        'velocity_prev_error': 0.0,
        'steering_integral': 0.0,
        'steering_prev_error': 0.0,
        'closest_point_idx': 0,
        'lap_progress': 0.0,
    })
    
    # Tracking metrics
    max_cross_track = 0
    total_cross_track = 0
    violations = 0
    lap_complete = False
    
    print(f"Track: {len(track.centerline)} points")
    print(f"Initial state: {car.state}")
    print("\nStarting simulation...")
    print("-"*60)
    
    for step in range(max_steps):
        # Get control
        desired = controller(car.state, car.parameters, track)
        control = lower_controller(car.state, desired, car.parameters)
        
        # Check position relative to track
        pos = car.state[:2]
        distances = np.linalg.norm(track.centerline - pos, axis=1)
        closest_idx = np.argmin(distances)
        cross_track_error = distances[closest_idx]
        
        # Check if within track bounds
        to_right = np.linalg.norm(pos - track.right_boundary[closest_idx])
        to_left = np.linalg.norm(pos - track.left_boundary[closest_idx])
        track_width_r = np.linalg.norm(track.right_boundary[closest_idx] - track.centerline[closest_idx])
        track_width_l = np.linalg.norm(track.left_boundary[closest_idx] - track.centerline[closest_idx])
        
        is_violating = cross_track_error > max(track_width_r, track_width_l)
        
        # Update metrics
        max_cross_track = max(max_cross_track, cross_track_error)
        total_cross_track += cross_track_error
        if is_violating:
            violations += 1
        
        # Print status every 10 steps
        if step % 10 == 0:
            print(f"Step {step:3d}: Pos=({car.state[0]:6.1f}, {car.state[1]:6.1f}) "
                  f"V={car.state[3]:5.1f} m/s "
                  f"δ={np.rad2deg(car.state[2]):5.1f}° "
                  f"CTE={cross_track_error:4.2f} m "
                  f"{'VIOLATION' if is_violating else 'OK'}")
        
        # Update car
        car.update(control)
        
        # Check lap completion (back near start)
        if step > 50:  # After initial movement
            dist_to_start = np.linalg.norm(car.state[:2] - track.initial_state[:2])
            if dist_to_start < 5.0 and car.state[3] > 5.0:
                lap_complete = True
                print(f"\n✓ LAP COMPLETE at step {step}!")
                break
        
        # Safety check
        if cross_track_error > 20:
            print(f"\n✗ Too far off track at step {step}!")
            break
    
    # Final report
    print("-"*60)
    print("RESULTS:")
    print(f"  Steps simulated: {step+1}")
    print(f"  Lap complete: {lap_complete}")
    print(f"  Track violations: {violations}/{step+1} ({100*violations/(step+1):.1f}%)")
    print(f"  Max cross-track error: {max_cross_track:.2f} m")
    print(f"  Avg cross-track error: {total_cross_track/(step+1):.2f} m")
    print(f"  Final velocity: {car.state[3]:.1f} m/s")
    print(f"  Final position: ({car.state[0]:.1f}, {car.state[1]:.1f})")
    
    # Success criteria
    success = violations < (step+1)*0.1 and max_cross_track < 10
    print(f"\n{'✓ TEST PASSED' if success else '✗ TEST FAILED'}")
    print("="*60)
    
    return success

if __name__ == "__main__":
    track = sys.argv[1] if len(sys.argv) > 1 else "./racetracks/Montreal.csv"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    
    success = test_tracking(track, steps)
    sys.exit(0 if success else 1)
