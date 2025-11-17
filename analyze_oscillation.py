#!/usr/bin/env python3
"""
Analyze controller oscillation behavior with detailed text output
"""

import numpy as np
import sys
from racetrack import RaceTrack
from racecar import RaceCar
from controller import controller, lower_controller, controller_state

def analyze_oscillation(track_file, start_step=0, num_steps=100):
    """
    Run simulation and analyze oscillation patterns.
    """
    print("="*70)
    print("OSCILLATION ANALYSIS")
    print("="*70)
    
    # Initialize
    track = RaceTrack(track_file)
    car = RaceCar(track.initial_state.T)
    
    # Reset controller state
    controller_state.clear()
    controller_state['last_idx'] = 0
    controller_state['velocity_integral'] = 0.0
    controller_state['velocity_prev_error'] = 0.0
    
    # Skip to problematic area if requested
    for _ in range(start_step):
        desired = controller(car.state, car.parameters, track)
        control = lower_controller(car.state, desired, car.parameters)
        car.update(control)
    
    # Tracking variables
    steering_history = []
    cte_history = []
    side_history = []  # Track which side of centerline
    
    print(f"\nStarting from step {start_step}")
    print(f"Initial position: ({car.state[0]:.1f}, {car.state[1]:.1f})")
    print("\n" + "-"*70)
    print("Step | Pos X | Pos Y | Vel | Steer | Des.St | CTE | Side | Status")
    print("-"*70)
    
    for step in range(num_steps):
        # Get control
        desired = controller(car.state, car.parameters, track)
        control = lower_controller(car.state, desired, car.parameters)
        
        # Find track position
        pos = car.state[:2]
        distances = np.linalg.norm(track.centerline - pos, axis=1)
        closest_idx = np.argmin(distances)
        cte = distances[closest_idx]
        
        # Determine which side of centerline
        closest_point = track.centerline[closest_idx]
        
        # Calculate if we're left or right of track
        if closest_idx < len(track.centerline) - 1:
            track_dir = track.centerline[closest_idx + 1] - track.centerline[closest_idx]
        else:
            track_dir = track.centerline[0] - track.centerline[closest_idx]
        
        to_car = pos - closest_point
        cross_product = track_dir[0] * to_car[1] - track_dir[1] * to_car[0]
        side = "L" if cross_product > 0 else "R"
        
        # Check track bounds
        to_right = np.linalg.norm(pos - track.right_boundary[closest_idx])
        to_left = np.linalg.norm(pos - track.left_boundary[closest_idx])
        track_width_r = np.linalg.norm(track.right_boundary[closest_idx] - track.centerline[closest_idx])
        track_width_l = np.linalg.norm(track.left_boundary[closest_idx] - track.centerline[closest_idx])
        
        if cte > track_width_r:
            status = "VIOL-R"
        elif cte > track_width_l:
            status = "VIOL-L"
        elif cte > 3.0:
            status = "FAR"
        elif cte > 1.5:
            status = "OFF"
        else:
            status = "OK"
        
        # Store history
        steering_history.append(car.state[2])
        cte_history.append(cte)
        side_history.append(side)
        
        # Print every 5 steps or when status changes
        if step % 5 == 0 or status != "OK":
            print(f"{step+start_step:4d} | {car.state[0]:5.1f} | {car.state[1]:5.1f} | "
                  f"{car.state[3]:4.1f} | {np.rad2deg(car.state[2]):6.1f} | "
                  f"{np.rad2deg(desired[0]):6.1f} | {cte:4.2f} | {side:4s} | {status}")
        
        # Update car
        car.update(control)
        
        # Check for severe issues
        if cte > 20:
            print(f"\n⚠️ Severely off track at step {step+start_step}")
            break
    
    # Analyze oscillation patterns
    print("\n" + "="*70)
    print("OSCILLATION METRICS")
    print("="*70)
    
    if len(steering_history) > 10:
        # Count sign changes in steering
        sign_changes = 0
        for i in range(1, len(steering_history)):
            if np.sign(steering_history[i]) != np.sign(steering_history[i-1]):
                sign_changes += 1
        
        # Count side switches
        side_switches = 0
        for i in range(1, len(side_history)):
            if side_history[i] != side_history[i-1]:
                side_switches += 1
        
        # Calculate steering variance
        steering_var = np.var(steering_history)
        cte_var = np.var(cte_history)
        
        print(f"Steering sign changes: {sign_changes} ({sign_changes/len(steering_history)*100:.1f}%)")
        print(f"Track side switches: {side_switches} ({side_switches/len(side_history)*100:.1f}%)")
        print(f"Steering variance: {steering_var:.4f}")
        print(f"CTE variance: {cte_var:.4f}")
        print(f"Max CTE: {max(cte_history):.2f} m")
        print(f"Avg CTE: {np.mean(cte_history):.2f} m")
        
        # Detect oscillation
        oscillation_ratio = sign_changes / len(steering_history)
        if oscillation_ratio > 0.3:
            print("\n⚠️ HIGH OSCILLATION DETECTED!")
            print("   Controller is unstable and switching steering too frequently")
        elif oscillation_ratio > 0.15:
            print("\n⚠️ MODERATE OSCILLATION")
            print("   Controller shows some instability")
        else:
            print("\n✓ LOW OSCILLATION")
            print("   Controller is relatively stable")
    
    print("="*70)

if __name__ == "__main__":
    track = sys.argv[1] if len(sys.argv) > 1 else "./racetracks/Montreal.csv"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    analyze_oscillation(track, start, steps)




