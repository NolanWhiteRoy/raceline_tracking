#!/usr/bin/env python3
"""
Test Runner for SE380 Raceline Tracking Controller
Tests the controller on both Montreal and IMS tracks
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

from simulator import RaceTrack, Simulator
from metrics import MetricsRecorder, PerformanceMetrics
from config import CONTROLLER_CONFIG


def run_test(track_file: str, raceline_file: str, track_name: str, 
             headless: bool = False, max_time: float = 300.0):
    """
    Run a single test on a track.
    
    Args:
        track_file: Path to track CSV file
        raceline_file: Path to raceline CSV file  
        track_name: Name of the track for reporting
        headless: If True, run without GUI
        max_time: Maximum simulation time in seconds
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing on {track_name}")
    print(f"{'='*60}")
    
    # Initialize track and simulator
    racetrack = RaceTrack(track_file)
    simulator = Simulator(racetrack)
    
    # Reset metrics
    metrics_recorder = MetricsRecorder.get_instance()
    metrics_recorder.reset()
    
    # Run simulation
    start_time = 0
    time_step = 0.1
    current_time = 0
    
    print(f"Starting simulation (max time: {max_time}s)...")
    
    # Simulation loop
    iteration = 0
    while current_time < max_time and not simulator.lap_finished:
        # Run one step
        success = simulator.run()
        if not success:
            break
            
        # Record metrics (if you want to integrate with metrics.py)
        # This would require modifying simulator.py to expose more data
        
        current_time += time_step
        iteration += 1
        
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"  Time: {current_time:.1f}s, "
                  f"Velocity: {simulator.car.state[3]:.1f} m/s, "
                  f"Violations: {simulator.track_limit_violations}")
    
    # Collect results
    results = {
        'track': track_name,
        'lap_completed': simulator.lap_finished,
        'lap_time': simulator.lap_time_elapsed if simulator.lap_finished else None,
        'track_violations': simulator.track_limit_violations,
        'max_velocity': np.max([simulator.car.state[3]]),  # Would need history
        'avg_velocity': simulator.car.state[3],  # Approximate
        'status': 'completed' if simulator.lap_finished else 'timeout',
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\nResults for {track_name}:")
    print(f"  Lap Completed: {results['lap_completed']}")
    if results['lap_completed']:
        print(f"  Lap Time: {results['lap_time']:.2f} seconds")
    print(f"  Track Violations: {results['track_violations']}")
    print(f"  Status: {results['status']}")
    
    return results


def run_all_tests():
    """Run tests on all available tracks."""
    
    print("\n" + "="*60)
    print("SE380 RACELINE TRACKING CONTROLLER TEST SUITE")
    print("="*60)
    
    # Define test tracks
    tracks = [
        {
            'track_file': './racetracks/Montreal.csv',
            'raceline_file': './racetracks/Montreal_raceline.csv',
            'name': 'Montreal'
        },
        {
            'track_file': './racetracks/IMS.csv',
            'raceline_file': './racetracks/IMS_raceline.csv',
            'name': 'Indianapolis Motor Speedway (IMS)'
        }
    ]
    
    # Run tests
    all_results = []
    for track_info in tracks:
        if os.path.exists(track_info['track_file']):
            results = run_test(
                track_info['track_file'],
                track_info['raceline_file'],
                track_info['name'],
                headless=True,  # Run without GUI for automated testing
                max_time=300.0  # 5 minute timeout
            )
            all_results.append(results)
        else:
            print(f"Warning: Track file not found: {track_info['track_file']}")
    
    # Generate summary report
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for result in all_results:
        print(f"\n{result['track']}:")
        print(f"  Status: {result['status']}")
        if result['lap_completed']:
            print(f"  Lap Time: {result['lap_time']:.2f} seconds")
            print(f"  Track Violations: {result['track_violations']}")
            
            # Calculate score (example scoring function)
            base_score = 100
            time_penalty = min(result['lap_time'] * 0.1, 50)  # Cap at 50 points
            violation_penalty = result['track_violations'] * 5
            score = max(0, base_score - time_penalty - violation_penalty)
            print(f"  Score: {score:.1f}/100")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


def tune_controller(track_file: str, raceline_file: str, track_name: str):
    """
    Helper function to tune controller parameters.
    Runs multiple tests with different parameters.
    """
    print(f"\nTuning controller for {track_name}...")
    
    # Parameter ranges to test
    param_ranges = {
        'velocity_kp': [1.0, 1.5, 2.0, 2.5, 3.0],
        'steering_kp': [1.5, 2.0, 2.5, 3.0, 3.5],
        'lookahead': [10.0, 15.0, 20.0, 25.0]
    }
    
    best_score = float('inf')
    best_params = {}
    
    # Grid search (simplified - you might want more sophisticated optimization)
    for v_kp in param_ranges['velocity_kp']:
        for s_kp in param_ranges['steering_kp']:
            for lookahead in param_ranges['lookahead']:
                # Update config
                CONTROLLER_CONFIG['velocity_controller']['kp'] = v_kp
                CONTROLLER_CONFIG['steering_controller']['kp'] = s_kp
                CONTROLLER_CONFIG['lookahead_distance'] = lookahead
                
                # Run test
                print(f"  Testing: v_kp={v_kp}, s_kp={s_kp}, lookahead={lookahead}")
                results = run_test(track_file, raceline_file, track_name, 
                                 headless=True, max_time=200.0)
                
                # Calculate score (minimize lap time + violations)
                if results['lap_completed']:
                    score = results['lap_time'] + results['track_violations'] * 10
                    if score < best_score:
                        best_score = score
                        best_params = {
                            'velocity_kp': v_kp,
                            'steering_kp': s_kp,
                            'lookahead': lookahead,
                            'lap_time': results['lap_time'],
                            'violations': results['track_violations']
                        }
    
    print(f"\nBest parameters for {track_name}:")
    print(f"  {best_params}")
    
    return best_params


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "tune":
            # Tune controller parameters
            if len(sys.argv) > 2 and sys.argv[2] == "montreal":
                tune_controller('./racetracks/Montreal.csv', 
                              './racetracks/Montreal_raceline.csv',
                              'Montreal')
            elif len(sys.argv) > 2 and sys.argv[2] == "ims":
                tune_controller('./racetracks/IMS.csv',
                              './racetracks/IMS_raceline.csv', 
                              'IMS')
            else:
                print("Usage: python test_runner.py tune [montreal|ims]")
        else:
            # Run single test with visualization
            track_file = sys.argv[1]
            raceline_file = sys.argv[2] if len(sys.argv) > 2 else None
            track_name = os.path.basename(track_file).split('.')[0]
            run_test(track_file, raceline_file, track_name, headless=False)
    else:
        # Run all tests
        run_all_tests()
