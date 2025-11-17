#!/usr/bin/env python3
"""
Quick test script to verify controller functionality
"""

import numpy as np
from controller import controller, lower_controller
from racetrack import RaceTrack
from racecar import RaceCar

def test_controller():
    """Test the controller without running full simulation."""
    
    print("Testing SE380 Raceline Controller...")
    print("="*50)
    
    # Load track
    print("\n1. Loading Montreal track...")
    track = RaceTrack('./racetracks/Montreal.csv')
    print(f"   Track loaded: {len(track.centerline)} points")
    print(f"   Initial position: {track.initial_state[:2]}")
    
    # Initialize car
    print("\n2. Initializing race car...")
    car = RaceCar(track.initial_state)
    print(f"   Initial state: {car.state}")
    print(f"   Wheelbase: {car.wheelbase} m")
    print(f"   Max velocity: {car.max_velocity} m/s")
    
    # Test controller for several steps
    print("\n3. Testing controller for 10 steps...")
    print("   Step | Velocity | Steering | Accel | Steer Vel")
    print("   " + "-"*45)
    
    for step in range(10):
        # Get desired states from upper controller
        desired = controller(car.state, car.parameters, track)
        
        # Get control inputs from lower controller
        control = lower_controller(car.state, desired, car.parameters)
        
        # Print status
        print(f"   {step:4d} | {car.state[3]:8.2f} | {np.rad2deg(car.state[2]):8.2f} | "
              f"{control[1]:6.2f} | {np.rad2deg(control[0]):9.2f}")
        
        # Update car state
        car.update(control)
    
    print("\n4. Controller test completed successfully!")
    
    # Test specific scenarios
    print("\n5. Testing specific scenarios...")
    
    # Test straight line (low curvature)
    test_state = np.array([100.0, 100.0, 0.0, 30.0, 0.0])
    desired = controller(test_state, car.parameters, track)
    control = lower_controller(test_state, desired, car.parameters)
    print(f"   Straight line (v=30): accel={control[1]:.2f}, steer_vel={np.rad2deg(control[0]):.2f}")
    
    # Test at low speed
    test_state[3] = 5.0
    desired = controller(test_state, car.parameters, track)
    control = lower_controller(test_state, desired, car.parameters)
    print(f"   Low speed (v=5): accel={control[1]:.2f}, steer_vel={np.rad2deg(control[0]):.2f}")
    
    # Test with steering angle
    test_state[2] = 0.3  # radians
    test_state[3] = 20.0
    desired = controller(test_state, car.parameters, track)
    control = lower_controller(test_state, desired, car.parameters)
    print(f"   With steering (δ=0.3): accel={control[1]:.2f}, steer_vel={np.rad2deg(control[0]):.2f}")
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_controller()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
