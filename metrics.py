"""
Performance Metrics Tracking for SE380 Raceline Controller
Tracks lap times, errors, and generates performance plots
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
from datetime import datetime


class PerformanceMetrics:
    """Track and analyze controller performance metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics for a new run."""
        self.timestamps = []
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.steering_angles = []
        self.steering_velocities = []
        self.cross_track_errors = []
        self.heading_errors = []
        self.curvatures = []
        self.lap_progress = []
        
        # Lap statistics
        self.lap_time = None
        self.max_velocity = 0
        self.avg_velocity = 0
        self.track_violations = 0
        self.total_distance = 0
        
    def update(self, timestamp: float, state: np.ndarray, control: np.ndarray,
               cross_track_error: float = 0, heading_error: float = 0,
               curvature: float = 0, progress: float = 0):
        """Update metrics with current state and control values."""
        self.timestamps.append(timestamp)
        self.positions.append(state[:2].copy())
        self.velocities.append(state[3])
        self.steering_angles.append(state[2])
        self.accelerations.append(control[1])
        self.steering_velocities.append(control[0])
        self.cross_track_errors.append(cross_track_error)
        self.heading_errors.append(heading_error)
        self.curvatures.append(curvature)
        self.lap_progress.append(progress)
        
        # Update statistics
        if state[3] > self.max_velocity:
            self.max_velocity = state[3]
            
    def finalize_lap(self, lap_time: float, violations: int):
        """Finalize lap statistics."""
        self.lap_time = lap_time
        self.track_violations = violations
        
        if len(self.velocities) > 0:
            self.avg_velocity = np.mean(self.velocities)
            
        # Calculate total distance
        if len(self.positions) > 1:
            positions = np.array(self.positions)
            diffs = np.diff(positions, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            self.total_distance = np.sum(distances)
    
    def generate_report(self) -> Dict:
        """Generate performance report dictionary."""
        report = {
            'lap_time': self.lap_time,
            'max_velocity': self.max_velocity,
            'avg_velocity': self.avg_velocity,
            'track_violations': self.track_violations,
            'total_distance': self.total_distance,
            'avg_cross_track_error': np.mean(np.abs(self.cross_track_errors)) if self.cross_track_errors else 0,
            'max_cross_track_error': np.max(np.abs(self.cross_track_errors)) if self.cross_track_errors else 0,
            'avg_heading_error': np.mean(np.abs(self.heading_errors)) if self.heading_errors else 0,
            'timestamp': datetime.now().isoformat()
        }
        return report
    
    def save_report(self, filename: str):
        """Save performance report to JSON file."""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Performance report saved to {filename}")
    
    def plot_metrics(self, save_path: str = None):
        """Generate comprehensive performance plots."""
        if len(self.timestamps) == 0:
            print("No data to plot")
            return
            
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Controller Performance Metrics', fontsize=16)
        
        t = np.array(self.timestamps)
        
        # Plot 1: Velocity Profile
        ax = axes[0, 0]
        ax.plot(t, self.velocities, 'b-', label='Velocity')
        ax.axhline(y=self.avg_velocity, color='r', linestyle='--', label=f'Avg: {self.avg_velocity:.1f} m/s')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Acceleration Profile
        ax = axes[0, 1]
        ax.plot(t, self.accelerations, 'g-')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title('Acceleration Commands')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Steering Angle
        ax = axes[0, 2]
        ax.plot(t, np.rad2deg(self.steering_angles), 'r-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Steering Angle (deg)')
        ax.set_title('Steering Angle')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Steering Velocity
        ax = axes[1, 0]
        ax.plot(t, np.rad2deg(self.steering_velocities), 'm-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Steering Rate (deg/s)')
        ax.set_title('Steering Velocity Commands')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Cross-track Error
        ax = axes[1, 1]
        if self.cross_track_errors:
            ax.plot(t, self.cross_track_errors, 'b-')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Cross-track Error (m)')
            ax.set_title('Lateral Deviation from Path')
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Track Curvature vs Speed
        ax = axes[1, 2]
        if self.curvatures:
            ax.scatter(self.curvatures, self.velocities, alpha=0.5, s=1)
            ax.set_xlabel('Track Curvature (1/m)')
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title('Speed vs Curvature')
            ax.grid(True, alpha=0.3)
        
        # Plot 7: XY Trajectory
        ax = axes[2, 0]
        if len(self.positions) > 0:
            positions = np.array(self.positions)
            ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7)
            ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
            ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Vehicle Trajectory')
            ax.legend()
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
        
        # Plot 8: Lap Progress
        ax = axes[2, 1]
        if self.lap_progress:
            ax.plot(t, self.lap_progress, 'g-')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Lap Progress (%)')
            ax.set_title('Lap Completion Progress')
            ax.grid(True, alpha=0.3)
        
        # Plot 9: Performance Summary Text
        ax = axes[2, 2]
        ax.axis('off')
        summary_text = f"""Performance Summary:
        
Lap Time: {self.lap_time:.2f} s
Max Velocity: {self.max_velocity:.1f} m/s
Avg Velocity: {self.avg_velocity:.1f} m/s
Track Violations: {self.track_violations}
Total Distance: {self.total_distance:.1f} m

Tracking Performance:
Avg Cross-track Error: {np.mean(np.abs(self.cross_track_errors)) if self.cross_track_errors else 0:.2f} m
Max Cross-track Error: {np.max(np.abs(self.cross_track_errors)) if self.cross_track_errors else 0:.2f} m
"""
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Performance plots saved to {save_path}")
        
        return fig


class MetricsRecorder:
    """Singleton class to record metrics during simulation."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.metrics = PerformanceMetrics()
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def reset(self):
        """Reset metrics for new run."""
        self.metrics.reset()
    
    def update(self, *args, **kwargs):
        """Update metrics."""
        self.metrics.update(*args, **kwargs)
    
    def finalize(self, *args, **kwargs):
        """Finalize lap."""
        self.metrics.finalize_lap(*args, **kwargs)
    
    def save_and_plot(self, track_name: str):
        """Save report and generate plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"results/{track_name}_report_{timestamp}.json"
        plot_file = f"results/{track_name}_plots_{timestamp}.png"
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs("results", exist_ok=True)
        
        self.metrics.save_report(report_file)
        self.metrics.plot_metrics(plot_file)
        
        return report_file, plot_file
