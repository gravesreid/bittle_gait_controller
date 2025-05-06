import matplotlib.pyplot as plt
import numpy as np
from .kinematics import KinematicsHelper
from .mpc_config import MPCConfig 
class DataLogger:
    def __init__(self):
        self.history = {
            'timesteps': [],
            'planned_angles': [],
            'actual_angles': [],  # Angles at MPC steps
            'planned_com': [],
            'actual_com': [],     # CoM at MPC steps
            'control_signals': [],
            'reference_angles': [],
            # New: Full trajectories at simulation steps
            'full_com_trajectory': [],
            'full_angle_trajectory': [],
            'full_timesteps': []
        }
    def log_pid_step(self, com, angles, timestep):
        """Log data at every PID/simulation timestep."""
        self.history['full_com_trajectory'].append(com)
        self.history['full_angle_trajectory'].append(angles)
        self.history['full_timesteps'].append(timestep) 
    def log_data(self, timestep, 
                planned_angles, planned_com,
                actual_angles, actual_com,
                controls, reference_angles, log_file):
        """Store all relevant data from one timestep"""
        self.history['timesteps'].append(timestep)
        self.history['planned_angles'].append(np.array(planned_angles))
        self.history['planned_com'].append(np.array(planned_com))
        self.history['actual_angles'].append(np.array(actual_angles))
        self.history['actual_com'].append(np.array(actual_com))
        self.history['control_signals'].append(np.array(controls))
        self.history['reference_angles'].append(np.array(reference_angles))  # <-- Log this too
         # Write to log file
        log_entry = (
            f"\n[Timestep {timestep:.3f}]\n"
            f"Planned Angles: {np.rad2deg(planned_angles).round(2)}\n"
            f"Actual Angles:  {np.rad2deg(actual_angles).round(2)}\n"
            f"Planned COM:    X={planned_com[0]:.4f}, Z={planned_com[1]:.4f}, Yaw={planned_com[2]:.4f}\n"
            f"Actual COM:     X={actual_com[0]:.4f}, Z={actual_com[1]:.4f}, Yaw={actual_com[2]:.4f}\n"
            f"Controls:       {controls.round(4)}\n"
            f"Reference Angles: {np.rad2deg(reference_angles).round(2)}\n"
        )
        log_file.write(log_entry)

    
    def plot_results(self, skill_name, reference_traj, kinematics):
        # Data extraction and processing
        full_com = np.array(self.history['full_com_trajectory'])
        full_angles = np.array(self.history['full_angle_trajectory'])
        full_time = np.array(self.history['full_timesteps'])
        
        # Generate reference trajectories
        sim_steps = len(full_time)
        gait_length = reference_traj.shape[0]
        num_repeats = (sim_steps // gait_length) + 1
        full_ref_angles = np.tile(reference_traj, (num_repeats, 1))[:sim_steps]
        full_ref_com = np.array([kinematics.joints_to_com(angles) for angles in full_ref_angles])

        # Create separate figures
        plt.figure(figsize=(20, 12))
        
        # Joint Angles Plotting Section
        for i in range(8):
            plt.subplot(4, 2, i+1)
            plt.plot(full_time, np.rad2deg(full_angles[:, i]), label='Actual')
            plt.plot(full_time, np.rad2deg(full_ref_angles[:, i]), '--', label='Reference')
            
            if 'planned_angles' in self.history:
                planned_angles = np.array(self.history['planned_angles'])
                
                # Ensure 2D shape
                if planned_angles.ndim == 1:
                    planned_angles = np.atleast_2d(planned_angles)
                    
                # Only plot if data exists for this joint
                if planned_angles.size > 0 and i < planned_angles.shape[1]:
                    mpc_time = np.array(self.history['timesteps'])
                    plt.plot(mpc_time, np.rad2deg(planned_angles[:, i]), ':', label='Planned')

        plt.tight_layout()
        plt.savefig(f'{skill_name}_joint_angles.png')
        plt.show()

        # CoM Figure (2x1 grid)
        plt.figure(figsize=(12, 8))
        
        # X Position
        plt.subplot(2, 1, 1)
        plt.plot(full_time, full_com[:, 0], label='Actual')
        plt.plot(full_time, full_ref_com[:, 0], '--', label='Reference')
        
        if 'planned_com' in self.history:
            mpc_time = np.array(self.history['timesteps'])
            planned_com = np.array(self.history['planned_com'])
            
            # Ensure 2D shape for CoM data
            if planned_com.ndim == 1:
                planned_com = np.atleast_2d(planned_com)
            
            # Only plot if we have X position data
            if planned_com.size > 0 and planned_com.shape[1] >= 1:
                plt.plot(mpc_time, planned_com[:, 0], ':', label='Planned')
        
        plt.title('CoM X Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.ylim(-100,100)
        plt.grid(True)
        plt.legend()

        # Z Position
        plt.subplot(2, 1, 2)
        plt.plot(full_time, full_com[:, 1], label='Actual')
        plt.plot(full_time, full_ref_com[:, 1], '--', label='Reference')
        
        if 'planned_com' in self.history:
            mpc_time = np.array(self.history['timesteps'])
            planned_com = np.array(self.history['planned_com'])
            
            # Ensure 2D shape for CoM data
            if planned_com.ndim == 1:
                planned_com = np.atleast_2d(planned_com)
            
            # Only plot if we have Z position data
            if planned_com.size > 0 and planned_com.shape[1] >= 2:
                plt.plot(mpc_time, planned_com[:, 1], ':', label='Planned')
        
        plt.title('CoM Z Position')
        plt.xlabel('Time (s)')

        plt.ylabel('Height (m)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{skill_name}_com_positions.png')
        plt.show()
            
    def plot_xy_displacement(self, skill_name):
        # Extract CoM trajectory (x,y positions)
        com_trajectory = np.array(self.history['full_com_trajectory'])[:,:2]  # Take only x,y components
        
        # Calculate displacement from starting position
        displacement = com_trajectory - com_trajectory[0]
        total_distance = np.sqrt(displacement[:,0]**2 + displacement[:,1]**2)
        
        # Create plot with large fonts
        plt.figure(figsize=(16, 12))
        
        # Plot XY trajectory
        plt.subplot(2, 1, 1)
        plt.plot(com_trajectory[:,0], com_trajectory[:,1], 'b-', linewidth=3, label='CoM Path')
        plt.plot(com_trajectory[0,0], com_trajectory[0,1], 'go', markersize=15, label='Start')
        plt.plot(com_trajectory[-1,0], com_trajectory[-1,1], 'ro', markersize=15, label='End')
        plt.xlabel('X Position (m)', fontsize=30)
        plt.ylabel('Y Position (m)', fontsize=30)
        plt.title(f'{skill_name} - CoM XY Trajectory', fontsize=30)
        plt.grid(True)
        plt.legend(fontsize=24)
        plt.axis('equal')
        plt.tick_params(axis='both', which='major', labelsize=24)

        # Plot displacement magnitude over time
        plt.subplot(2, 1, 2)
        time = np.array(self.history['full_timesteps'])
        plt.plot(time, total_distance, 'k-', linewidth=3)
        plt.xlabel('Time (s)', fontsize=30)
        plt.ylabel('Displacement Magnitude (m)', fontsize=30)
        plt.title('Total Displacement from Start', fontsize=30)
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=24)

        plt.tight_layout()
        plt.savefig(f'{skill_name}_xy_displacement.png', dpi=100, bbox_inches='tight')
        plt.show()
        
    def plot_final_displacement(self, skill_name):
        # Extract XY positions
        com_trajectory = np.array(self.history['full_com_trajectory'])[:,:2]  # Get x,y columns
        
        # Calculate final displacement vector and magnitude
        start_pos = com_trajectory[0]
        end_pos = com_trajectory[-1]
        displacement_vector = end_pos - start_pos
        displacement_magnitude = np.linalg.norm(displacement_vector)
        
        # Create the plot with large fonts
        plt.figure(figsize=(16, 12))
        
        # Plot displacement vector
        plt.quiver(start_pos[0], start_pos[1], 
                displacement_vector[0], displacement_vector[1],
                angles='xy', scale_units='xy', scale=1,
                color='r', width=0.01, label=f'Displacement: {displacement_magnitude:.3f}m')
        
        # Mark start and end points
        plt.plot(start_pos[0], start_pos[1], 'go', markersize=20, label='Start')
        plt.plot(end_pos[0], end_pos[1], 'ro', markersize=20, label='End')
        
        # Formatting with large fonts
        plt.xlabel('X Position (m)', fontsize=30)
        plt.ylabel('Y Position (m)', fontsize=30)
        plt.title(f'{skill_name} - Final XY Displacement', fontsize=30)
        plt.grid(True)
        plt.legend(fontsize=24)
        plt.axis('equal')
        plt.tick_params(axis='both', which='major', labelsize=24)
        
        # Annotate the magnitude with large font
        plt.annotate(f'Distance: {displacement_magnitude:.3f}m',
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', va='top', fontsize=28,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{skill_name}_final_displacement.png', dpi=100, bbox_inches='tight')
        plt.show()