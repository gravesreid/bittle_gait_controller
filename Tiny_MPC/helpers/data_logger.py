import matplotlib.pyplot as plt
import numpy as np
from .kinematics import KinematicsHelper
from .mpc_config import MPCConfig 
class DataLogger:
    def __init__(self):
        self.history = {
            'timesteps': [],
            'planned_angles': [],
            'actual_angles': [],
            'planned_com': [],
            'actual_com': [],
            'control_signals': [],
            'mpc_solutions': [],  # New field for full MPC solutions
            'reference_angles': []  # <-- Add this
        }
        
    
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

    
    def plot_results(self, u_ref, mpc_history, reference_angles):
        """Generate all plots"""
        mpc_config = MPCConfig(N=10)
        joint_limits = mpc_config.joint_limits
        kinematics = KinematicsHelper()
        reference_com = [kinematics.joints_to_com(joint) for joint in reference_angles]
        # Convert reference trajectory
        u_ref_deg = np.rad2deg(u_ref.T)  # Shape (N, 8)
        
        # Convert logged controls (use control_signals instead of mpc_solutions)
        controls = np.array(mpc_history['control_signals'])  # (T, 8)
        
        # Plotting code
        fig, axs = plt.subplots(4, 2, figsize=(15, 10))
        axs = axs.flatten()
        joint_labels = [
            "left-back-shoulder", "left-back-knee",
            "left-front-shoulder", "left-front-knee",
            "right-back-shoulder", "right-back-knee",
            "right-front-shoulder", "right-front-knee"
        ]

        for i in range(8):
            # Plot reference (first N steps)
            axs[i].plot(u_ref_deg[:len(controls), i], 'b--', label='Reference')
            
            # Plot actual controls
            axs[i].plot(np.rad2deg(controls[:, i]), 'g-', label='Actual')
            
            axs[i].set_title(joint_labels[i])
            axs[i].set_ylabel('Angle (deg)')
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()

        # Joint names and colors
        joint_names = ['left-back-shoulder', 'left-back-knee', 
                    'left-front-shoulder', 'left-front-knee',
                    'right-back-shoulder', 'right-back-knee',
                    'right-front-shoulder', 'right-front-knee']
        colors = plt.cm.viridis(np.linspace(0, 1, 8))
        com_labels = ['X', 'Z', 'Yaw']

        # Joint Angles with Limits - Subplots
        fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()

        for j in range(8):
            ax = axes[j]

            # Get actual and planned for joint j
            planned = np.rad2deg([angles[j] for angles in mpc_history['planned_angles']])
            actual = np.rad2deg([angles[j] for angles in mpc_history['actual_angles']])

            # Try to get reference (safely handles shorter reference arrays)
            if len(reference_angles) >= len(mpc_history['timesteps']):
                ref = np.rad2deg([angles[j] for angles in reference_angles[:len(mpc_history['timesteps'])]])
            else:
                ref = np.rad2deg([angles[j] for angles in reference_angles])
                last_val = ref[-1]
                ref = list(ref) + [last_val] * (len(mpc_history['timesteps']) - len(ref))  # pad with last value
            
            # Now plot all three
            ax.plot(mpc_history['timesteps'], ref, color='blue', linestyle=':', label='Reference')
            ax.plot(mpc_history['timesteps'], planned, color=colors[j], linestyle='-', label='Planned')
            ax.plot(mpc_history['timesteps'], actual, color=colors[j], linestyle='--', label='Actual')

            ax.set_ylabel("Angle (deg)")
            ax.set_title(joint_names[j])
            ax.grid(True)
            ax.legend()

        # Pad reference_com to match timesteps
        T = len(mpc_history['timesteps'])
        if len(reference_com) < T:
            last_val = reference_com[-1]
            reference_com += [last_val] * (T - len(reference_com))
        else:
            reference_com = reference_com[:T]

        # Now plot CoM
        plt.figure(figsize=(10, 6))
        for i in range(3):
            ref = [com[i] for com in reference_com]
            planned = [com[i] for com in mpc_history['planned_com']]
            actual = [com[i] for com in mpc_history['actual_com']]
            
            plt.plot(mpc_history['timesteps'], ref, linestyle=':', label=f'Reference {com_labels[i]}')
            plt.plot(mpc_history['timesteps'], planned, linestyle='-', label=f'Planned {com_labels[i]}')
            plt.plot(mpc_history['timesteps'], actual, linestyle='--', label=f'Actual {com_labels[i]}')

        plt.title('Center of Mass Trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)/Angle (rad)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('com_trajectory.png')
        plt.show()

        # 3. Control Signals
        plt.figure(figsize=(12, 6))
        for j in range(8):
            plt.plot(
                mpc_history['timesteps'],
                [ctrl[j] for ctrl in mpc_history['control_signals']],
                color=colors[j], label=f'{joint_names[j]} torque'
            )
        plt.title('Control Signals')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('control_signals.png')
        plt.show()