import matplotlib.pyplot as plt
import numpy as np
from .mpc_config import MPCConfig
class DataLogger:
    def __init__(self):
        self.history = {
            'timesteps': [],
            'planned_angles': [],
            'actual_angles': [],
            'planned_com': [],
            'actual_com': [],
            'control_signals': []
        }
    
    def log_data(self, timestep, 
                planned_angles, planned_com,
                actual_angles, actual_com,
                controls):
        """Store all relevant data from one timestep"""
        self.history['timesteps'].append(timestep)
        self.history['planned_angles'].append(np.array(planned_angles))
        self.history['planned_com'].append(np.array(planned_com))
        self.history['actual_angles'].append(np.array(actual_angles))
        self.history['actual_com'].append(np.array(actual_com))
        self.history['control_signals'].append(np.array(controls))
    
    def plot_results(self, u_ref,u_opt, mpc_history):
        """Generate all plots"""
        joint_limits = MPCConfig.joint_limits
        # Convert to degrees for visualization
        u_ref_deg = np.rad2deg(u_ref.T)    # Shape (N-1, 8)
        u_opt_deg = np.rad2deg(u_opt)      # Shape (N-1, 8)

        fig, axs = plt.subplots(4, 2, figsize=(15, 10))
        axs = axs.flatten()
        joint_labels = [
            "left-back-shoulder", "left-back-knee",
            "left-front-shoulder", "left-front-knee",
            "right-back-shoulder", "right-back-knee",
            "right-front-shoulder", "right-front-knee"
        ]

        for i in range(8):
            axs[i].plot(u_ref_deg[:, i], 'b--', label='u_ref (MPC input)')
            axs[i].plot(u_opt_deg[:, i], 'g-', label='u_opt (optimal control)')
            axs[i].set_title(joint_labels[i])
            axs[i].set_ylabel('Angle (deg)')
            axs[i].set_xlabel('MPC Horizon Step')
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
            planned = np.rad2deg([angles[j] for angles in mpc_history['planned_angles']])
            actual = np.rad2deg([angles[j] for angles in mpc_history['actual_angles']])
            
            ax.plot(mpc_history['timesteps'], planned, color=colors[j], linestyle='-', label='Planned')
            ax.plot(mpc_history['timesteps'], actual, color=colors[j], linestyle='--', label='Actual')
            
            # Joint limits
            min_limit = np.rad2deg(joint_limits[j, 0])
            max_limit = np.rad2deg(joint_limits[j, 1])
            ax.axhline(min_limit, color='gray', linestyle=':', linewidth=1, alpha=0.6)
            ax.axhline(max_limit, color='gray', linestyle=':', linewidth=1, alpha=0.6)
            
            ax.set_title(joint_names[j])
            ax.set_ylabel('Angle (deg)')
            ax.grid(True)
            if j >= 6:
                ax.set_xlabel('Time (s)')
            ax.legend(loc='upper right')

        plt.suptitle('Joint Angles: Planned vs Actual with Limits', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('joint_angles_subplots.png')
        plt.show()

        # 2. Center of Mass (CoM) Trajectory
        plt.figure(figsize=(12, 5))
        for i in range(3):
            plt.plot(
                mpc_history['timesteps'],
                [com[i] for com in mpc_history['planned_com']],
                label=f'Planned {com_labels[i]}'
            )
            plt.plot(
                mpc_history['timesteps'],
                [com[i] for com in mpc_history['actual_com']],
                linestyle='--', label=f'Actual {com_labels[i]}'
            )
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