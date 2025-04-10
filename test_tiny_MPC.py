import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import os
from tinyMPC import MPC  # Requires: pip install tinyMPC

from PID_Controller import PID_Controller
from skills import bk
from petoi_kinematics import PetoiKinematics

# Initialize components
petoi = PetoiKinematics(render_mode=None)
pid_controller = PID_Controller("urdf/bittle.xml")

# MPC Parameters
N = 10          # Prediction horizon
mpc_dt = 0.1    # MPC timestep (100ms)
sim_dt = 0.001  # Simulation timestep (1ms)
steps_per_mpc_step = int(mpc_dt / sim_dt)  # 100 steps

# State and control dimensions
nx = 6  # [c_x, c_z, θ, ẋ, ż, ω]
nu = 8  # Joint angles [α1, β1, ..., α4, β4]

# Initialize TinyMPC
mpc = MPC(N, nx, nu)

# Set weights
mpc.Q = np.diag([1e3, 1e3, 1e2, 1e2, 1e2, 1e2])  # State weights
mpc.R = np.diag([1e-2]*8)                         # Control weights

# Joint angle limits (in radians)
joint_limits = np.deg2rad([
    [-90, 70], [-70, 85],  # Left-back
    [-90, 70], [-70, 85],  # Left-front
    [-90, 70], [-70, 85],  # Right-back
    [-90, 70], [-70, 85]   # Right-front
])
mpc.umin = joint_limits[:, 0]
mpc.umax = joint_limits[:, 1]

# Dynamics function for TinyMPC
def dynamics(x, u):
    theta = x[2]
    cg, sg = np.cos(theta), np.sin(theta)
    
    foot_positions = []
    for i in range(4):
        alpha = u[i*2]
        beta = u[i*2+1]
        L = petoi.leg_length + petoi.foot_length * np.sin(beta)
        x_foot = L * np.sin(alpha)
        z_foot = -L * np.cos(alpha)
        
        if i in [0, 3]:
            foot_pos = petoi.T01_front @ np.array([x_foot, z_foot, 1])
        else:
            foot_pos = petoi.T01_back @ np.array([x_foot, z_foot, 1])
        foot_positions.append(foot_pos[:2])

    m = 1.0  # Mass
    g = 9.81
    sum_f = np.zeros(2)
    for fp in foot_positions:
        f_x = -10.0 * (fp[0] - x[0]) - 1.0 * x[3]
        f_z = -10.0 * (fp[1] - x[1]) - 1.0 * x[4] + m*g
        sum_f += np.array([f_x, f_z])

    dxdt = np.array([
        x[3], x[4], x[5],
        sum_f[0]/m, sum_f[1]/m, 0.0
    ])
    return x + dxdt * mpc_dt

mpc.f = dynamics

# Convert 'bk' skill to reference trajectory
bk_ref = np.deg2rad(np.array(bk))
reference_traj = bk_ref[:, [3,7,0,4,2,6,1,5]]  # Reorder to match actuator mapping
if len(reference_traj) < N:
    reference_traj = np.tile(reference_traj, (N//len(reference_traj)+1, 1))[:N]

def joint_angles_to_com(joint_angles):
    xz_positions = []
    for i in range(4):
        alpha = joint_angles[i*2]
        beta = joint_angles[i*2+1]
        L = petoi.leg_length + petoi.foot_length * np.sin(beta)
        x = L * np.sin(alpha)
        z = -L * np.cos(alpha)
        xz_positions.append([x, z])
        
    com_x = np.mean([p[0] for p in xz_positions])
    com_z = np.mean([p[1] for p in xz_positions])
    return np.array([com_x, com_z, 0, 0, 0, 0])

# Data logging
mpc_history = {
    'timesteps': [],
    'planned_angles': [],
    'actual_angles': [],
    'planned_com': [],
    'actual_com': [],
    'control_signals': []
}

with mujoco.viewer.launch_passive(pid_controller.model, pid_controller.data) as viewer:
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)

    state = np.zeros(6)
    actuator_nums = [3,7,0,4,2,6,1,5]
    
    for step in range(1000):
        current_time = step * sim_dt
        
        if step % steps_per_mpc_step == 0:  # Run MPC at the specified rate
            # Get current state
            orientation, gyro = pid_controller.get_imu_readings()
            state = np.array([
                pid_controller.data.qpos[0],
                pid_controller.data.qpos[1],
                orientation[2],
                pid_controller.data.qvel[0],
                pid_controller.data.qvel[1],
                gyro[2]
            ])
            
            # Get reference window
            ref_idx = (step // steps_per_mpc_step) % len(reference_traj)
            ref_window = reference_traj[ref_idx:ref_idx+N]
            if len(ref_window) < N:
                ref_window = np.vstack([ref_window, reference_traj[:N-len(ref_window)]])
            
            # Set references
            state_ref = np.array([joint_angles_to_com(ref_window[k % len(ref_window)]) for k in range(N+1)]).T
            ctrl_ref = ref_window[:N, :8].T
            
            # Solve MPC
            mpc.x_ref = state_ref
            mpc.u_ref = ctrl_ref
            mpc.x0 = state
            u_opt = mpc.solve()
            
            # Log data
            mpc_history['timesteps'].append(current_time)
            mpc_history['planned_angles'].append(u_opt[:,0].copy())
            mpc_history['planned_com'].append(mpc.X_pred[:,1])
            
        # Apply control (using first step of MPC solution)
        behavior = np.rad2deg(u_opt[:,0]).reshape(1, -1)
        pid_controller.execute(behavior, 1, sim_dt, 1e5, 50, 100, viewer=viewer, plotty=False)
        
        # Log actual state
        mpc_history['actual_angles'].append(np.array([pid_controller.data.qpos[i] for i in actuator_nums]))
        mpc_history['actual_com'].append(np.array([
            pid_controller.data.qpos[0],
            pid_controller.data.qpos[1],
            orientation[2]
        ]))
        mpc_history['control_signals'].append(np.array([pid_controller.data.ctrl[i] for i in range(8)]))

# Plotting (same as before)
plt.figure(figsize=(15, 10))

# 1. Joint Angle Comparison
plt.subplot(3, 1, 1)
colors = plt.cm.viridis(np.linspace(0, 1, 8))
for j in range(8):
    plt.plot(
        mpc_history['timesteps'],
        np.rad2deg([angles[j] for angles in mpc_history['planned_angles']]),
        color=colors[j], linestyle='-', label=f'Planned {j}'
    )
    plt.plot(
        mpc_history['timesteps'],
        np.rad2deg([angles[j] for angles in mpc_history['actual_angles']]),
        color=colors[j], linestyle='--', label=f'Actual {j}'
    )
plt.title('Joint Angles: Planned vs Actual (deg)')
plt.ylabel('Angle (deg)')
plt.legend(ncol=2)
plt.grid(True)

# 2. CoM Trajectory
plt.subplot(3, 1, 2)
com_labels = ['X', 'Z', 'Yaw']
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
plt.ylabel('Position (m)/Angle (rad)')
plt.legend()
plt.grid(True)

# 3. Control Signals
plt.subplot(3, 1, 3)
for j in range(8):
    plt.plot(
        mpc_history['timesteps'],
        [ctrl[j] for ctrl in mpc_history['control_signals']],
        color=colors[j], label=f'Joint {j}'
    )
plt.title('Actual Control Signals')
plt.xlabel('Time (s)')
plt.ylabel('Control Effort')
plt.legend(ncol=2)
plt.grid(True)

plt.tight_layout()
plt.savefig('mpc_performance.png')
plt.show()

np.savez('mpc_history.npz', **mpc_history)
print("Simulation data saved to mpc_history.npz")