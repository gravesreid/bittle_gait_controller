import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on your setup
import os
import casadi as cs  # Requires installation: pip install casadi

from PID_Controller import PID_Controller
from skills import bk
# Fix 1: Add proper imports and initialize PetoiKinematics correctly
from petoi_kinematics import PetoiKinematics

# Initialize components
petoi = PetoiKinematics(render_mode=None)
pid_controller = PID_Controller("urdf/bittle.xml")

# MPC Parameters
N = 10  # Prediction horizon
dt = 0.05  # Timestep (50ms)
Q = cs.diag([1e3, 1e3, 1e2, 1e2, 1e2, 1e2])  # State weights
R = cs.diag([1e-2]*8)  # Control effort weights

# Create dynamics function
def create_dynamics():
    x = cs.MX.sym('x', 6)  # [c_x, c_z, θ, ẋ, ż, ω]
    u = cs.MX.sym('u', 8)  # Joint angles [α1, β1, ..., α4, β4]
    
    theta = x[2]
    cg, sg = cs.cos(theta), cs.sin(theta)
    
    foot_positions = []
    for i in range(4):
        alpha = u[i*2]
        beta = u[i*2+1]
        L = petoi.leg_length + petoi.foot_length * cs.sin(beta)
        x_foot = L * cs.sin(alpha)
        z_foot = -L * cs.cos(alpha)
        
        if i in [0, 3]:
            foot_pos = cs.mtimes(petoi.T01_front, cs.vertcat(x_foot, z_foot, 1))
        else:
            foot_pos = cs.mtimes(petoi.T01_back, cs.vertcat(x_foot, z_foot, 1))
        foot_positions.append(foot_pos[:2])

    m = 1.0  # Mass
    g = 9.81
    sum_f = cs.MX.zeros(2)
    for fp in foot_positions:
        f_x = -10.0 * (fp[0] - x[0]) - 1.0 * x[3]
        f_z = -10.0 * (fp[1] - x[1]) - 1.0 * x[4] + m*g
        sum_f += cs.vertcat(f_x, f_z)

    dxdt = cs.vertcat(
        x[3], x[4], x[5],
        sum_f[0]/m, sum_f[1]/m, 0.0
    )
    return cs.Function('f', [x, u], [dxdt])

f_continuous = create_dynamics()

# MPC Parameters
N = 10  # Prediction horizon
dt = 0.1  # Timestep

# Convert 'bk' skill to radians and pad
bk_ref = np.deg2rad(np.array(bk))
if len(bk_ref) < N+1:
    bk_ref = np.tile(bk_ref, (N//len(bk_ref)+2, 1))[:N+1]

# MPC Formulation
opti = cs.Opti()
X = opti.variable(6, N+1)  # State trajectory
U = opti.variable(8, N)    # Control trajectory
X_ref = opti.parameter(6, N+1)
U_ref = opti.parameter(8, N)

# Dynamics constraints
for k in range(N):
    opti.subject_to(X[:,k+1] == X[:,k] + f_continuous(X[:,k], U[:,k])*dt)
# Joint angle limits [min, max] for each of the 8 joints (in radians)
joint_limits = np.deg2rad([
    # Left-back-shoulder (Joint 0)
    [-90, 70],    # URDF range: -1.5708(-90°) to 1.22173(70°)
    
    # Left-back-knee (Joint 1)
    [-70, 85],    # URDF range: -1.22173(-70°) to 1.48353(85°)
    
    # Left-front-shoulder (Joint 2)
    [-90, 70],    # Same as other shoulders
    
    # Left-front-knee (Joint 3)
    [-70, 85],    # Same as other knees
    
    # Right-back-shoulder (Joint 4)
    [-90, 70],
    
    # Right-back-knee (Joint 5)
    [-70, 85],
    
    # Right-front-shoulder (Joint 6)
    [-90, 70],
    
    # Right-front-knee (Joint 7)
    [-70, 85]
]).T  # Transposed for easier column access

# Add joint angle constraints
for k in range(N):
    for j in range(8):
        opti.subject_to(U[j,k] >= joint_limits[0,j])  # Minimum angle
        opti.subject_to(U[j,k] <= joint_limits[1,j])  # Maximum angle

# Cost function
cost = 0
for k in range(N):
    cost += cs.sumsqr(Q @ (X[:,k] - X_ref[:,k])) 
    cost += cs.sumsqr(R @ (U[:,k] - U_ref[:,k]))
opti.minimize(cost)

# Solver setup
opts = {'ipopt.print_level': 0,'print_time': False, 'error_on_fail': False,'ipopt.sb': 'yes'}
opti.solver('ipopt', opts)  # Use IPOPT for nonlinear MPC

# Convert bk skill to reference trajectory
bk_ref = np.deg2rad(np.array(bk))
reference_traj = bk_ref[:, [3,7,0,4,2,6,1,5]]  # Reorder to match actuator mapping
# Ensure reference trajectory is at least N steps long by repeating if needed
if len(reference_traj) < N:
    reference_traj = np.tile(reference_traj, (N//len(reference_traj)+1, 1))[:N]
# Main control loop
def joint_angles_to_com(joint_angles):
    """Convert joint angles to CoM position using forward kinematics"""
    # Use petoi's leg_ik in reverse
    xz_positions = []
    for i in range(4):
        alpha = joint_angles[i*2]
        beta = joint_angles[i*2+1]
        L = petoi.leg_length + petoi.foot_length * np.sin(beta)
        x = L * np.sin(alpha)
        z = -L * np.cos(alpha)
        xz_positions.append([x, z])
        
    # Average foot positions for CoM estimate
    com_x = np.mean([p[0] for p in xz_positions])
    com_z = np.mean([p[1] for p in xz_positions])
    return np.array([com_x, com_z, 0, 0, 0, 0])  # Simplified orientation/velocities
# Initialize plotting containers (before the main loop)
mpc_history = {
    'planned_angles': [],       # MPC's planned joint angles
    'actual_angles': [],        # Actually applied joint angles
    'planned_com': [],          # MPC's planned CoM trajectory
    'actual_com': [],           # Actual CoM position
    'timesteps': [],             # Time points
    'control_signals': []
}



with mujoco.viewer.launch_passive(pid_controller.model, pid_controller.data) as viewer:
    # Windows fullscreen
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)

    state = np.zeros(6)
    
    for _ in range(1000):
        current_time = _ * dt
        # Pad reference trajectory if needed
        if len(reference_traj) < N:
            reference_traj = np.tile(reference_traj, (N//len(reference_traj)+2, 1))[:N]
        orientation, gyro = pid_controller.get_imu_readings()
        state = np.array([
            pid_controller.data.qpos[0],  # c_x
            pid_controller.data.qpos[1],  # c_z
            orientation[2],               # θ
            pid_controller.data.qvel[0],  # ẋ
            pid_controller.data.qvel[1],  # ż
            gyro[2]                      # ω
        ])
        
        # Get reference window - ensure it's N steps long
        ref_window = reference_traj[_%len(reference_traj):_%len(reference_traj)+N]
        if len(ref_window) < N:
            # If we're at the end of the trajectory, wrap around
            remaining = N - len(ref_window)
            ref_window = np.vstack((ref_window, reference_traj[:remaining]))

        # Convert joint angle references to state references
        state_ref = np.array([joint_angles_to_com(ref_window[k % len(ref_window)]) for k in range(N+1)]).T
        ctrl_ref = ref_window[:N, :8].T  # Shape (8, N)

        # Set references
        opti.set_value(X_ref, state_ref)
        opti.set_value(U_ref, ctrl_ref)
        
        # Solve MPC
        sol = opti.solve()
        u_opt = sol.value(U[:,:])
        
        # Apply control
        actuator_nums = [3,7,0,4,2,6,1,5]
        kp = 1e5
        ki = 50
        kd = 100
        dt = 0.001
        # Create behavior from u_opt
        behavior = np.rad2deg(u_opt).reshape(1, -1)  # Convert to degrees and make 2D
        pid_controller.execute(behavior, len(u_opt), dt, kp, ki, kd, viewer=viewer, plotty=False)
        # After getting u_opt from MPC:
        # After getting u_opt from MPC:
        mpc_history['timesteps'].append(current_time)
        mpc_history['planned_angles'].append(u_opt.copy())
        mpc_history['planned_com'].append(sol.value(X[:,1])) 
        
        # Store actual states
        mpc_history['actual_angles'].append(
            np.array([pid_controller.data.qpos[i] for i in actuator_nums])
        )
        mpc_history['actual_com'].append(np.array([
            pid_controller.data.qpos[0],  # c_x
            pid_controller.data.qpos[1],  # c_z
            orientation[2]                # θ
        ]))
        mpc_history['control_signals'].append(
            np.array([pid_controller.data.ctrl[i] for i in range(8)])
        )
        # mujoco.mj_step(pid_controller.model, pid_controller.data)
        # viewer.sync()
# Post-simulation analysis plots
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
plt.savefig('mpc_performance.png')  # Save to file
plt.show()

# Save all data for further analysis
np.savez('mpc_history.npz', **mpc_history)
print("Simulation data saved to mpc_history.npz")