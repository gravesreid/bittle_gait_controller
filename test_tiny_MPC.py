
## Change MPC formulation to use torques, change torque to theta with fixed KP
## Refer to: https://chatgpt.com/c/67f808c6-9bd0-8008-a2e0-ce2bf9a0b5ff

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import os
import tinympc as MPC
import casadi as cs
from PID_Controller import PID_Controller
from skills import bk, balance, wkf
from petoi_kinematics import PetoiKinematics

# Initialize components
petoi = PetoiKinematics(render_mode=None)
pid_controller = PID_Controller("urdf/bittle.xml")

# MPC Parameters
N = 10        # Prediction horizon
mpc_dt = 0.1    # MPC timestep (100ms)
sim_dt = 0.001  # Simulation timestep (1ms)
steps_per_mpc_step = int(mpc_dt / sim_dt)  # 100 steps

# State and control dimensions
# Your model has 7 body pos/quat + 8 joint angles = 15 qpos
# And 6 body vel/angvel + 8 joint velocities = 14 qvel
nx = 6  # 29 total states
nu = 8        # 8 joint torques
def joint_angles_to_com(joint_angles_rad):
    """
    Uses joint angles to estimate the CoM via PetoiKinematics and validates with leg_ik.
    Returns:
        np.array([x, z, yaw]) - estimated center of mass state
    """
    # Split into shoulder and knee angles
    alphas = joint_angles_rad[[0, 2, 4, 6]]
    betas  = joint_angles_rad[[1, 3, 5, 7]]

    # Update kinematic model
    petoi.alphas = alphas
    petoi.betas = betas
    petoi.update_gamma_h()

    # Get estimated foot positions using current joint angles (forward kinematics)
    foot_positions = np.zeros((4, 2))
    for i in range(4):
        T = petoi.T01_front if i in [0, 3] else petoi.T01_back
        alpha = alphas[i]
        beta = betas[i]
        L = petoi.leg_length + petoi.foot_length * np.sin(beta)
        x = L * np.sin(alpha)
        z = -L * np.cos(alpha)
        foot_world = T @ np.array([x, z, 1])
        foot_positions[i] = foot_world[:2]

    # Use inverse kinematics to get back the joint angles from foot positions (optional validation)
    alphas_ik, betas_ik = petoi.leg_ik(foot_positions)

    # (Optional: Check closeness to original angles if needed for debug)
    # print("IK alpha error:", np.rad2deg(alphas - alphas_ik))
    # print("IK beta error:", np.rad2deg(betas - betas_ik))

    # Compute CoM x, z from midpoint of front and back frames
    x = (petoi.T01_front[0, 2] + petoi.T01_back[0, 2]) / 2
    z = (petoi.T01_front[1, 2] + petoi.T01_back[1, 2]) / 2
    yaw = petoi.gamma[0] if petoi.gamma.size > 0 else 0.0

    return np.array([x, z, yaw])
# Initialize TinyMPC

mpc = MPC.TinyMPC()
print("TinyMPC initialized successfully")

# Set parameters
mpc.N = N
mpc.nx = nx
mpc.nu = nu

Q = np.diag([1e4, 1e8, 1e1,   # [x, z, yaw]
             1e3, 1e3, 1e2])  # [dx, dz, dyaw]
                

# Simple knob: tracking_weight scales how much MPC tries to match joint angles
tracking_weight = 1e-8  # try 1e3 or 1e4 if needed

shoulder_weight = 1.0
knee_weight = 0.01  # knees are more influential — you can upweight them

weights = np.array([
    tracking_weight * shoulder_weight,  # left-back-shoulder
    tracking_weight * knee_weight,      # left-back-knee
    tracking_weight * shoulder_weight,  # left-front-shoulder
    tracking_weight * knee_weight,      # left-front-knee
    tracking_weight * shoulder_weight,  # right-back-shoulder
    tracking_weight * knee_weight,      # right-back-knee
    tracking_weight * shoulder_weight,  # right-front-shoulder
    tracking_weight * knee_weight       # right-front-knee
])

R = np.diag(weights)
mpc.R = R

Qf = 10 * Q            # Terminal cost

mpc.Q = Q
mpc.R = R


# Joint angle limits (from XML)
joint_limits = np.array([
    [-1.5708, 1.22173],   # left-back-shoulder
    [-1.22173, 1.48353],  # left-back-knee
    [-1.5708, 1.22173],   # left-front-shoulder
    [-1.22173, 1.48353],  # left-front-knee
    [-1.5708, 1.22173],   # right-back-shoulder
    [-1.22173, 1.48353],  # right-back-knee
    [-1.5708, 1.22173],   # right-front-shoulder
    [-1.22173, 1.48353]   # right-front-knee
])

# mpc.u_min = -10 * np.ones(nu)  # Torque limits
# mpc.u_max = 10 * np.ones(nu)

def mujoco_dynamics(pid_controller, x, u, sim_dt=0.001):
    """
    Computes dx = f(x, u) using MuJoCo simulation.

    Args:
        pid_controller: An instance with .model and .data already initialized.
        x: Full state vector (29,)
        u: Control input (8,)
        sim_dt: Simulation timestep

    Returns:
        dx: State derivative (29,) = [qvel, qacc]
    """
    qpos = x[:15]
    qvel = x[15:]

    mujoco.mj_resetData(pid_controller.model, pid_controller.data)
    pid_controller.data.qpos[:] = qpos
    pid_controller.data.qvel[:] = qvel
    pid_controller.data.ctrl[:] = u

    # Compute forward dynamics
    mujoco.mj_forward(pid_controller.model, pid_controller.data)
    qacc = pid_controller.data.qacc

    # Concatenate qvel + qacc = (14 + 14 = 28), need to insert dummy for the extra qpos element
    dx = np.concatenate([qvel, qacc])

    # Insert dummy derivative at index 14 (to make 29)
    # Could also be np.insert(dx, 14, 0.0)
    dx_full = np.zeros(29)
    dx_full[:14] = qvel
    dx_full[14] = 0.0  # dummy value (e.g., for 4th quat component)
    dx_full[15:] = qacc

    return dx_full
def create_dynamics():
    petoi = PetoiKinematics()

    x = cs.MX.sym('x', 6)  # [c_x, c_z, θ, ẋ, ż, ω]
    u = cs.MX.sym('u', 8)  # Joint angles [α1, β1, ..., α4, β4]
    
    theta = x[2]
    cg, sg = cs.cos(theta), cs.sin(theta)

    foot_positions = []
    for i in range(4):
        alpha = u[i*2]
        beta = u[i*2+1]
        
        # Basic 2D leg geometry
        L = petoi.leg_length + petoi.foot_length * cs.sin(beta)
        x_foot = L * cs.sin(alpha)
        z_foot = -L * cs.cos(alpha)

        # Apply transformation matrix (using front/back distinction)
        if i in [0, 3]:  # front legs
            foot_pos = cs.mtimes(petoi.T01_front, cs.vertcat(x_foot, z_foot, 1))
        else:  # back legs
            foot_pos = cs.mtimes(petoi.T01_back, cs.vertcat(x_foot, z_foot, 1))
        
        foot_positions.append(foot_pos[:2])
    # Dynamics: CoM + yaw
    total_torque = 0.0
    com_pos = x[0:2]  # [x, z]
    # Ground reaction force model
    m = 1.0  # mass
    g = 9.81
    sum_f = cs.MX.zeros(2)
    for fp in foot_positions:
        lever_arm = fp - com_pos
        force = cs.vertcat(
            -10.0 * (fp[0] - x[0]) - 1.0 * x[3],     # fx
            -10.0 * (fp[1] - x[1]) - 1.0 * x[4] + m*g  # fz
        )
        # Torque about the yaw axis (cross product in 2D: r × F = rx*fz - rz*fx)
        torque = lever_arm[0] * force[1] - lever_arm[1] * force[0]
        total_torque += torque

    I = 0.05  # Moment of inertia about yaw axis (tune this)
    yaw_acc = total_torque / I
    # Dynamics: CoM + yaw
    dxdt = cs.vertcat(
        x[3], x[4], x[5],       # positions → velocities
        sum_f[0]/m,             # linear x acceleration
        sum_f[1]/m,             # linear z acceleration
        yaw_acc                     # simple model: yaw acceleration = 0
    )

    return cs.Function('f', [x, u], [dxdt])

# Convert 'bk' skill to reference trajectory
bk_ref = np.deg2rad(np.array(balance))
reference_traj = bk_ref[:, [3,7,0,4,2,6,1,5]]  # Reorder to match actuator mapping
if len(reference_traj) < N:
    reference_traj = np.tile(reference_traj, (N//len(reference_traj)+1, 1))[:N]

# Data logging
mpc_history = {
    'timesteps': [],
    'planned_angles': [],
    'actual_angles': [],
    'planned_com': [],
    'actual_com': [],
    'control_signals': []
}

# Initialize u_opt before the loop
u_opt = np.zeros((nu, N))
def linearize_dynamics(dynamics_func, x0, u0, nx, nu, eps=1e-5):
    """
    Linearize the dynamics function around (x0, u0) using finite differences.
    Returns A and B matrices for linearized dynamics: dx = A(x - x0) + B(u - u0)
    """
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    f0 = dynamics_func(x0, u0)

    # Compute A = df/dx
    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        f_plus = dynamics_func(x0 + dx, u0)
        A[:, i] = (f_plus - f0) / eps

    # Compute B = df/du
    for i in range(nu):
        du = np.zeros(nu)
        du[i] = eps
        f_plus = dynamics_func(x0, u0 + du)
        B[:, i] = (f_plus - f0) / eps

    return A, B
def linearize_symbolic_dynamics(f, nx=6, nu=8):
    x = cs.MX.sym("x", nx)
    u = cs.MX.sym("u", nu)

    # Evaluate dynamics symbolically
    dx = f(x, u)

    # Compute Jacobians
    A = cs.jacobian(dx, x)
    B = cs.jacobian(dx, u)

    # Turn them into CasADi functions
    A_func = cs.Function("A", [x, u], [A])
    B_func = cs.Function("B", [x, u], [B])

    return A_func, B_func

with mujoco.viewer.launch_passive(pid_controller.model, pid_controller.data) as viewer:
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)

    state = np.zeros(nx)
    actuator_nums = [3,7,0,4,2,6,1,5]  # Mapping to joint order in XML
    f = create_dynamics()
    A_func, B_func = linearize_symbolic_dynamics(f)
    for step in range(int(1e6)):
        current_time = step * sim_dt
        
        if step % steps_per_mpc_step == 0:  # Run MPC at the specified rate
            state = np.zeros(6)
            state[0] = pid_controller.data.qpos[0]  # x
            state[1] = pid_controller.data.qpos[2]  # z
            state[2] = pid_controller.data.qpos[6]  # yaw (assuming placeholder from quaternion)
            state[3] = pid_controller.data.qvel[0]  # dx
            state[4] = pid_controller.data.qvel[2]  # dz
            state[5] = pid_controller.data.qvel[5]  # dyaw
            
            # Total length of trajectory
            T_ref = len(reference_traj)

            # MPC step count
            mpc_step = step // steps_per_mpc_step

            # Rolling index into reference
            ref_start = mpc_step % T_ref  # starting index

            # Roll out horizon, wrap around if needed
            ref_window = np.array([
                reference_traj[(ref_start + k) % T_ref]
                for k in range(N)
            ])
            
            # Create reference state trajectory (focus on joint angles)
            x_ref = np.zeros((nx, N))
            state_ref = np.array([
                joint_angles_to_com(ref_window[k % len(ref_window)]) for k in range(N)
            ]).T  # shape (3, N)

            x_ref = np.zeros((6, N))
            for k in range(N):
                x_ref[0, k] = state_ref[0, k]  # x
                x_ref[1, k] = state_ref[1, k]  # z
                x_ref[2, k] = state_ref[2, k]  # yaw
                # [3,4,5] velocities assumed zero or can be estimated
            # Create reference control trajectory
            u_ref = np.zeros((nu, N-1))
            for k in range(N-1):
                u_ref[:, k] = ref_window[k % len(ref_window), :]
            


            # Evaluate around some state and control
            x0 = np.zeros(6)
            u0 = u_ref[0]
            # Grab current state
            qpos = pid_controller.data.qpos
            qvel = pid_controller.data.qvel

            u0 = qpos[7:15]   # joint angles as reference input
            # Assume: symbolic A is (6x6), B is (6x8)
            A_sym = np.array(A_func(state, u0))  # (6, 6)
            B_sym = np.array(B_func(state, u0))  # (6, 8)
          
            
            # Setup MPC with current linearization
            if step == 0:
                mpc.setup(A_sym, B_sym, Q, R, N)
            else:
                mpc.A = A_sym
                mpc.B = B_sym
                mpc.R = R
                mpc.Q = Q
            # Set joint angle limits as control bounds
            mpc.u_min = joint_limits[:, 0]  # lower bounds in radians
            mpc.u_max = joint_limits[:, 1]  # upper bounds in radians

            # mpc.x_min = x_min
            # mpc.x_max = x_max

            # Solve MPC
            mpc.set_x_ref(x_ref)
            mpc.set_u_ref(u_ref)
            mpc.set_x0(state)
            u_opt = mpc.solve()
            kp = 6e4
            ki = 5e2
            kd = 5e1
            dt = 1e-3
            
            copy = u_opt["controls"]
            u_opt = u_opt["controls"].reshape(1, -1)
            print("MPC angles being sent (degrees):")
            print(np.rad2deg(u_opt))
            print("Example wkf step:", np.rad2deg(u_ref[0]))
            print("Example u_opt step (degrees):", np.rad2deg(u_opt[:8]))
            # Now it's safe to convert to degrees if needed
            behavior = np.rad2deg(u_opt)
           
            pid_controller.execute(
                behavior=behavior,
                num_timesteps=100,
                viewer=viewer,
                dt=dt,
                kp=kp,
                ki=ki,
                kd=kd,
                plotty=False
            )
            
            # # # Log data
            # mpc_history['timesteps'].append(current_time)
            # mpc_history['planned_angles'].append(u_opt[:,0].copy())
            # mpc_history['planned_com'].append(state[:3])  # Log COM position
            if step % steps_per_mpc_step == 0:
                ...
                mpc_history['timesteps'].append(current_time)
                mpc_history['planned_angles'].append(copy[:8])
                mpc_history['planned_com'].append(state[:3])  # COM
                mpc_history['actual_angles'].append(pid_controller.data.qpos[7:15].copy())
                mpc_history['actual_com'].append(pid_controller.data.qpos[:3].copy())
                mpc_history['control_signals'].append(pid_controller.data.ctrl.copy())
            
        # Apply control (using first step of MPC solution)
        #pid_controller.data.ctrl[:] = u_opt[:,0]
        
        # Step simulation
        #mujoco.mj_step(pid_controller.model, pid_controller.data)
        
        # # Log actual state
        # # To:
        # mpc_history['actual_angles'].append(pid_controller.data.qpos[7:15].copy())  # or .tolist()
        # mpc_history['actual_com'].append(pid_controller.data.qpos[:3].copy())  # or .tolist()
        # mpc_history['control_signals'].append(pid_controller.data.ctrl.copy())

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
print(T_ref)
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