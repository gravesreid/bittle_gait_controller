
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
nx = 22  # [x, z, yaw, dx, dz, dyaw, q1-q8, q_vel 1-8]
nu = 8   # Joint torques
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

Q = np.diag([
    1e4, 1e8, 1e1,       # [x, z, yaw]
    1e3, 1e3, 1e2,       # [dx, dz, dyaw]
    *[1e2]*8,            # Joint angles (q1-q8)
    *[1e-1]*8            # Joint velocities (dq1-dq8)
])

                
# Simple knob: tracking_weight scales how much MPC tries to match joint angles
tracking_weight = 1e3  # try 1e3 or 1e4 if needed

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

def create_dynamics():
    """Full dynamics model matching XML parameters
    State (22 elements):
        [0:3] CoM position (x, z, yaw)
        [3:6] CoM velocity (dx, dz, dyaw)
        [6:14] Joint angles (8)
        [14:22] Joint velocities (8)
    Control (8 elements): Joint torques (limited to ±0.75 Nm)
    """
    petoi = PetoiKinematics()
    
    # State and control variables
    x = cs.MX.sym('x', 22)
    u = cs.MX.sym('u', 8)

    # XML-derived parameters
    total_mass = 0.165  # From root body inertial
    com_inertia = 0.001  # From root body diaginertia (izz)
    
    # Joint parameters from XML
    joint_inertias = np.array([
        # Shoulders (0.00044) and knees (0.00063) alternating
        0.00044, 0.00063, 0.00044, 0.00063,
        0.00044, 0.00063, 0.00044, 0.00063
    ])
    
    joint_damping = 0.01  # From <joint damping="0.01">
    torque_limit = 0.75   # From actuator forcerange

    # Unpack state variables
    com_pos = x[0:3]        # [x, z, theta]
    com_vel = x[3:6]        # [dx, dz, dtheta]
    joint_angles = x[6:14]  # q1-q8
    joint_vel = x[14:22]    # dq1-dq8

    # 1. CoM Dynamics ------------------------------------------
    foot_positions = []
    for i in range(4):
        alpha = joint_angles[i*2]
        beta = joint_angles[i*2+1]
        
        # XML-based leg geometry
        L = petoi.leg_length + petoi.foot_length * cs.sin(beta)
        x_foot = L * cs.sin(alpha)
        z_foot = -L * cs.cos(alpha)
        
        # Transform using XML-measured offsets
        if i in [0, 3]:  # front legs
            T = petoi.T01_front @ cs.vertcat(x_foot, z_foot, 1)
        else:  # back legs
            T = petoi.T01_back @ cs.vertcat(x_foot, z_foot, 1)
        foot_positions.append(T[:2])

    # Ground reaction model
    total_force = cs.MX.zeros(2)
    total_torque = 0.0
    k_ground = 1500
    c_ground = 75
    
    for fp in foot_positions:
        penetration = cs.fmax(0 - fp[1], 0)
        f_z = k_ground * penetration - c_ground * com_vel[1]
        f_x = -0.25 * f_z * cs.tanh(com_vel[0] * 15)
        lever_arm = fp - com_pos[:2]
        total_torque += lever_arm[0] * f_z - lever_arm[1] * f_x
        total_force += cs.vertcat(f_x, f_z)
    
    # CoM accelerations
    com_acc = cs.vertcat(
        total_force[0]/total_mass,
        total_force[1]/total_mass - 9.81,
        total_torque / com_inertia
    )

    # 2. Joint Dynamics ----------------------------------------
    # Element-wise operations for joint accelerations
    joint_acc = (cs.fmin(cs.fmax(u, -torque_limit), torque_limit)  # Apply torque limits
                 - joint_damping * joint_vel) / joint_inertias

    # 3. Full State Derivative ---------------------------------
    dxdt = cs.vertcat(
        com_vel[0],        # dx/dt
        com_vel[1],        # dz/dt
        com_vel[2],        # dtheta/dt
        com_acc[0],        # ddx/dt
        com_acc[1],        # ddz/dt
        com_acc[2],        # ddtheta/dt
        joint_vel,         # dq/dt
        joint_acc          # ddq/dt
    )
    
    return cs.Function('dynamics', [x, u], [dxdt], ['x', 'u'], ['dxdt'])
# Convert 'bk' skill to reference trajectory
bk_ref = np.deg2rad(np.array(bk))
reference_traj = bk_ref[:, [3,7,0,4,2,6,1,5]]  # Reorder to match actuator mapping
def compute_pd_torque(u_angle_trajectory, kp, kd, ki, pid_dt=1e-3):
    """
    Compute joint torques using PD control law with aligned dimensions.
    
    Args:
        u_angle_trajectory: (N, 8) array of target joint angles
        kp: Proportional gain (float or 8-element array)
        kd: Derivative gain (float or 8-element array)
        ki: Integral gain (float or 8-element array)
        pid_dt: Time step between trajectory points
        
    Returns:
        torque: (N-2, 8) array of computed torques
    """
    trajectory = np.array(u_angle_trajectory)
    
    # Central differences for consistent dimensions (N-2, 8)
    # Position difference (Δθ between i+1 and i-1)
    del_theta = trajectory[2:] - trajectory[:-2]
    
    # Velocity (θ_dot = Δθ/(2Δt))
    theta_dot = del_theta / (2 * pid_dt)
    
    # Integral term (cumulative sum of position errors)
    int_theta = np.cumsum(del_theta * pid_dt, axis=0)
    
    # Torque calculation (all terms now have shape (N-2, 8))
    torque = (
        kp * del_theta +
        kd * theta_dot +
        ki * int_theta
    )
    
    return torque

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
def linearize_symbolic_dynamics(f, nx=22, nu=8):
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
            state = np.zeros(22)
            state[0] = pid_controller.data.qpos[0]          # base x
            state[1] = pid_controller.data.qpos[2]          # base z
            state[2] = pid_controller.data.qpos[6]          # base yaw
            state[3:6] = pid_controller.data.qvel[[0,2,5]]  # base velocities
            state[6:14] = pid_controller.data.qpos[7:15]    # joint angles
            state[14:22] = pid_controller.data.qvel[6:14]   # joint velocities
            
             # Grab current state
            qpos = pid_controller.data.qpos
            qvel = pid_controller.data.qvel
            
            #PID settings
            kp = 1e2
            ki = 5e-1
            kd = 5e-1
            dt = 1e-3
            
            # Compute torque reference trajectory
            T_ref = len(reference_traj)
            torque_ref = compute_pd_torque(reference_traj, kp, kd, ki, dt)
            T_torque = len(torque_ref)  # Actual length of torque_ref

            # Determine starting index in torque_ref
            mpc_step = step // steps_per_mpc_step
            ref_start = mpc_step % T_torque  # Use T_torque for modulus

            # Create reference window with valid indices
            ref_window = np.array([
                torque_ref[(ref_start + k) % T_torque]  # Modulus based on T_torque
                for k in range(N)
            ])
            # Create reference window with valid indices
            ref_angle_window = np.array([
                reference_traj[(ref_start + k) % T_ref]  # Modulus based on T_torque
                for k in range(N)
            ])
            
 
            x_ref = np.zeros((nx, N))
            for k in range(N-1):
                com_des = joint_angles_to_com(ref_angle_window[k])
                com_des_dot = (joint_angles_to_com(ref_window[k+1]) -joint_angles_to_com(ref_window[k]))/dt 
                x_ref[0:3, k] = com_des          # CoM position
                x_ref[3:6, k] =  com_des_dot     # CoM velocity target
                x_ref[6:14, k] = ref_window[k]   # Joint angles
                x_ref[14:22, k] = ((
                    ref_window[k+1]-ref_window[k])/dt # joint velocity target
                    )          
           
            
            # Create reference control trajectory
            u_ref = np.zeros((nu, N-1))
            for k in range(N-1):
                u_ref[:, k] = ref_window[k % len(ref_window), :]
            # Get initial control input from reference trajectory
            if step == 0:
                # For first step, use first control input from reference
                current_u = u_ref[:, 0]
            else:
                # Use previous MPC solution's first control input
                current_u = u_opt[:, 0]

            # Compute A and B matrices at current state and control input
            A_sym = np.array(A_func(state, current_u))  # Correct: state (22), current_u (8)
            B_sym = np.array(B_func(state, current_u))  # Correct: state (22), current_u (8)
          
            
            # Setup MPC with current linearization
            if step == 0:
                mpc.setup(A_sym, B_sym, Q, R, N)
            else:
                mpc.A = A_sym
                mpc.B = B_sym
                mpc.R = R
                mpc.Q = Q
            # Set constraints
            
            #In your MPC setup section:
            # Initialize constraints
            mpc.x_min = -np.inf*np.ones(nx)
            mpc.x_max = np.inf*np.ones(nx)
            mpc.u_min = -0.75 * np.ones(8)
            mpc.u_max = 0.75 * np.ones(8)

            # Set joint angle limits for all timesteps
            for k in range(N):
                mpc.x_min[6:14] = joint_limits[:, 0]
                mpc.x_max[6:14] = joint_limits[:, 1]
            
            # Solve MPC
            mpc.set_x_ref(x_ref)
            mpc.set_u_ref(u_ref)
            mpc.set_x0(state)
            u_opt = mpc.solve()
            
            
            copy = u_opt["controls"]
            u_opt = np.clip(u_opt["controls"].reshape(1, -1), -0.75,0.75)
            
            theta_ref = state[6:14] + u_opt / kp
            print("step: ", step)
            print("torque reference:", np.rad2deg(u_ref[0]))
            print()
            print("angle reference:", np.rad2deg(ref_angle_window[0]))
            print()
            print("MPC torques: ", u_opt)
            print()
            print("MPC theta step (degrees):", np.rad2deg(theta_ref))
            print()
            # Now it's safe to convert to degrees if needed
            behavior = np.rad2deg(theta_ref)
            
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