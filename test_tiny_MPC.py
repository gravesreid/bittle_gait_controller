import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import os
import tinympc as MPC

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
# Your model has 7 body pos/quat + 8 joint angles = 15 qpos
# And 6 body vel/angvel + 8 joint velocities = 14 qvel
nx = 15 + 14  # 29 total states
nu = 8        # 8 joint torques

# Initialize TinyMPC
try:
    mpc = MPC.TinyMPC()
    print("TinyMPC initialized successfully")
    
    # Set parameters
    mpc.N = N
    mpc.nx = nx
    mpc.nu = nu
    
    # Weight matrices - adjust these based on your priorities
    Q = np.diag([1e3, 1e3, 1e3,       # Position (x,y,z)
                 1e2, 1e2, 1e2, 1e2,  # Orientation (quaternion)
                 1e1, 1e1, 1e1,       # Linear velocity
                 1e1, 1e1, 1e1,       # Angular velocity
                 1e2, 1e2, 1e2, 1e2,  # Joint angles (8)
                 1e2, 1e2, 1e2, 1e2,
                 1e1, 1e1, 1e1, 1e1,  # Joint velocities (8)
                 1e1, 1e1, 1e1, 1e1])
                 
    R = np.diag([1e-1]*8)  # Control effort
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
    
    mpc.u_min = -10 * np.ones(nu)  # Torque limits
    mpc.u_max = 10 * np.ones(nu)
    
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


except Exception as e:
    print(f"Failed to initialize/setup MPC: {e}")
    raise

# Convert 'bk' skill to reference trajectory
bk_ref = np.deg2rad(np.array(bk))
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
try:
    with mujoco.viewer.launch_passive(pid_controller.model, pid_controller.data) as viewer:
        if os.name == 'nt':
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            ctypes.windll.user32.ShowWindow(hwnd, 3)

        state = np.zeros(nx)
        actuator_nums = [3,7,0,4,2,6,1,5]  # Mapping to joint order in XML
        
        for step in range(int(1e8)):
            current_time = step * sim_dt
            
            if step % steps_per_mpc_step == 0:  # Run MPC at the specified rate
                # Get current state from simulation
                state[:15] = pid_controller.data.qpos[:]  # pos/quat + joint angles
                state[15:] = pid_controller.data.qvel[:]  # vel/angvel + joint velocities
                
                # Get reference window
                ref_idx = (step // steps_per_mpc_step) % len(reference_traj)
                ref_window = reference_traj[ref_idx:ref_idx+N]
                if len(ref_window) < N:
                    ref_window = np.vstack([ref_window, reference_traj[:N-len(ref_window)]])
                
                # Create reference state trajectory (focus on joint angles)
                x_ref = np.zeros((nx, N))
                for k in range(N):
                    # Set joint angle references (positions 7-14 in qpos)
                    x_ref[7:15, k] = ref_window[k % len(ref_window), :] 
                
                # Create reference control trajectory
                u_ref = np.zeros((nu, N-1))
                for k in range(N-1):
                    u_ref[:, k] = ref_window[k % len(ref_window), :]
                
                # Compute Jacobians numerically for current state
                eps = 1e-6
                A = np.zeros((nx, nx))
                B = np.zeros((nx, nu))
                
                # Current control (zero for linearization)
                u0 = np.zeros(nu)

                A, B = linearize_dynamics(
                    lambda x, u: mujoco_dynamics(pid_controller, x, u),
                    state,
                    u0,
                    nx,
                    nu
                )

                
                # Setup MPC with current linearization
                if step == 0:
                    mpc.setup(A, B, Q, R, N)
                else:
                    mpc.A = A
                    mpc.B = B
                    mpc.R = R
                    mpc.Q = Q
                # Add joint angle constraints
                x_min = -np.inf * np.ones(nx)
                x_max = np.inf * np.ones(nx)
                x_min[7:15] = joint_limits[:, 0]
                x_max[7:15] = joint_limits[:, 1]

                mpc.x_min = x_min
                mpc.x_max = x_max

                # Solve MPC
                mpc.set_x_ref(x_ref)
                mpc.set_u_ref(u_ref)
                mpc.set_x0(state)
                u_opt = mpc.solve()
                kp = 1e5
                ki = 50
                kd = 100
                dt = 0.001
                print(u_opt["controls"])
                u_opt = u_opt["controls"].reshape(1, -1)
                time.sleep(10000000)
                # Now it's safe to convert to degrees if needed
                behavior = np.rad2deg(u_opt)
                pid_controller.execute(
                    behavior=u_opt,
                    num_timesteps=len(u_opt),
                    viewer=viewer,
                    dt=sim_dt,
                    kp=kp,
                    ki=ki,
                    kd=kd,
                    plotty=False
                )
                
                # # Log data
                # mpc_history['timesteps'].append(current_time)
                # mpc_history['planned_angles'].append(u_opt[:,0].copy())
                # mpc_history['planned_com'].append(state[:3])  # Log COM position
                
            # Apply control (using first step of MPC solution)
            #pid_controller.data.ctrl[:] = u_opt[:,0]
            
            # Step simulation
            #mujoco.mj_step(pid_controller.model, pid_controller.data)
            
            # # Log actual state
            # # To:
            # mpc_history['actual_angles'].append(pid_controller.data.qpos[7:15].copy())  # or .tolist()
            # mpc_history['actual_com'].append(pid_controller.data.qpos[:3].copy())  # or .tolist()
            # mpc_history['control_signals'].append(pid_controller.data.ctrl.copy())

except Exception as e:
    print(f"Simulation error: {e}")

# # Plotting
# plt.figure(figsize=(15, 10))

# # 1. Joint Angle Comparison
# plt.subplot(3, 1, 1)
# joint_names = ['left-back-shoulder', 'left-back-knee', 
#                'left-front-shoulder', 'left-front-knee',
#                'right-back-shoulder', 'right-back-knee',
#                'right-front-shoulder', 'right-front-knee']
# colors = plt.cm.viridis(np.linspace(0, 1, 8))
# for j in range(8):
#     plt.plot(
#         mpc_history['timesteps'],
#         np.rad2deg([angles[j] for angles in mpc_history['planned_angles']]),
#         color=colors[j], linestyle='-', label=f'Planned {joint_names[j]}'
#     )
#     plt.plot(
#         mpc_history['timesteps'],
#         np.rad2deg([angles[j] for angles in mpc_history['actual_angles']]),
#         color=colors[j], linestyle='--', label=f'Actual {joint_names[j]}'
#     )
# plt.title('Joint Angles: Planned vs Actual (deg)')
# plt.ylabel('Angle (deg)')
# plt.legend(ncol=2)
# plt.grid(True)

# # 2. CoM Trajectory
# plt.subplot(3, 1, 2)
# com_labels = ['X', 'Z', 'Yaw']
# for i in range(3):
#     plt.plot(
#         mpc_history['timesteps'],
#         [com[i] for com in mpc_history['planned_com']],
#         label=f'Planned {com_labels[i]}'
#     )
#     plt.plot(
#         mpc_history['timesteps'],
#         [com[i] for com in mpc_history['actual_com']],
#         linestyle='--', label=f'Actual {com_labels[i]}'
#     )
# plt.title('Center of Mass Trajectory')
# plt.ylabel('Position (m)/Angle (rad)')
# plt.legend()
# plt.grid(True)

# # 3. Control Signals
# plt.subplot(3, 1, 3)
# for j in range(8):  # Plot all 8 joints
#     plt.plot(
#         mpc_history['timesteps'],
#         [ctrl[j] for ctrl in mpc_history['control_signals']],
#         color=colors[j], label=f'{joint_names[j]} torque'
#     )
# plt.title('Control Signals')
# plt.xlabel('Time (s)')
# plt.ylabel('Torque (Nm)')
# plt.legend(ncol=2)
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('mpc_performance.png')
# plt.show()

# np.savez('mpc_history.npz', **mpc_history)
# print("Simulation data saved to mpc_history.npz")