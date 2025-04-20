import numpy as np
import mujoco
import casadi as cs
from helpers.util import Util
from helpers.mpc_config import MPCConfig
from helpers.kinematics import KinematicsHelper
from helpers.PID_Controller import PID_Controller
from helpers.data_logger import DataLogger
from helpers.skills import bk, wkf, balance
from scipy.spatial.transform import Rotation
from helpers.petoi_kinematics import PetoiKinematics
def main():
    petoi = PetoiKinematics()
    # Open log file
    with open('simulation_log.txt', 'w') as log_file:
        def log(message):
            """Helper function to write to both console and log file"""
            print(message)
            log_file.write(message + '\n')
        
        log("=== Starting main() ===")
        # Simulation parameters
        log("Setting simulation parameters...")
        mpc_dt = 0.1     # 100ms MPC timestep
        sim_dt = 0.001    # 1ms simulation timestep
        steps_per_mpc_step = int(mpc_dt / sim_dt)
        nx = 22           # State dimension
        nu = 8            # Control dimension
        num_timesteps = 1e4
        log(f"Parameters set: mpc_dt={mpc_dt}, sim_dt={sim_dt}, steps_per_mpc_step={steps_per_mpc_step}")

        # PID parameters
        kp = 1e2
        kd = 5e-1
        ki = 5e-1
        
        # Initialize components
        log("Initializing components...")
        mpc_config = MPCConfig(N=8)
        kinematics = KinematicsHelper()
        pid = PID_Controller("urdf/bittle.xml", 
                             dt=sim_dt,
                             kp=kp,
                             ki=ki,
                             kd=kd)
        logger = DataLogger()
        log("Components initialized")
        
        # Convert skill to reference trajectory
        log("Creating reference trajectory...")
        bk_ref = np.deg2rad(np.repeat(balance, 100, axis=0))  # Repeat each row 100 times
        reference_traj = bk_ref.copy()  # Reorder to match actuator mapping
        log(f"Reference trajectory shape: {reference_traj.shape}")

        with mujoco.viewer.launch_passive(pid.model, pid.data) as viewer:
            log("Mujoco viewer launched")
            state = np.zeros(nx)
            u_opt = np.zeros(nu)
            f = mpc_config.create_dynamics()
            A_func, B_func = MPCConfig.linearize_dynamics(f)

            for step in range(int(num_timesteps)):
                current_time = step * sim_dt
                
                if step % steps_per_mpc_step == 0:  # MPC update rate
                    log(f"\nMPC update step {step//steps_per_mpc_step}")
                    # Get current state
                    log("Getting current state...")
                    
                    # Base position (x,z)
                    state[0] = pid.data.qpos[0]  # x
                    state[1] = pid.data.qpos[2]  # z
                    
                    # Base orientation (yaw from quaternion)
                    quat = pid.data.qpos[3:7]  # [x,y,z,w]
                    yaw = Rotation.from_quat(quat).as_euler('zyx')[0]
                    state[2] = yaw
                    
                    # Base velocity (dx, dz, dyaw)
                    state[3:6] = [pid.data.qvel[0], pid.data.qvel[2], pid.data.qvel[5]]
                    
                    # Joint states (8 joints)
                    state[6:14] = pid.get_angles([3,7,0,4,2,6,1,5])   # Joint angles
                    state[14:22] = pid.get_velocities([3,7,0,4,2,6,1,5])   # Joint velocities
                    log(f"State updated: {state.tolist()}")
                    
                    # Compute reference trajectories
                    log("Computing reference trajectories...")
                    torque_ref = Util.compute_pd_torque(reference_traj, kp, kd, ki, sim_dt)
                    T_torque = len(torque_ref)
                    mpc_step = step // steps_per_mpc_step
                    log(f"Reference torque shape: {torque_ref.shape}")
            
                    # Create reference window
                    log("Creating reference window...")
                    ref_window = np.array([
                        torque_ref[(mpc_step + k) % T_torque]
                        for k in range(mpc_config.N)
                    ])
                    ref_angle_window = np.array([
                        reference_traj[(mpc_step + k) % len(reference_traj)]
                        for k in range(mpc_config.N)
                    ])
                    print("ref_angle_window", ref_angle_window)
                    log(f"ref_window shape: {ref_window.shape}, ref_angle_window shape: {ref_angle_window.shape}")

                    # Build reference trajectories
                    log("Building reference trajectories...")
                    x_ref = np.zeros((nx, mpc_config.N))
                    for k in range(mpc_config.N-1):
                        #com_des = kinematics.joints_to_com(ref_angle_window[k])
                        #com_des_dot = (kinematics.joints_to_com(ref_angle_window[k+1]) - com_des)/mpc_dt
                        x_ref[0:3, k] = state[0:3]
                        x_ref[1,k] = 0.09
                        x_ref[2,k] = 0
                        #log(f"com_des: {com_des.tolist()}")
                        x_ref[3:6, k] = state[3:6]
                        print(ref_angle_window[:,k])
                        x_ref[6:14, k] = ref_angle_window[:,k]
                        x_ref[14:22, k] = (ref_angle_window[k+1] - ref_angle_window[k])/mpc_dt
                    # Fill the last joint angle reference at horizon end
                    x_ref[6:14, mpc_config.N - 1] = ref_angle_window[-1]

                    # Optionally, set joint velocities at last step to zero or approximate
                    x_ref[14:22, mpc_config.N - 1] = 0  # or (ref_angle_window[-1] - ref_angle_window[-2]) / mpc_dt

                    #log(f"x_ref shape: {x_ref.shape}")
                    print("xref ", x_ref[6:14])
                    # Linearize dynamics
                    log("Linearizing dynamics...")
                    current_u = u_opt if step > 0 else ref_window[0]
                    u_ref = np.zeros((nu, mpc_config.N-1))
                    for k in range(mpc_config.N-1):
                        u_ref[:, k] = ref_window[k % len(ref_window), :]
                    grf_ref = u_ref.copy()
                    for k in range(mpc_config.N-1):
                        grf_numeric = mpc_config.joints_to_GRF_func(ref_angle_window[k], u_ref[:, k])
                        grf_ref[:, k] = np.array(grf_numeric).flatten()
                    if step == 0:
                        log("First step - using reference control")
                        current_u = grf_ref[:, 0]
                    else:
                        log(f"Using previous MPC solution: {u_opt.tolist()}")
                        current_u = grf_opt
                    
                   
                
                    # Setup MPC
                    if step == 0:
                        log("Computing A and B matrices...")
                        standing_state = state.copy()
                        standing_state[6:14] = np.deg2rad([30, 30, 30, 30, 30, 30, 30, 30])
                        A_sym = np.array(A_func(standing_state, current_u))
                        B_sym = np.array(B_func(standing_state, current_u))
                        log(f"A_sym shape: {A_sym.shape}, B_sym shape: {B_sym.shape}")
                        log("Setting up MPC...")
                        mpc_config.mpc.setup(A_sym, B_sym, mpc_config.Q, mpc_config.R, mpc_config.N)
                    
                        # Set constraints
                        log("Setting constraints...")
                        mpc_config.mpc.x_min = mpc_config.mpc.x_min
                        mpc_config.mpc.x_max = mpc_config.mpc.x_max
                        mpc_config.mpc.u_min = mpc_config.mpc.u_min 
                        mpc_config.mpc.u_max = mpc_config.mpc.u_max 

                        # Solve MPC
                        log("Solving MPC...")
                    
                    mpc_config.mpc.set_x_ref(x_ref)
                    mpc_config.mpc.set_u_ref(grf_ref)
                    
                    mpc_config.mpc.set_x0(state)
                    #mpc_config.mpc.set_x0(state)
                    mpc_solution = mpc_config.mpc.solve()
                    # In the main loop after solving MPC:
                    grf_opt = mpc_solution["controls"][:8]
                    theta_opt = mpc_solution["states_all"][:,6:14]  # First column of x
                    joint_angles = pid.get_angles([3,7,0,4,2,6,1,5])

                    # Compute Jacobians for current state
                    tau_opt = np.zeros(8)
                    for i in range(4):
                        alpha = joint_angles[2*i]
                        beta = joint_angles[2*i + 1]
                        # Get precomputed Jacobian function
                        J_func = mpc_config.leg_jacobian(cs.MX.sym('alpha'), cs.MX.sym('beta'))
                        J_current = J_func(alpha, beta)
                        # Compute torques: τ = J^T * F
                        grf = grf_opt.reshape((4, 2))
                        F_leg = grf[i, :]
                        tau_leg = cs.mtimes(J_current.T, F_leg.T)
                        tau_opt[2*i] += tau_leg[0]
                        tau_opt[2*i + 1] += tau_leg[1]


                    # Apply torque limits
                    u_opt = np.clip(tau_opt, -0.75, 0.75)
                    #log(f"MPC solution keys: {list(mpc_solution.keys())}")
                    
                    #u_opt = np.clip(mpc_solution["controls"][:8], -0.75, 0.75)
                    #log(f"u_opt after clip: {u_opt.tolist()}")
                    
                    # Store solution
                    logger.history['mpc_solutions'].append(np.array(mpc_solution["controls"][:8]))
                    #log(f"Stored mpc_solutions length: {len(logger.history['mpc_solutions'])}")

                    # Convert torque to theta_ref
                    theta_ref = state[6:14] + u_opt / kp
                    log(f"theta_ref: {x_ref[6:14,:].tolist()}")
                    log(f"theta_opt: {theta_opt.tolist()}")
                    log(f"theta_cur: {state[6:14].tolist()}")

                    # Execute control
                    log("Updating joint targets...")
                    if mpc_step <= 20:
                        pid.set_targets(target=np.rad2deg(x_ref[6:14,0]))
                    else:
                        print("Switching to MPC control")
                        pid.set_targets(target=(np.rad2deg(theta_opt)))

                    # Log data
                    log("Logging data...")
                    logger.log_data(
                        timestep=current_time,
                        planned_angles=theta_opt,
                        planned_com= x_ref[0:3,0],
                        actual_angles=pid.get_angles([3,7,0,4,2,6,1,5]),
                        actual_com= np.array([pid.data.qpos[0], pid.data.qpos[2], pid.data.qpos[6]]),
                        controls=u_opt,
                        reference_angles=ref_angle_window[0],  # Log first reference angle of the window
                        log_file = log_file
                    )
                    log("Data logged")

                # MPC Update finished
                # Simulation step
                pid.step(viewer)

            # Generate plots
            log("Generating plots...")
            logger.plot_results(
                u_ref=reference_traj,
                mpc_history=logger.history,
                reference_angles=ref_angle_window[:, :]  # pass entire N-step reference angles
            )
            log("Plots generated")


if __name__ == "__main__":
    
    main()