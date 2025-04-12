import numpy as np
import mujoco
from helpers.util import Util
from helpers.mpc_config import MPCConfig
from helpers.kinematics import KinematicsHelper
from helpers.PID_Controller import PID_Controller
from helpers.data_logger import DataLogger
from helpers.skills import bk, wkf, balance

def main():
    # Initialize components
    mpc_config = MPCConfig(N=10)
    kinematics = KinematicsHelper()
    pid = PID_Controller("urdf/bittle.xml")
    logger = DataLogger()

    # Simulation parameters
    mpc_dt = 0.1     # 100ms MPC timestep
    sim_dt = 0.001    # 1ms simulation timestep
    steps_per_mpc_step = int(mpc_dt / sim_dt)
    nx = 22           # State dimension
    nu = 8            # Control dimension
    num_timesteps = 1e6
    
    # Convert skill to reference trajectory
    bk_ref = np.deg2rad(np.array(bk))
    reference_traj = bk_ref[:, [3,7,0,4,2,6,1,5]]  # Reorder to match actuator mapping

    with mujoco.viewer.launch_passive(pid.model, pid.data) as viewer:
        state = np.zeros(nx)
        u_opt = np.zeros(nu)
        f = MPCConfig.create_dynamics(mpc_config)
        A_func, B_func = MPCConfig.linearize_dynamics(f)
        for step in range(int(num_timesteps)):  # Run indefinitely until viewer closes
            current_time = step * sim_dt
            
            if step % steps_per_mpc_step == 0:  # MPC update rate
                # Get current state
                state[0] = pid.data.qpos[0]          # base x
                state[1] = pid.data.qpos[2]          # base z
                state[2] = pid.data.qpos[6]          # base yaw
                state[3:6] = pid.data.qvel[[0,2,5]]  # base velocities
                state[6:14] = pid.data.qpos[7:15]    # joint angles
                state[14:22] = pid.data.qvel[6:14]   # joint velocities

                # Compute reference trajectories
                kp = 1e2
                kd = 5e-1
                ki = 5e-1
                torque_ref = Util.compute_pd_torque(reference_traj, kp, kd, ki, sim_dt)
                T_torque = len(torque_ref)
                mpc_step = step // steps_per_mpc_step

                # Create reference window
                ref_window = np.array([
                    torque_ref[(mpc_step + k) % T_torque]
                    for k in range(mpc_config.N)
                ])
                ref_angle_window = np.array([
                    reference_traj[(mpc_step + k) % len(reference_traj)]
                    for k in range(mpc_config.N)
                ])

                # Build reference trajectories
                x_ref = np.zeros((nx, mpc_config.N))
                for k in range(mpc_config.N-1):
                    com_des = kinematics.joints_to_com(ref_angle_window[k])
                    com_des_dot = (kinematics.joints_to_com(ref_angle_window[k+1]) - com_des)/mpc_dt
                    x_ref[0:3, k] = com_des
                    x_ref[3:6, k] = com_des_dot
                    x_ref[6:14, k] = ref_window[k]
                    x_ref[14:22, k] = (ref_window[k+1] - ref_window[k])/mpc_dt

                # Linearize dynamics
                current_u = u_opt if step > 0 else ref_window[0]
                # In your main simulation loop:
                # In your main loop:
                # Compute A and B matrices at current state and control input
                A = np.array(A_func(state, current_u))  # Correct: state (22), current_u (8)
                B = np.array(B_func(state, current_u))  # Correct: state (22), current_u (8)

                # Solve MPC
                mpc_config.mpc.setup(A, B, mpc_config.Q, mpc_config.R, mpc_config.N)
                mpc_config.mpc.set_x_ref(x_ref)
                mpc_config.mpc.set_x0(state)
                mpc_solution = mpc_config.mpc.solve()
                u_opt = np.clip(mpc_solution["controls"].reshape(1, -1), -0.75, 0.75)

                # Convert torque to theta_ref
                theta_ref = state[6:14] + u_opt / kp

                # Execute control
                pid.execute(
                    behavior=np.rad2deg(theta_ref),
                    num_timesteps=steps_per_mpc_step,
                    viewer=viewer,
                    dt=sim_dt,
                    kp=kp,
                    ki=ki,
                    kd=kd
                )

                # Log data
                logger.log_data(
                    timestep=current_time,
                    planned_angles=theta_ref,
                    planned_com=kinematics.joints_to_com(theta_ref),
                    actual_angles=pid.data.qpos[7:15],
                    actual_com=pid.data.qpos[:3],
                    controls=u_opt
                )

            # Simulation step
            mujoco.mj_step(pid.model, pid.data)
            viewer.sync()

        # Store MPC references for plotting
        logger.store_mpc_refs(
            u_ref=reference_traj,
            u_opt=mpc_solution["controls"],
            joint_limits=mpc_config.joint_limits
        )

    # Generate plots
    logger.plot_results()

if __name__ == "__main__":
    main()