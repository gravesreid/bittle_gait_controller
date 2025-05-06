import numpy as np
import mujoco
import casadi as cs
from helpers.util import Util
from helpers.mpc_config import MPCConfig
from helpers.kinematics import KinematicsHelper
from helpers.PID_Controller import PID_Controller
from helpers.data_logger import DataLogger
from helpers.skills import bk, wkf, balance, wkf_matched_sequence
from scipy.spatial.transform import Rotation
from helpers.petoi_kinematics import PetoiKinematics
import argparse



##TO DO - Improve learning. Should store error over entire trajectory, then update reference based on that
parser = argparse.ArgumentParser(description="Tiny_MPC")
parser.add_argument("skill", type=int, help="1-bk \n 2-wkf \n 3-balance \n 4-wkf_matched_sequence")
parser.add_argument("improve_reference_gait", type=int, help="0 - Improve Reference Gait \n 1 - Do not improve reference gait")
args = parser.parse_args()

if args.skill == 1:
    skill = bk
    skill_name = "bk"
elif args.skill == 2:
    skill = wkf
    skill_name = "wkf"
elif args.skill == 3:
    skill = balance
    skill_name = "balance"
elif args.skill == 4:
    skill = wkf_matched_sequence
    skill_name = "wkf_matched_sequence"


print("Skill: ", skill_name)
print("Improve Reference Gait: ", args.improve_reference_gait)
   

class TrajectoryStore:
    def __init__(self):
        self.optimal_trajectories = []
        self.trajectory_costs = []
        self.best_cost = float('inf')
        self.max_stored = 5  # Store best N trajectories

    def add_trajectory(self, trajectory, cost):
        if len(self.optimal_trajectories) < self.max_stored:
            self.optimal_trajectories.append(trajectory)
            self.trajectory_costs.append(cost)
        else:
            # Replace worst trajectory if this one is better
            max_cost_idx = np.argmax(self.trajectory_costs)
            if cost < self.trajectory_costs[max_cost_idx]:
                self.optimal_trajectories[max_cost_idx] = trajectory
                self.trajectory_costs[max_cost_idx] = cost
                self.best_cost = cost

    def get_best_trajectory(self):
        if not self.optimal_trajectories:
            return None
        best_idx = np.argmin(self.trajectory_costs)
        self.best_cost = np.min(self.trajectory_costs)
        return self.optimal_trajectories[best_idx]

def compute_trajectory_cost(actual_angles, actual_com, x_ref, state, dt):
    """Compute cost for a trajectory considering multiple factors"""
    cost = 0
    cost_ratio = 1
    angle_error = actual_angles - x_ref[6:14, 0]
    com_error = actual_com - x_ref[0:3, 0]
    # Cost for angle error
    cost += (1-cost_ratio) * np.linalg.norm(angle_error) ** 2
    # Cost for CoM error
    cost += cost_ratio * np.linalg.norm(com_error) ** 2
    return cost

def main():
    petoi = PetoiKinematics()
    # Open log file
    with open('simulation_log.txt', 'w') as log_file:
        def log(message):
            print(message)
            log_file.write(message + '\n')
            log_file.flush()  # Ensure messages are written immediately
            
        try:
            log("=== Starting main() ===")
            # Simulation parameters
            log("Setting simulation parameters...")
            
            sim_dt = 0.001    # 1ms simulation timestep
            steps_per_step = int(sim_dt / sim_dt)
            nx = 22           # State dimension
            nu = 8            
            # Control dimension
            num_timesteps = 20
            log(f"Parameters set: sim_dt={sim_dt}, sim_dt={sim_dt}, steps_per_step={steps_per_step}")

            # PID parameters
            kp = 11
            kd = 1
            ki = 8e-1

            error_threshold = 0.06
            max_timesteps = 77
            # Convert skill to reference trajectory
            log("Creating reference trajectory...")
            gait = skill
            if gait == balance:
                bk_ref = np.deg2rad(np.repeat(gait, 64, axis=0))  # Repeat each row 100 times
            else:
                bk_ref =  np.deg2rad(gait)

            print(f"Reference trajectory shape: {bk_ref.shape}")
            reference_traj = bk_ref.copy()  
            #reference_traj = np.deg2rad(np.array(wkf))
            print(f"Reference trajectory shape after conversion: {reference_traj.shape}")
            log(f"Reference trajectory shape: {reference_traj.shape}")
            # Initialize components
            log("Initializing components...")
            horizon_multiplier = .8
            mpc_config = MPCConfig(N=int(reference_traj.shape[0]*horizon_multiplier))
            
            kinematics = KinematicsHelper()
            pid = PID_Controller("urdf/bittle.xml", 
                                 dt=sim_dt,
                                 kp=kp,
                                 ki=ki,
                                 kd=kd,max_deg_per_sec=2500)
            logger = DataLogger()
            trajectory_store = TrajectoryStore()
            prev_theta_opt = None
            log("Components initialized")

            with mujoco.viewer.launch_passive(pid.model, pid.data) as viewer:
                log("Mujoco viewer launched")
                # ... rest of the main control loop ...
                state = np.zeros(nx)
                u_opt = np.zeros(nu)
                f = mpc_config.create_dynamics()
                A_func, B_func = MPCConfig.linearize_dynamics(f)

                for step in range(int(num_timesteps)):
                    current_time = step * sim_dt
                    
                    
                    targets = np.zeros((1, 8))  # Initialize targets
                    
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
                    
                    actual_angles=pid.get_angles([3,7,0,4,2,6,1,5])
                    actual_com= np.array([pid.data.qpos[0], pid.data.qpos[2], pid.data.qpos[6]])
                    
                    log(f"State updated: {state.tolist()}")
                    
                    # Compute reference trajectories
                    log("Computing reference trajectories...")
                    torque_ref = Util.compute_pd_torque(reference_traj, kp, kd, ki, sim_dt)
                    T_torque = len(torque_ref)
                    #step = step // steps_per_step
                    log(f"Reference torque shape: {torque_ref.shape}")
            
                    # Create reference window
                    log("Creating reference window...")
                    ref_window = np.array([
                        torque_ref[k % T_torque]
                        for k in range(mpc_config.N)
                    ])
                    ref_angle_window = np.array([
                        reference_traj[k % len(reference_traj)]
                        for k in range(mpc_config.N)
                    ])
                    #print("ref_angle_window", ref_angle_window)
                    log(f"ref_window shape: {ref_window.shape}, ref_angle_window shape: {ref_angle_window.shape}")

                    # Build reference trajectories
                    log("Building reference trajectories...")
                    x_ref = np.zeros((nx, mpc_config.N))
                    for k in range(mpc_config.N-1):
                        booster = -2e5
                        # Compute desired CoM position from joint angles
                        com_des = kinematics.joints_to_com(ref_angle_window[k])
                        # Compute desired CoM velocity from finite difference
                        com_des_dot = (kinematics.joints_to_com(ref_angle_window[k+1]) - com_des) / sim_dt
                        
                        # Set CoM position reference (x, y, yaw)
                        x_ref[0:3, k] = com_des
                        # If you want to fix height or yaw, you can override here, e.g.:
                        #x_ref[1, k] = 0.06  # fixed height
                        x_ref[2, k] = 0     # fixed yaw
                        x_ref[0, k] = state[0] + booster # x position +
                        # Set CoM velocity reference
                        x_ref[3:6, k] = com_des_dot
                        x_ref[3,k] = booster
                        
                        # Set joint angles reference
                        x_ref[6:14, k] = ref_angle_window[k]
                        
                        # Set joint velocities reference by finite difference
                        x_ref[14:22, k] = (ref_angle_window[k+1] - ref_angle_window[k]) / sim_dt

                    # For the terminal step, fill in last references (angles and velocities)
                    x_ref[6:14, mpc_config.N - 1] = ref_angle_window[-1]
                    x_ref[14:22, mpc_config.N - 1] = 0  # or approximate velocity if desired
                    # Optionally set terminal CoM position and velocity similarly
                    x_ref[0:3, mpc_config.N - 1] = kinematics.joints_to_com(ref_angle_window[-1])
                    x_ref[3:6, mpc_config.N - 1] = 0
                    #log(f"x_ref shape: {x_ref.shape}")
                    #print("xref ", x_ref[6:14])
                    # Linearize dynamics
                    log("Linearizing dynamics...")
                    current_u = u_opt if step > 0 else ref_window[0]
                    u_ref = np.zeros((nu, mpc_config.N-1))
                    for k in range(mpc_config.N-1):
                        u_ref[:, k] = ref_angle_window[k % len(ref_window), :]
                    grf_ref = u_ref.copy()
                    # for k in range(mpc_config.N-1):
                    #     grf_numeric = mpc_config.joints_to_GRF_func(ref_angle_window[k], u_ref[:, k])
                    #     grf_ref[:, k] = np.array(grf_numeric).flatten()
                    if step == 0:
                        log("First step - using reference control")
                        current_u = grf_ref[:, 0]
                    else:
                        log(f"Using previous MPC solution: {grf_opt.tolist()}")
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
                    theta_opt = mpc_solution["states_all"][:,6:14]  # Get optimal joint angles
                    # Ensure theta_opt is 2D (N_steps, 8_joints)
                    theta_opt = mpc_solution["states_all"]
                    if theta_opt.ndim == 1:
                        theta_opt = theta_opt.reshape(1, -1)  # Force 2D
                    theta_opt = theta_opt[:, 6:14]  # Extract joint angles

                    # Log only the first step of the MPC horizon
                    logger.log_data(
                        timestep=current_time,
                        planned_angles=theta_opt[0],  # Now guaranteed to be 8 elements
                        planned_com=x_ref[0:3, 0],
                        actual_angles=actual_angles,
                        actual_com=actual_com,
                        controls=u_opt,
                        reference_angles=ref_angle_window,
                        log_file=log_file
                    )
                    if args.improve_reference_gait == 0:
                        # Compute cost and store if good
                        traj_cost = compute_trajectory_cost(actual_angles, actual_com, x_ref, state, sim_dt)
                        trajectory_store.add_trajectory(theta_opt.copy(), traj_cost)
                        
                        # Get best trajectory to track
                        best_trajectory = trajectory_store.get_best_trajectory()
                        if best_trajectory is not None:
                            if trajectory_store.best_cost > traj_cost: 
                                # Update reference with best known trajectory
                                log(f"Found better trajectory (cost: {trajectory_store.trajectory_costs[trajectory_store.trajectory_costs.index(min(trajectory_store.trajectory_costs))]:.3f}), updating reference...")
                                x_ref[6:14,:] = best_trajectory.T
                                # Update velocities 
                                for k in range(mpc_config.N-1):
                                    x_ref[14:22,k] = (best_trajectory[k+1] - best_trajectory[k]) / sim_dt
                                x_ref[14:22,-1] = x_ref[14:22,-2]  # Copy last velocity
                            else:
                                log(f"Current trajectory is better (cost: {traj_cost:.3f}), not updating reference.")

                    # Add trajectory smoothing
                    blend_ratio = 0.5  # Tune based on stability
                    if prev_theta_opt is None:
                        theta_ref = theta_opt  # First iteration, use optimal trajectory directly
                    else:
                        theta_ref = (1-blend_ratio)*prev_theta_opt + blend_ratio*theta_opt
                    targets = np.rad2deg(theta_opt)
                    prev_theta_opt = theta_ref

                    # MPC Update finished
                    # Simulation step
                    #targets = np.array(wkf)
                    for t,target in enumerate(targets):
                        pid.set_targets(target)
                        for s in range(max_timesteps):
                            error = pid.step(viewer)
                            
                            # Capture CoM and angles at simulation timestep
                            actual_com = np.array([
                                pid.data.qpos[0],  # x
                                pid.data.qpos[1],  # z (height)
                                Rotation.from_quat(pid.data.qpos[3:7]).as_euler('zyx')[0]  # yaw
                            ])
                            actual_angles = pid.get_angles([3,7,0,4,2,6,1,5])
                            
                            # Log PID step
                            logger.log_pid_step(actual_com, actual_angles, current_time)
                            current_time += sim_dt  # Increment simulation time
                            
                            if np.all(error < error_threshold):
                                break
                    # for i in range(500):
                    #     pid.set_targets(targets[-1])
                    #     for s in range(max_timesteps):
                    #         error = pid.step(viewer)
                    #         if np.all((error) < error_threshold):
                    #             print(f"Converged in {s}")
                    #             break
                    
                    
                    
                # pid.execute(targets,100,sim_dt,kp,ki,kd, viewer=viewer, plotty=False)
                # targets = []

        except KeyboardInterrupt:
            log("\nReceived interrupt signal, terminating gracefully...")
        except Exception as e:
            log(f"\nError occurred: {str(e)}")
            raise
        finally:
            log("\nCleaning up and saving final results...")
            # Save any final plots/data
            if 'logger' in locals():
                logger.plot_xy_displacement(skill_name)
                logger.plot_final_displacement(skill_name)
            log("Program terminated")

if __name__ == "__main__":
    
    main()