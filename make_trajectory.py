import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from cheetahgym.controllers.trajectory_generator import TrajectoryGenerator
from cheetahgym.controllers.foot_swing_trajectory import FootSwingTrajectory
import os

class DummyLowLevelState:
    def __init__(self):
        self.body_pos = np.array([0.0, 0.0, 0.29])
        self.body_rpy = np.array([0.0, 0.0, 0.0])
        self.body_linear_vel = np.array([0.1, 0.0, 0.0])

dummy_state = DummyLowLevelState()

# Load model and data
model = mujoco.MjModel.from_xml_path("urdf/bittle_PID_latest.xml")
data = mujoco.MjData(model)

# Build mappings for qpos and qvel indices for each actuator/joint
actuator_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    for i in range(model.nu)
]
actuator_to_qpos = {}
for name in actuator_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    actuator_to_qpos[name] = model.jnt_qposadr[joint_id]

# Initialize Trajectory Generator
def generate_reference_gait():
    traj_gen = TrajectoryGenerator(planning_horizon=10, iterationsBetweenMPC=17, dt=0.002)

    # Define gait parameters
    offsets = np.array([0, 5, 10, 15])  # Staggered offsets for legs
    durations = np.array([5, 5, 5, 5])  # Equal durations for all legs
    vel = np.array([0.1, 0.0, 0.0])  # Forward velocity
    vel_rpy = np.array([0.0, 0.0, 0.0])  # No rotational velocity
    initial_pose = np.array([0.0, 0.0, 0.29])  # Initial body pose

    # Initialize static gait
    traj_gen.initialize_static_gait(offsets, durations, traj_gen.iterationsBetweenMPC, vel, vel_rpy, initial_pose)
    

    # Generate reference gait
    reference_gait = []
    for i in range(traj_gen.planning_horizon):
        foot_positions = traj_gen._get_pfoot_des().reshape(4, 3)  # Get foot positions for each leg
        print(f'Foot positions at step {i}: {foot_positions}')
        reference_gait.append(foot_positions[:, 2])  # Use z-coordinates as reference
        print(f'Foot positions at step {i}: {foot_positions[:, 2]}')
        traj_gen.step(foot_positions, low_level_state=dummy_state)  # Step the trajectory generator
        print(f'Foot positions at step {i}: {foot_positions[:, 2]}')


    return reference_gait

# Execute the reference gait in the PID framework
def execute_reference_gait(reference_gait, num_timesteps, dt, kp, ki, kd, plotty=False):
    actuator_map = {
        3: "left-back-shoulder-joint",
        7: "left-back-knee-joint",
        0: "left-front-shoulder-joint",
        4: "left-front-knee-joint",
        2: "right-back-shoulder-joint",
        6: "right-back-knee-joint",
        1: "right-front-shoulder-joint",
        5: "right-front-knee-joint",
    }
    actuator_to_ctrl = {name: i for i, name in enumerate(actuator_names)}
    joint_to_qpos = {name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in actuator_names}

    num_joints = 8
    angle_holder = np.zeros((num_timesteps, num_joints))
    reference_holder = np.zeros((num_timesteps, num_joints))
    actuator_nums = [3, 7, 0, 4, 2, 6, 1, 5]

    # PID control loop
    error_vec = np.zeros(num_joints)
    prev_error_vec = np.zeros(num_joints)
    int_error_vec = np.zeros(num_joints)

    for t in range(num_timesteps):
        desired_angles = np.array([np.deg2rad(reference_gait[t % len(reference_gait)][num]) for num in actuator_nums])

        # Calculate errors
        error_vec = desired_angles - np.array([data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums])
        int_error_vec += error_vec * dt
        de_dt_vec = (error_vec - prev_error_vec) / dt

        # PID control
        for j, num in enumerate(actuator_nums):
            ctrl = kp * error_vec[j] + ki * int_error_vec[j] + kd * de_dt_vec[j]
            data.ctrl[actuator_to_ctrl[actuator_map[num]]] = ctrl
            prev_error_vec[j] = error_vec[j]

        # Step simulation
        mujoco.mj_step(model, data)

        # Store angles
        angle_holder[t, :] = [data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums]
        reference_holder[t, :] = desired_angles

    # Plot results
    if plotty:
        time_array = np.arange(num_timesteps) * dt
        plt.figure(figsize=(15, 20))
        joint_names = [actuator_map[num] for num in actuator_nums]
        for j in range(num_joints):
            plt.subplot(4, 2, j + 1)
            plt.plot(time_array, angle_holder[:, j], label="Actual")
            plt.plot(time_array, reference_holder[:, j], "--", label="Reference")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle (rad)")
            plt.title(f"{joint_names[j]} Trajectory")
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    kp = 6e4
    ki = 5e2
    kd = 5e1
    dt = 1e-3
    num_timesteps = int(1e4)

    reference_gait = generate_reference_gait()
    print(f'Reference gait: {reference_gait}')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        execute_reference_gait(reference_gait, num_timesteps, dt, kp, ki, kd, plotty=True)