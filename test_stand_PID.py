import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Load model and data
model = mujoco.MjModel.from_xml_path("urdf/bittle_PID_latest.xml")
data = mujoco.MjData(model)

# Build mappings for qpos and qvel indices for each actuator/joint
actuator_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    for i in range(model.nu)
]
print("Actuator names:", actuator_names)
actuator_to_qpos = {}
actuator_to_qvel = {}

for name in actuator_names:
    # Get the joint id by name
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    # qpos index for configuration (e.g., angle) and qvel index for velocity
    qpos_index = model.jnt_qposadr[joint_id]
    qvel_index = model.jnt_dofadr[joint_id]
    actuator_to_qpos[name] = qpos_index
    actuator_to_qvel[name] = qvel_index

print("Actuator to qpos mapping:", actuator_to_qpos)
print("Actuator to qvel mapping:", actuator_to_qvel)




balance = [
    [30, -30, -30, 30, 30, -30, -30, 30]
]


index = 0
kp = 20
ki = 1
kd = 1
dt = 0.001
joint_to_qpos = {}
for name in actuator_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_to_qpos[name] = model.jnt_qposadr[joint_id]
actuator_map = {
    3:"left-back-shoulder-joint",
    7:"left-back-knee-joint",
    0:"left-front-shoulder-joint",
    4:"left-front-knee-joint",
    2:"right-back-shoulder-joint",
    6:"right-back-knee-joint",
    1:"right-front-shoulder-joint",
    5:"right-front-knee-joint",

}
actuator_to_ctrl = {name: i for i, name in enumerate(actuator_names)}
print("Actuator to ctrl mapping:", actuator_to_ctrl)
e = 10000
lowest_norm = 100000
error_vec = 100*np.ones(8)
prev_error_vec = np.zeros(8)
prev_e = 0
num_timesteps = 10
num_joints = 8
angle_holder = np.zeros((num_timesteps, num_joints))
reference_holder = np.zeros((num_timesteps, num_joints))
actuator_nums = [3,7,0,4,2,6,1,5]
max_inner_iterations = 1000
control_effort = np.zeros((num_timesteps, num_joints))
with mujoco.viewer.launch_passive(model, data) as viewer:
    #if os.name == 'nt':
    #    import ctypes
    #    hwnd = ctypes.windll.user32.GetForegroundWindow()
    #    ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3

    for i in range(num_timesteps):
        
        # Desired angles for current timestep
        #desired_angles = np.array([np.deg2rad(wkf[index][num]) for num in actuator_nums])
        desired_angles = np.array([np.deg2rad(balance[0][num]) for num in actuator_nums])

        # Initialize errors
        error_vec = desired_angles - np.array([data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums])
        prev_error_vec = np.copy(error_vec)

        inner_iter = 0
        start_time = time.time()
        while np.any(np.abs(error_vec) >= 0.1) and inner_iter < max_inner_iterations or (time.time() - start_time) < 10:
            for j, num in enumerate(actuator_nums):
                current_angle = data.qpos[joint_to_qpos[actuator_map[num]]]
                desired_angle = desired_angles[j]

                # Calculate errors
                e = desired_angle - current_angle
                prev_e = prev_error_vec[j]

                # PID control
                ctrl = kp*e + ki*e*dt + kd*((e - prev_e))/dt
                data.ctrl[actuator_to_ctrl[actuator_map[num]]] = np.clip(ctrl, -10, 10)
                clipped_ctrl = data.ctrl[actuator_to_ctrl[actuator_map[num]]] 
                control_effort[i, j] = clipped_ctrl

                # Update previous error storage
                prev_error_vec[j] = e

                # Simulation step (once per joint update cycle)
                mujoco.mj_step(model, data)
                viewer.sync()

            # Update error vector after stepping simulation
            error_vec = desired_angles - np.array([data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums])

            inner_iter += 1

        if inner_iter == max_inner_iterations:
            print(f"Warning: timestep {i} did not fully converge!")

        # After convergence (or reaching max iterations), store final angles:
        angle_holder[i, :] = [data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums]
        reference_holder[i, :] = desired_angles


            

        #data.ctrl[actuator_to_ctrl["left-back-knee-joint"]] = np.clip(np.deg2rad(wkf[index][7]) / np.pi, -1, 1)
        #index += 1
        #time.sleep(0.005)



        
# Create time array
time_array = np.arange(num_timesteps) * dt

# Create subplots for all joints
plt.figure(figsize=(15, 20))

# Get joint names from actuator_map for titles
joint_names = [actuator_map[num] for num in actuator_nums]

for j in range(num_joints):
    plt.subplot(4, 2, j+1)
    plt.plot(time_array, angle_holder[:, j], label='Actual')
    plt.plot(time_array, reference_holder[:, j], '--', label='Reference')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(f'{joint_names[j]} Trajectory')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 20))

for j in range(num_joints):
    plt.subplot(4, 2, j+1)
    plt.plot(time_array, control_effort[:, j])
    plt.xlabel('Time (s)')
    plt.ylabel('Control Effort')
    plt.title(f'{joint_names[j]} Control Effort')
    plt.grid(True)

plt.tight_layout()
plt.show()
