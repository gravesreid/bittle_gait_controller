import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Load model and data
model = mujoco.MjModel.from_xml_path("urdf/bittle_nograv.xml")
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


wkf = [  
  [ 9,  49,  53,  45,  24,  20,  -2,  15],
  [ 8,  50,  41,  46,  28,  21,  -1,  15],
  [10,  51,  26,  47,  26,  22,   6,  16],
  [12,  52,  23,  48,  24,  24,   9,  17],
  [14,  52,  20,  49,  22,  26,  12,  18],
  [16,  53,  17,  51,  21,  27,  17,  18],
  [18,  53,  14,  52,  20,  29,  22,  19],
  [21,  54,  11,  54,  18,  30,  27,  19],
  [22,  54,  11,  54,  18,  32,  29,  20],
  [25,  54,  13,  55,  16,  34,  27,  21],
  [26,  54,  16,  56,  16,  37,  24,  23],
  [28,  54,  18,  56,  15,  39,  23,  24],
  [30,  52,  20,  57,  14,  45,  22,  26],
  [32,  54,  22,  57,  14,  44,  21,  28],
  [33,  58,  24,  57,  15,  36,  20,  30],
  [34,  61,  26,  57,  15,  31,  19,  32],
  [36,  64,  28,  57,  14,  24,  18,  35],
  [38,  66,  29,  57,  14,  20,  18,  38],
  [39,  67,  31,  57,  14,  16,  17,  40],
  [41,  64,  32,  56,  14,   5,  17,  43],
  [42,  55,  35,  57,  14,  -1,  16,  44],
  [44,  44,  37,  62,  15,  -3,  14,  35],
  [45,  30,  39,  66,  15,   1,  14,  29],
  [46,  21,  40,  68,  15,   5,  14,  23],
  [47,  19,  42,  70,  16,   9,  14,  19],
  [48,  16,  43,  70,  17,  12,  15,  17],
  [49,  12,  44,  67,  18,  17,  15,   5],
  [49,   9,  46,  59,  20,  24,  15,  -2],
  [50,   8,  47,  47,  21,  28,  16,  -2],
  [51,  10,  48,  34,  22,  26,  16,   1],
  [52,  12,  49,  24,  24,  24,  17,   6],
  [52,  14,  50,  21,  26,  22,  18,  10],
  [53,  16,  51,  19,  27,  21,  19,  12],
  [53,  18,  52,  15,  29,  20,  20,  19],
  [54,  21,  54,  12,  30,  18,  19,  24],
  [54,  22,  55,  12,  32,  18,  20,  27],
  [54,  25,  55,  11,  34,  16,  22,  29],
  [54,  26,  56,  14,  37,  16,  24,  26],
  [54,  28,  56,  17,  39,  15,  25,  24],
  [52,  30,  57,  18,  45,  14,  27,  23],
  [54,  32,  57,  21,  44,  14,  29,  21],
  [58,  33,  57,  23,  36,  15,  31,  20],
  [61,  34,  57,  24,  31,  15,  33,  20],
  [64,  36,  57,  26,  24,  14,  36,  19],
  [66,  38,  57,  28,  20,  14,  39,  18],
  [67,  39,  56,  30,  16,  14,  42,  17],
  [64,  41,  56,  32,   5,  14,  45,  17],
  [55,  42,  59,  33,  -1,  14,  41,  17],
  [44,  44,  64,  35,  -3,  15,  33,  16],
  [30,  45,  67,  38,   1,  15,  27,  14],
  [21,  46,  68,  39,   5,  15,  22,  14],
  [19,  47,  70,  41,   9,  16,  18,  14],
  [16,  48,  69,  42,  12,  17,  11,  14],
  [12,  49,  63,  44,  17,  18,   1,  15]
]


max_index = len(wkf)
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
for frame in wkf:
    # Right-back-shoulder-joint is index 2 in actuator_nums (original index 2)
    # Right-front-shoulder-joint is index 1 in actuator_nums (original index 1)
    frame[1] *= -1  # Right-front-shoulder
    frame[2] *= -1  # Right-back-shoulder
    frame[5] *= -1  # Right-front-knee
    frame[6] *= -1  # Right-back-knee
with mujoco.viewer.launch_passive(model, data) as viewer:
    #if os.name == 'nt':
    #    import ctypes
    #    hwnd = ctypes.windll.user32.GetForegroundWindow()
    #    ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3

    for i in range(num_timesteps):
        
        # Desired angles for current timestep
        desired_angles = np.array([np.deg2rad(wkf[index][num]) for num in actuator_nums])

        # Initialize errors
        error_vec = desired_angles - np.array([data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums])
        prev_error_vec = np.copy(error_vec)

        inner_iter = 0
        while np.any(np.abs(error_vec) >= 0.1) and inner_iter < max_inner_iterations:
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

        # Increment to next waypoint/frame
        index += 1
        if index == max_index:
            index = 0

            

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