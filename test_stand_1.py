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
    [18,  54,  58,  52,   7,  13,  -2,   9],
  [14,  56,  52,  54,  12,  14,  -3,  10],
  [15,  57,  46,  55,  17,  15,  -4,  11],
  [16,  58,  38,  57,  16,  17,  -2,  12],
  [19,  59,  31,  58,  13,  19,   1,  14],
  [22,  59,  24,  60,  12,  22,   6,  14],
  [26,  60,  21,  61,  10,  24,  12,  16],
  [28,  58,  23,  62,   9,  30,  16,  18],
  [31,  61,  25,  63,   8,  30,  13,  20],
  [34,  67,  28,  63,   7,  23,  12,  24],
  [36,  69,  31,  63,   6,  15,  11,  27],
  [38,  68,  34,  63,   7,   8,  10,  31],
  [41,  65,  36,  62,   7,   3,   9,  35],
  [43,  60,  39,  65,   7,  -1,   9,  37],
  [46,  55,  41,  70,   7,  -3,   9,  27],
  [47,  50,  43,  71,   7,  -5,   9,  23],
  [49,  43,  47,  71,   8,  -5,   7,  14],
  [51,  35,  48,  69,   8,  -3,   8,   8],
  [53,  28,  51,  66,   9,   1,   8,   2],
  [54,  18,  52,  63,  12,   7,   9,  -2],
  [55,  15,  54,  56,  14,  10,  10,  -3],
  [56,  13,  55,  51,  15,  16,  11,  -4],
  [57,  16,  57,  43,  17,  16,  12,  -3],
  [58,  18,  58,  36,  19,  14,  14,  -2],
  [59,  21,  60,  28,  21,  12,  14,   2],
  [60,  25,  61,  23,  24,  10,  16,   9],
  [60,  28,  62,  22,  27,   9,  18,  15],
  [60,  31,  63,  23,  30,   8,  20,  16],
  [66,  32,  63,  26,  24,   7,  24,  13],
  [68,  35,  63,  29,  19,   6,  27,  11],
  [68,  38,  63,  32,  10,   6,  31,  10],
  [66,  40,  62,  35,   4,   7,  35,  10],
  [62,  42,  65,  37,   0,   7,  37,   9],
  [57,  45,  70,  39,  -2,   7,  27,   9],
  [52,  47,  71,  41,  -5,   7,  23,   9],
  [45,  48,  71,  44,  -5,   8,  14,   9],
  [38,  51,  69,  47,  -3,   8,   8,   7],
  [30,  52,  66,  49,  -1,   9,   2,   8],
  [22,  53,  63,  51,   4,  12,  -2,   8]
]

max_index = len(wkf)
index = 0
kp = 12
ki = 0
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
e = 100
error_vec = 100*np.ones(8)
prev_e = 0
num_timesteps = 100
num_joints = 8
angle_holder = np.zeros((num_timesteps, num_joints))
reference_holder = np.zeros((num_timesteps, num_joints))
actuator_nums = [3,7,0,4,2,6,1,5]
with mujoco.viewer.launch_passive(model, data) as viewer:
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
    for i in range(num_timesteps):
        # Just try to control left back leg
        while np.linalg.norm(error_vec) > 2.5:
            
            for j, num in enumerate(actuator_nums):
                #num = 5
                current_angle = data.qpos[joint_to_qpos[actuator_map[num]]]
                desired_angle = np.deg2rad(wkf[index][num])
                angle_holder[i, j] = current_angle  # Store for each joint
                reference_holder[i, j] = desired_angle  # Store for each 
                prev_e = e
                e = -current_angle + desired_angle
                error_vec[j] = e
                ctrl = kp*e + ki*e*dt + kd*((e-prev_e))/dt

                data.ctrl[actuator_to_ctrl[actuator_map[num]]] = np.clip(ctrl, -1, 1)
                mujoco.mj_step(model, data)
                viewer.sync()
                #time.sleep(0.005)
            print(np.linalg.norm(error_vec))
            e = 10
        error_vec = 100*np.ones(8)
       
        index+=1
        
        # if abs(e) < 0.1:
        #     index +=1
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