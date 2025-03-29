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




wkf = [
[18, 54, 58, 52, 7, 13, -2, 9],
[14, 56, 52, 54, 12, 14, -3, 10],
[15, 57, 46, 55, 17, 15, -4, 11],
[16, 58, 38, 57, 16, 17, -2, 12],
[46, 55, 41, 70, 7, -3, 9, 27],
[47, 50, 43, 71, 7, -5, 9, 23],
[49, 43, 47, 71, 8, -5, 7, 14],
[51, 35, 48, 69, 8, -3, 8, 8],
[53, 28, 51, 66, 9, 1, 8, 7, 9, -2],
[55, 15, 54, 56, 14, 10, 10, -3],
[56, 13, 55, 51, 15, 16, 11, -4],
[57, 16, 57, 43, 17, 16, 12, -3],
[58, 18, 58, 36, 19, 14, 14, -2],
[52, 47, 71, 41, -5, 7, 23, 9],
[45, 48, 71, 44, -5, 8, 14, 9],
[38, 51, 69, 47, -3, 8, 8, 7],
[30, 52, 66, 49, -1, 9, 2, 8],
[22, 53, 63, 51, 4, 12, -2, 8]
]

bk = [
  [36,  40,  36,  62,   6,  -3,   6,   1],
  [34,  47,  32,  63,   7,  -4,   7,   4],
  [30,  53,  28,  59,   8,  -3,   9,   9],
  [26,  58,  25,  57,  10,  -2,  10,  10],
  [22,  57,  26,  55,  12,   2,   6,   8],
  [18,  51,  29,  52,  14,   8,   2,   7],
  [15,  51,  36,  50,  15,   6,  -2,   6],
  [17,  48,  43,  47,   9,   5,  -3,   5],
  [21,  45,  49,  44,   5,   5,  -4,   5],
  [29,  43,  55,  42,   2,   5,  -3,   5],
  [35,  39,  60,  38,  -1,   6,  -1,   6],
  [42,  36,  63,  35,  -3,   6,   1,   6],
  [49,  32,  62,  31,  -4,   7,   6,   8],
  [54,  28,  58,  28,  -3,   9,  10,   9],
  [57,  26,  57,  24,   0,  10,   9,  11],
  [56,  21,  54,  26,   3,  12,   8,   4],
  [51,  17,  52,  31,   8,  15,   6,   1],
  [50,  15,  49,  38,   6,  14,   6,  -2],
  [47,  18,  47,  44,   5,   8,   5,  -3],
  [45,  24,  44,  51,   5,   4,   5,  -4],
  [42,  30,  41,  56,   5,   1,   5,  -3],
  [38,  37,  37,  60,   6,  -2,   6,  -1]
]
bk_left = [[45,  54,  45,  56,  -4, -13,   5, -10],
  [44,  61,  43,  58,  -3, -12,   5, -10],
  [43,  67,  39,  60,  -3, -10,   6,  -9],
  [42,  72,  36,  62,  -3,  -6,   6,  -8],
  [41,  70,  32,  62,  -3,   0,   7,  -4],
  [40,  63,  30,  60,  -3,   8,   8,  -3],
  [38,  58,  26,  58,  -2,  10,  10,  -2],
  [37,  56,  21,  57,  -3,   9,  10,  -2],
  [36,  54,  22,  56,  -4,   8,   2,  -3],
  [39,  52,  28,  55,  -8,   6,  -6,  -3],
  [41,  49,  37,  55,  -9,   6, -10,  -3],
  [44,  47,  46,  54,  -9,   5, -12,  -3],
  [48,  44,  54,  53, -10,   5, -13,  -3],
  [49,  42,  61,  52, -10,   5, -12,  -3],
  [52,  38,  67,  51, -10,   6, -10,  -4],
  [55,  35,  72,  51, -10,   6,  -6,  -4],
  [54,  31,  70,  49,  -7,   8,   0,  -4],
  [53,  28,  63,  48,  -5,   9,   8,  -4],
  [51,  25,  58,  46,  -4,  10,  10,  -4],
  [50,  21,  56,  46,  -4,   9,   9,  -5],
  [49,  23,  54,  48,  -4,  -1,   8,  -9],
  [48,  30,  52,  48,  -4,  -7,   6, -10],
  [47,  39,  49,  51,  -4, -11,   6, -10],
  [46,  48,  47,  54,  -4, -13,   5, -10]
  ]
wkf_left = [  [47,  54,  58,  51,  -2,  13,  -2,   2],
  [45,  56,  52,  52,   0,  14,  -3,   2],
  [45,  57,  46,  52,   1,  15,  -4,   2],
  [45,  58,  38,  53,   1,  17,  -2,   2],
  [46,  59,  31,  54,   1,  19,   1,   2],
  [47,  59,  24,  54,   1,  22,   6,   3],
  [48,  60,  21,  55,   2,  24,  12,   3],
  [49,  58,  23,  56,   1,  30,  16,   3],
  [49,  61,  25,  57,   1,  30,  13,   3],
  [50,  67,  28,  57,   1,  23,  12,   4],
  [51,  69,  31,  58,   2,  15,  11,   5],
  [52,  68,  34,  58,   2,   8,  10,   5],
  [52,  65,  36,  59,   2,   3,   9,   6],
  [53,  60,  39,  61,   2,  -1,   9,   5],
  [54,  55,  41,  62,   2,  -3,   9,   3],
  [54,  50,  43,  62,   2,  -5,   9,   2],
  [54,  43,  47,  60,   3,  -5,   7,   1],
  [55,  35,  48,  58,   3,  -3,   8,   1],
  [56,  28,  51,  57,   3,   1,   8,  -1],
  [57,  18,  52,  55,   3,   7,   9,  -1],
  [57,  15,  54,  53,   4,  10,  10,  -2],
  [58,  13,  55,  50,   5,  16,  11,  -2],
  [58,  16,  57,  49,   5,  16,  12,  -2],
  [59,  18,  58,  46,   6,  14,  14,  -2],
  [59,  21,  60,  44,   6,  12,  14,  -2],
  [60,  25,  61,  43,   7,  10,  16,   0],
  [60,  28,  62,  42,   7,   9,  18,   1],
  [60,  31,  63,  42,  10,   8,  20,   3],
  [62,  32,  63,  42,   6,   7,  24,   3],
  [62,  35,  63,  44,   6,   6,  27,   1],
  [63,  38,  63,  44,   3,   6,  31,   1],
  [61,  40,  62,  45,   2,   7,  35,   1],
  [60,  42,  65,  46,   1,   7,  37,   1],
  [58,  45,  70,  47,   0,   7,  27,   1],
  [57,  47,  71,  48,  -1,   7,  23,   2],
  [54,  48,  71,  49,  -1,   8,  14,   1],
  [53,  51,  69,  49,  -2,   8,   8,   1],
  [50,  52,  66,  50,  -2,   9,   2,   1],
  [49,  53,  63,  51,  -2,  12,  -2,   2]]

balance = [[20, 55, 55, 55, 7, 10, 0, 10]]




def execute(behavior, num_timesteps, dt, kp, ki, kd, clipped_control = False, limits = [0,0], plotty =False):
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
    #print("Actuator to ctrl mapping:", actuator_to_ctrl)
    e = 10000
    error_vec = 100*np.ones(8)
    prev_error_vec = np.zeros(8)
    prev_e = 0
    
    num_joints = 8
    angle_holder = np.zeros((num_timesteps, num_joints))
    reference_holder = np.zeros((num_timesteps, num_joints))
    actuator_nums = [3,7,0,4,2,6,1,5]
    control_effort = np.zeros((num_timesteps, num_joints))
    index = 0

    for frame in behavior:
        # Right-back-shoulder-joint is index 2 in actuator_nums (original index 2)
        # Right-front-shoulder-joint is index 1 in actuator_nums (original index 1)
        frame[1] *= -1  # Right-front-shoulder
        frame[2] *= -1  # Right-back-shoulder

    #walking loop
    for i in range(num_timesteps):
        
        desired_angles = np.array([np.deg2rad(behavior[index][num]) for num in actuator_nums])
        
        # Calculate errors
        error_vec = desired_angles - np.array([data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums])
        
        # PID control - single update per timestep
        for j, num in enumerate(actuator_nums):
            e = error_vec[j]
            prev_e = prev_error_vec[j]
            ctrl = kp*e + ki*e*dt + kd*(e - prev_e)/dt
            # PID control with clipping
            if clipped_control == True:
                ctrl = np.clip(ctrl,limits[0],limits[1])
            
            data.ctrl[actuator_to_ctrl[actuator_map[num]]] = ctrl
            prev_error_vec[j] = e 
        
        # Single simulation step
        mujoco.mj_step(model, data)
        viewer.sync()


        if i % 50 == 0:
            index = (index + 1) % len(behavior)

        
        # After convergence (or reaching max iterations), store final angles:
        angle_holder[i, :] = [data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums]
        reference_holder[i, :] = desired_angles


            

        #data.ctrl[actuator_to_ctrl["left-back-knee-joint"]] = np.clip(np.deg2rad(wkf[index][7]) / np.pi, -1, 1)
        #index += 1
        #time.sleep(0.005)

    
    def plot():        
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
    
    if plotty:
        plot()

kp = 10000
ki = 50
kd = 100
dt = 0.001
num_timesteps = 10000
with mujoco.viewer.launch_passive(model, data) as viewer:
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
    def sequence(commands, actions):
        for i, command in enumerate(commands):
            print(actions[i])
            execute(command,num_timesteps,dt, kp,ki,kd, plotty=True)
    commands1 = [balance,wkf, wkf_left,bk_left,bk]
    commands2 = [balance, bk_left]
    actions = ["Stand", "Backwards Left"]
    sequence(commands2, actions)