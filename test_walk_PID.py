import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on your setup

import os

# Load model and data
model = mujoco.MjModel.from_xml_path("urdf/bittle.xml")
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
  [45,  59,  47,  58, -10,  -4,  -5,  -4],
  [47,  59,  47,  59,  -5,  -4,  -5,  -4],
  [50,  60,  48,  59,  -5,  -3,  -5,  -4],
  [50,  60,  48,  59,  -5,  -3,  -5,  -4],
  [50,  61,  49,  60,  -5,  -3,  -5,  -3],
  [51,  61,  49,  60,  -5,  -3,  -5,  -3],
  [51,  61,  50,  62,  -5,  -3,  -5,  -2],
  [52,  61,  50,  68,  -5,  -3,  -5,  -5],
  [52,  61,  50,  73,  -5,  -3,  -5, -10],
  [52,  62,  51,  79,  -5,  -2,  -5, -17],
  [52,  62,  51,  83,  -5,  -2,  -5, -23],
  [52,  62,  52,  87,  -5,  -2,  -5, -31],
  [53,  63,  52,  86,  -5,  -2,  -5, -37],
  [53,  63,  52,  84,  -5,  -2,  -5, -41],
  [54,  64,  52,  79,  -5,   0,  -5, -44],
  [54,  70,  53,  79,  -5,  -2,  -5, -44],
  [54,  76,  53,  79,  -5,  -7,  -5, -44],
  [55,  81,  53,  78,  -5, -14,  -5, -44],
  [55,  85,  54,  71,  -5, -21,  -5, -43],
  [56,  91,  54,  62,  -5, -29,  -5, -40],
  [56,  91,  55,  54,  -5, -35,  -5, -35],
  [56,  89,  55,  48,  -5, -40,  -5, -29],
  [56,  84,  56,  44,  -4, -43,  -5, -21],
  [56,  84,  56,  42,  -4, -43,  -5, -14],
  [57,  84,  56,  41,  -4, -43,  -4,  -9],
  [57,  84,  56,  43,  -4, -43,  -4,  -5],
  [58,  77,  56,  45,  -4, -43,  -4,  -5],
  [58,  69,  57,  45,  -4, -40,  -4,  -5],
  [58,  61,  57,  45,  -4, -36,  -4,  -5],
  [58,  55,  58,  46,  -4, -30,  -4,  -5],
  [58,  49,  58,  46,  -4, -23,  -4,  -5],
  [59,  46,  58,  47,  -4, -15,  -4,  -5],
  [59,  45,  58,  47,  -4, -10,  -4,  -5],
  [59,  47,  59,  48,  -4,  -5,  -4,  -5],
  [60,  50,  59,  48,  -3,  -5,  -4,  -5],
  [60,  50,  59,  48,  -3,  -5,  -4,  -5],
  [61,  50,  60,  49,  -3,  -5,  -3,  -5],
  [61,  51,  60,  49,  -3,  -5,  -3,  -5],
  [61,  51,  62,  50,  -3,  -5,  -2,  -5],
  [61,  52,  68,  50,  -3,  -5,  -5,  -5],
  [61,  52,  73,  51,  -3,  -5, -10,  -5],
  [62,  52,  79,  51,  -2,  -5, -17,  -5],
  [62,  52,  83,  52,  -2,  -5, -23,  -5],
  [63,  52,  87,  52,  -2,  -5, -31,  -5],
  [63,  53,  86,  52,  -2,  -5, -37,  -5],
  [63,  53,  84,  52,  -1,  -5, -41,  -5],
  [67,  54,  79,  52,   0,  -5, -44,  -5],
  [73,  54,  79,  53,  -4,  -5, -44,  -5],
  [78,  54,  79,  53, -10,  -5, -44,  -5],
  [83,  55,  78,  54, -17,  -5, -44,  -5],
  [89,  55,  71,  54, -25,  -5, -43,  -5],
  [90,  56,  62,  55, -31,  -5, -40,  -5],
  [91,  56,  54,  55, -37,  -5, -35,  -5],
  [87,  56,  48,  55, -41,  -5, -29,  -5],
  [84,  56,  44,  56, -43,  -4, -21,  -5],
  [84,  56,  42,  56, -43,  -4, -14,  -5],
  [84,  57,  41,  56, -43,  -4,  -9,  -4],
  [81,  57,  43,  56, -43,  -4,  -5,  -4],
  [73,  58,  45,  57, -42,  -4,  -5,  -4],
  [64,  58,  45,  57, -39,  -4,  -5,  -4],
  [57,  58,  45,  58, -33,  -4,  -5,  -4],
  [52,  58,  46,  58, -26,  -4,  -5,  -4],
  [48,  58,  46,  58, -18,  -4,  -5,  -4],
  [46,  59,  47,  58, -12,  -4,  -5,  -4],
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

#balance = [[20, 55, 55, 55, 7, 10, 0, 10]]
balance = [[30, 30, 30, 30, 30, 30, 30, 30]]




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
    int_error_vec = np.zeros(8)
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
        frame[5] *= -1  # Right-back-knee
        frame[6] *= -1  # Right-front-shoulder

    #walking loop
    for i in range(num_timesteps):
        
        desired_angles = np.array([np.deg2rad(behavior[index][num]) for num in actuator_nums])
        
        # Calculate errors
        error_vec = desired_angles - np.array([data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums])
        int_error_vec += error_vec*dt
        de_dt_vec = (error_vec - prev_error_vec)/dt

        # PID control - single update per timestep
        for j, num in enumerate(actuator_nums):
            e = error_vec[j]
            de_dt = de_dt_vec[j]
            int_e = int_error_vec[j]

            ctrl = kp*e + ki*int_e + kd*de_dt

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

kp = 6e4
ki = 5e2
kd = 5e1
dt = 1e-3
num_timesteps = int(1e4)
with mujoco.viewer.launch_passive(model, data) as viewer:
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
    #print("Stand")
    #execute(balance,num_timesteps,dt, kp,ki,kd, plotty=True)
    #print("Walk Forward")
    #execute(wkf,num_timesteps,dt,kp,ki,kd,dt, plotty = True)
    print("Walk Backward")
    execute(bk,num_timesteps,dt,kp,ki,kd,dt, plotty = True)
