import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on your setup

import os

from skills import balance, wkf, bk

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
    
    num_joints = 8
    angle_holder = np.zeros((num_timesteps, num_joints))
    reference_holder = np.zeros((num_timesteps, num_joints))
    actuator_nums = [3,7,0,4,2,6,1,5]
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
