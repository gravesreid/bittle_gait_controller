import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on your setup

import os

from skills import balance
from PID_Controller import PID_Controller

test = [[-60, 60, 31, 13, 0, 7, 18, 40]]


pid_controller = PID_Controller("urdf/bittle.xml")
orientation, gyro = pid_controller.get_imu_readings()
kp = 1e5
ki = 50
kd = 100
dt = 0.001
num_timesteps = int(20000)
with mujoco.viewer.launch_passive(pid_controller.model, pid_controller.data) as viewer:
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
    print("Stand")
    #pid_controller.execute(balance,num_timesteps,dt, kp,ki,kd, viewer=viewer, plotty=True)
    pose_sequence = [
    [0, 30, 0, 30, 0, 30, 0, 30],  # pose 1
    [10, 20, 10, 20, 10, 20, 10, 20],  # pose 2
    [5, 25, 5, 25, 5, 25, 5, 25]   # pose 3
]
    #pid_controller.track_pose_sequence(pose_sequence, hold_time=5, dt=0.01, kp=10, ki=0.5, kd=1, viewer=viewer)
    pid_controller.execute(pose_sequence,num_timesteps,dt, kp,ki,kd, viewer=viewer, plotty=True)
