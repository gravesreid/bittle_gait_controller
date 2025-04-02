import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on your setup

import os

from skills import balance, wkf, bk
from PID import PID_Controller

pid_controller = PID_Controller()

kp = 6e4
ki = 5e2
kd = 5e1
dt = 1e-3
num_timesteps = int(1e4)
with mujoco.viewer.launch_passive(pid_controller.model, pid_controller.data) as viewer:
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
    #print("Stand")
    #execute(balance,num_timesteps,dt, kp,ki,kd, plotty=True)
    #print("Walk Forward")
    #execute(wkf,num_timesteps,dt,kp,ki,kd, plotty = True)
    print("Walk Backward")
    pid_controller.execute(bk,num_timesteps,dt,kp,ki,kd, viewer=viewer, plotty=True)
