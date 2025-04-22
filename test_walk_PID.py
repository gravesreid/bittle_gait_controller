import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on your setup

import os

from skills import wkf, bk
from PID_Controller import PID_Controller

pid_controller = PID_Controller("urdf/bittle.xml")

kp = 1e2
ki = 5e-1
kd = 5e-1
dt = 1e-3
num_timesteps = int(2e4)

with mujoco.viewer.launch_passive(pid_controller.model, pid_controller.data) as viewer:
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
    print("Walk Forward")
    for t in range(num_timesteps):
        pid_controller.execute(bk[t%len(bk)],1,dt,kp,ki,kd, viewer=viewer, plotty=False)