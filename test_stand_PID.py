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

pid_controller = PID_Controller("urdf/bittle.xml")
kp = 1e5
ki = 50
kd = 100
dt = 0.001
num_timesteps = int(1.5e4)
with mujoco.viewer.launch_passive(pid_controller.model, pid_controller.data) as viewer:
    if os.name == 'nt':
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
    print("Stand")
    for i in range(num_timesteps):
        pid_controller.execute(balance[0],1,dt, kp,ki,kd, viewer=viewer, plotty=False)