import numpy as np
import mujoco
import casadi as cs
from helpers.util import Util
from helpers.mpc_config import MPCConfig
from helpers.kinematics import KinematicsHelper
from helpers.PID_Controller import PID_Controller
from helpers.data_logger import DataLogger
from helpers.skills import bk, wkf, balance, bk_converge_to_30
from scipy.spatial.transform import Rotation
from helpers.petoi_kinematics import PetoiKinematics



kp = 11
kd = 1
ki = 8e-1
dt = 1e-3
num_timesteps = int(2e4)
pid_controller = PID_Controller("urdf/bittle.xml",dt=dt,
                             kp=kp,
                             ki=ki,
                             kd=kd)
gait = bk
error_threshold = 0.06
max_timesteps = 77

with mujoco.viewer.launch_passive(pid_controller.model, pid_controller.data) as viewer:
    # if os.name == 'nt':
    #     import ctypes
    #     hwnd = ctypes.windll.user32.GetForegroundWindow()
    #     ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
    print("Walk Forward")
    for t in range(num_timesteps):
        pid_controller.set_targets(np.array(gait[t%len(gait)]))
        #print(np.array(gait[t%len(gait)]))
        #pid_controller.execute(gait,500,dt,kp,ki,kd, viewer=viewer, plotty=False)
        for step in range(max_timesteps):
            error = pid_controller.step(viewer)
            if np.all((error) < error_threshold):
                print(f"Converged in {step} steps at timestep {t}")
                break
        