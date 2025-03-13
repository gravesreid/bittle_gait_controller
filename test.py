import mujoco
import mujoco.viewer
import numpy as np

# Load model and data
model = mujoco.MjModel.from_xml_path("urdf/bittle.xml")
data = mujoco.MjData(model)

# PD controller gains
kp = 100.0
kd = 2.0

# Define desired joint angles (radians)
# Order should correspond to the actuator order defined in your XML
desired_angles = np.array([-np.pi, 0, 0, 0, 0, 0, 0, 0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # For each actuator, compute the control command
        print(f'qpos: {data.qpos}')
        print(f'qvel: {data.qvel}')
        for i in range(8):
            # Retrieve the current joint angle and velocity.
            current_angle = data.qpos[i]
            current_vel = data.qvel[i]
            error = desired_angles[i] - current_angle
            data.ctrl[i] = kp * error - kd * current_vel
        mujoco.mj_step(model, data)
        viewer.sync()
