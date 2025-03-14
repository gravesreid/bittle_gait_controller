import mujoco
import mujoco.viewer
import numpy as np

# Load model and data
model = mujoco.MjModel.from_xml_path("urdf/bittle.xml")
data = mujoco.MjData(model)

# Build mappings for qpos and qvel indices for each actuator/joint
actuator_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    for i in range(model.nu)
]
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

# PD controller gains
kp = 100.01
kd = .1

# Define desired angles in degrees for each actuator
desired_angles_deg = {
    "left-back-shoulder-joint": -2.5,
    "left-back-knee-joint": 8.0,
    "left-front-shoulder-joint": 1.5,
    "left-front-knee-joint": -12.0,
    "right-back-shoulder-joint": -4.5,
    "right-back-knee-joint": 12.0,
    "right-front-shoulder-joint": 4.5,
    "right-front-knee-joint": -8.0,
}

# Convert desired angles from degrees to radians
desired_angles_rad = {
    name: np.deg2rad(angle) for name, angle in desired_angles_deg.items()
}

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Iterate over each actuator
        for i in range(model.nu):
            actuator_name = actuator_names[i]
            print("Actuator name:", actuator_name)
            qpos_idx = actuator_to_qpos[actuator_name]
            qvel_idx = actuator_to_qvel[actuator_name]
            # Use qpos for angle and qvel for velocity
            current_angle = data.qpos[qpos_idx]
            # This bypasses the PD controller and sets the desired angle directly
            data.qpos[qpos_idx] = desired_angles_rad[actuator_name]
            current_angle_degrees = np.rad2deg(current_angle)
            print("Current angle (degrees):", current_angle_degrees)
            current_vel = data.qvel[qvel_idx]
            error = desired_angles_rad[actuator_name] - current_angle
            error_degrees = np.rad2deg(error)
            print("Error (degrees):", error_degrees)
            # this uses PD control instead of directly setting the desired angle
            #data.ctrl[i] = kp * error - kd * current_vel

        mujoco.mj_step(model, data)
        viewer.sync()
