import mujoco
import mujoco.viewer
import numpy as np
import time

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
  [18,  54,  58,  52,   7,  13,  -2,   9],
  [14,  56,  52,  54,  12,  14,  -3,  10],
  [15,  57,  46,  55,  17,  15,  -4,  11],
  [16,  58,  38,  57,  16,  17,  -2,  12],
  [19,  59,  31,  58,  13,  19,   1,  14],
  [22,  59,  24,  60,  12,  22,   6,  14],
  [26,  60,  21,  61,  10,  24,  12,  16],
  [28,  58,  23,  62,   9,  30,  16,  18],
  [31,  61,  25,  63,   8,  30,  13,  20],
  [34,  67,  28,  63,   7,  23,  12,  24],
  [36,  69,  31,  63,   6,  15,  11,  27],
  [38,  68,  34,  63,   7,   8,  10,  31],
  [41,  65,  36,  62,   7,   3,   9,  35],
  [43,  60,  39,  65,   7,  -1,   9,  37],
  [46,  55,  41,  70,   7,  -3,   9,  27],
  [47,  50,  43,  71,   7,  -5,   9,  23],
  [49,  43,  47,  71,   8,  -5,   7,  14],
  [51,  35,  48,  69,   8,  -3,   8,   8],
  [53,  28,  51,  66,   9,   1,   8,   2],
  [54,  18,  52,  63,  12,   7,   9,  -2],
  [55,  15,  54,  56,  14,  10,  10,  -3],
  [56,  13,  55,  51,  15,  16,  11,  -4],
  [57,  16,  57,  43,  17,  16,  12,  -3],
  [58,  18,  58,  36,  19,  14,  14,  -2],
  [59,  21,  60,  28,  21,  12,  14,   2],
  [60,  25,  61,  23,  24,  10,  16,   9],
  [60,  28,  62,  22,  27,   9,  18,  15],
  [60,  31,  63,  23,  30,   8,  20,  16],
  [66,  32,  63,  26,  24,   7,  24,  13],
  [68,  35,  63,  29,  19,   6,  27,  11],
  [68,  38,  63,  32,  10,   6,  31,  10],
  [66,  40,  62,  35,   4,   7,  35,  10],
  [62,  42,  65,  37,   0,   7,  37,   9],
  [57,  45,  70,  39,  -2,   7,  27,   9],
  [52,  47,  71,  41,  -5,   7,  23,   9],
  [45,  48,  71,  44,  -5,   8,  14,   9],
  [38,  51,  69,  47,  -3,   8,   8,   7],
  [30,  52,  66,  49,  -1,   9,   2,   8],
  [22,  53,  63,  51,   4,  12,  -2,   8]
]

max_index = len(wkf)
index = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Iterate over each actuator
        for i in range(model.nu):
            actuator_name = actuator_names[i]
            qpos_idx = actuator_to_qpos[actuator_name]
            if actuator_name == "left-back-shoulder-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][3])
            elif actuator_name == "left-back-knee-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][7])
            elif actuator_name == "left-front-shoulder-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][0])
            elif actuator_name == "left-front-knee-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][4])
            elif actuator_name == "right-back-shoulder-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][2] - 90)
            elif actuator_name == "right-back-knee-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][6])
            elif actuator_name == "right-front-shoulder-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][1] - 90)
            elif actuator_name == "right-front-knee-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][5])
        index += 1
        #time.sleep(0.1)
        if index == max_index:
            index = 0


        mujoco.mj_step(model, data)
        viewer.sync()
