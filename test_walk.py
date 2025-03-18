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
  [ 9,  49,  53,  45,  24,  20,  -2,  15],
  [ 8,  50,  41,  46,  28,  21,  -1,  15],
  [10,  51,  26,  47,  26,  22,   6,  16],
  [12,  52,  23,  48,  24,  24,   9,  17],
  [14,  52,  20,  49,  22,  26,  12,  18],
  [16,  53,  17,  51,  21,  27,  17,  18],
  [18,  53,  14,  52,  20,  29,  22,  19],
  [21,  54,  11,  54,  18,  30,  27,  19],
  [22,  54,  11,  54,  18,  32,  29,  20],
  [25,  54,  13,  55,  16,  34,  27,  21],
  [26,  54,  16,  56,  16,  37,  24,  23],
  [28,  54,  18,  56,  15,  39,  23,  24],
  [30,  52,  20,  57,  14,  45,  22,  26],
  [32,  54,  22,  57,  14,  44,  21,  28],
  [33,  58,  24,  57,  15,  36,  20,  30],
  [34,  61,  26,  57,  15,  31,  19,  32],
  [36,  64,  28,  57,  14,  24,  18,  35],
  [38,  66,  29,  57,  14,  20,  18,  38],
  [39,  67,  31,  57,  14,  16,  17,  40],
  [41,  64,  32,  56,  14,   5,  17,  43],
  [42,  55,  35,  57,  14,  -1,  16,  44],
  [44,  44,  37,  62,  15,  -3,  14,  35],
  [45,  30,  39,  66,  15,   1,  14,  29],
  [46,  21,  40,  68,  15,   5,  14,  23],
  [47,  19,  42,  70,  16,   9,  14,  19],
  [48,  16,  43,  70,  17,  12,  15,  17],
  [49,  12,  44,  67,  18,  17,  15,   5],
  [49,   9,  46,  59,  20,  24,  15,  -2],
  [50,   8,  47,  47,  21,  28,  16,  -2],
  [51,  10,  48,  34,  22,  26,  16,   1],
  [52,  12,  49,  24,  24,  24,  17,   6],
  [52,  14,  50,  21,  26,  22,  18,  10],
  [53,  16,  51,  19,  27,  21,  19,  12],
  [53,  18,  52,  15,  29,  20,  20,  19],
  [54,  21,  54,  12,  30,  18,  19,  24],
  [54,  22,  55,  12,  32,  18,  20,  27],
  [54,  25,  55,  11,  34,  16,  22,  29],
  [54,  26,  56,  14,  37,  16,  24,  26],
  [54,  28,  56,  17,  39,  15,  25,  24],
  [52,  30,  57,  18,  45,  14,  27,  23],
  [54,  32,  57,  21,  44,  14,  29,  21],
  [58,  33,  57,  23,  36,  15,  31,  20],
  [61,  34,  57,  24,  31,  15,  33,  20],
  [64,  36,  57,  26,  24,  14,  36,  19],
  [66,  38,  57,  28,  20,  14,  39,  18],
  [67,  39,  56,  30,  16,  14,  42,  17],
  [64,  41,  56,  32,   5,  14,  45,  17],
  [55,  42,  59,  33,  -1,  14,  41,  17],
  [44,  44,  64,  35,  -3,  15,  33,  16],
  [30,  45,  67,  38,   1,  15,  27,  14],
  [21,  46,  68,  39,   5,  15,  22,  14],
  [19,  47,  70,  41,   9,  16,  18,  14],
  [16,  48,  69,  42,  12,  17,  11,  14],
  [12,  49,  63,  44,  17,  18,   1,  15]
]

for frame in wkf:
    # Right-back-shoulder-joint is index 2 in actuator_nums (original index 2)
    # Right-front-shoulder-joint is index 1 in actuator_nums (original index 1)
    frame[1] *= -1  # Right-front-shoulder
    frame[2] *= -1  # Right-back-shoulder
    frame[5] *= -1  # Right-front-knee
    frame[6] *= -1  # Right-back-knee

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
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][2])
            elif actuator_name == "right-back-knee-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][6])
            elif actuator_name == "right-front-shoulder-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][1])
            elif actuator_name == "right-front-knee-joint":
                data.qpos[qpos_idx] = np.deg2rad(wkf[index][5])
        index += 1
        time.sleep(0.01)
        if index == max_index:
            index = 0


        mujoco.mj_step(model, data)
        viewer.sync()
