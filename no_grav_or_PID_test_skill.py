import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model and data
model = mujoco.MjModel.from_xml_path("urdf/bittle_nograv.xml")
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
  [45,  59,  47,  58, -10,  -4,  -5,  -4],
  [47,  59,  47,  59,  -5,  -4,  -5,  -4],
  [50,  60,  48,  59,  -5,  -3,  -5,  -4],
  [50,  60,  48,  59,  -5,  -3,  -5,  -4],
  [50,  61,  49,  60,  -5,  -3,  -5,  -3],
  [51,  61,  49,  60,  -5,  -3,  -5,  -3],
  [51,  61,  50,  62,  -5,  -3,  -5,  -2],
  [52,  61,  50,  68,  -5,  -3,  -5,  -5],
  [52,  61,  50,  73,  -5,  -3,  -5, -10],
  [52,  62,  51,  79,  -5,  -2,  -5, -17],
  [52,  62,  51,  83,  -5,  -2,  -5, -23],
  [52,  62,  52,  87,  -5,  -2,  -5, -31],
  [53,  63,  52,  86,  -5,  -2,  -5, -37],
  [53,  63,  52,  84,  -5,  -2,  -5, -41],
  [54,  64,  52,  79,  -5,   0,  -5, -44],
  [54,  70,  53,  79,  -5,  -2,  -5, -44],
  [54,  76,  53,  79,  -5,  -7,  -5, -44],
  [55,  81,  53,  78,  -5, -14,  -5, -44],
  [55,  85,  54,  71,  -5, -21,  -5, -43],
  [56,  91,  54,  62,  -5, -29,  -5, -40],
  [56,  91,  55,  54,  -5, -35,  -5, -35],
  [56,  89,  55,  48,  -5, -40,  -5, -29],
  [56,  84,  56,  44,  -4, -43,  -5, -21],
  [56,  84,  56,  42,  -4, -43,  -5, -14],
  [57,  84,  56,  41,  -4, -43,  -4,  -9],
  [57,  84,  56,  43,  -4, -43,  -4,  -5],
  [58,  77,  56,  45,  -4, -43,  -4,  -5],
  [58,  69,  57,  45,  -4, -40,  -4,  -5],
  [58,  61,  57,  45,  -4, -36,  -4,  -5],
  [58,  55,  58,  46,  -4, -30,  -4,  -5],
  [58,  49,  58,  46,  -4, -23,  -4,  -5],
  [59,  46,  58,  47,  -4, -15,  -4,  -5],
  [59,  45,  58,  47,  -4, -10,  -4,  -5],
  [59,  47,  59,  48,  -4,  -5,  -4,  -5],
  [60,  50,  59,  48,  -3,  -5,  -4,  -5],
  [60,  50,  59,  48,  -3,  -5,  -4,  -5],
  [61,  50,  60,  49,  -3,  -5,  -3,  -5],
  [61,  51,  60,  49,  -3,  -5,  -3,  -5],
  [61,  51,  62,  50,  -3,  -5,  -2,  -5],
  [61,  52,  68,  50,  -3,  -5,  -5,  -5],
  [61,  52,  73,  51,  -3,  -5, -10,  -5],
  [62,  52,  79,  51,  -2,  -5, -17,  -5],
  [62,  52,  83,  52,  -2,  -5, -23,  -5],
  [63,  52,  87,  52,  -2,  -5, -31,  -5],
  [63,  53,  86,  52,  -2,  -5, -37,  -5],
  [63,  53,  84,  52,  -1,  -5, -41,  -5],
  [67,  54,  79,  52,   0,  -5, -44,  -5],
  [73,  54,  79,  53,  -4,  -5, -44,  -5],
  [78,  54,  79,  53, -10,  -5, -44,  -5],
  [83,  55,  78,  54, -17,  -5, -44,  -5],
  [89,  55,  71,  54, -25,  -5, -43,  -5],
  [90,  56,  62,  55, -31,  -5, -40,  -5],
  [91,  56,  54,  55, -37,  -5, -35,  -5],
  [87,  56,  48,  55, -41,  -5, -29,  -5],
  [84,  56,  44,  56, -43,  -4, -21,  -5],
  [84,  56,  42,  56, -43,  -4, -14,  -5],
  [84,  57,  41,  56, -43,  -4,  -9,  -4],
  [81,  57,  43,  56, -43,  -4,  -5,  -4],
  [73,  58,  45,  57, -42,  -4,  -5,  -4],
  [64,  58,  45,  57, -39,  -4,  -5,  -4],
  [57,  58,  45,  58, -33,  -4,  -5,  -4],
  [52,  58,  46,  58, -26,  -4,  -5,  -4],
  [48,  58,  46,  58, -18,  -4,  -5,  -4],
  [46,  59,  47,  58, -12,  -4,  -5,  -4],
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
