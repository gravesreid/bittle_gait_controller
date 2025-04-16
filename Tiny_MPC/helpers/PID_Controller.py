import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on your setup

class PID_Controller:
    def __init__(self, xml_path):
        # Load self.model and self.data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # initialize imu
        # Get the sensor IDs
        self.imu_orientation_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )
        self.imu_gyro_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity"
        )

        # Get the addresses (start indices) in the sensordata array
        self.imu_orientation_address = self.model.sensor_adr[self.imu_orientation_sensor_id]
        self.imu_gyro_address = self.model.sensor_adr[self.imu_gyro_sensor_id]

        # Get the dimensions
        self.imu_orientation_dim = self.model.sensor_dim[self.imu_orientation_sensor_id]  # Should be 4
        self.imu_gyro_dim = self.model.sensor_dim[self.imu_gyro_sensor_id]  # Should be 3


        # Build mappings for qpos and qvel indices for each actuator/joint
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        print("Actuator names:", self.actuator_names)
        self.actuator_to_qpos = {}
        self.actuator_to_qvel = {}

        for name in self.actuator_names:
            # Get the joint id by name
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            # qpos index for configuration (e.g., angle) and qvel index for velocity
            qpos_index = self.model.jnt_qposadr[joint_id]
            qvel_index = self.model.jnt_dofadr[joint_id]
            self.actuator_to_qpos[name] = qpos_index
            self.actuator_to_qvel[name] = qvel_index

        print("Actuator to qpos mapping:", self.actuator_to_qpos)
        print("Actuator to qvel mapping:", self.actuator_to_qvel)



    def execute(self, behavior, num_timesteps, dt, kp, ki, kd, viewer, clipped_control = False, limits = [0,0], plotty =False):
        joint_to_qpos = {}
        for name in self.actuator_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_to_qpos[name] = self.model.jnt_qposadr[joint_id]
        actuator_map = {
            3:"left-back-shoulder-joint",
            7:"left-back-knee-joint",
            0:"left-front-shoulder-joint",
            4:"left-front-knee-joint",
            2:"right-back-shoulder-joint",
            6:"right-back-knee-joint",
            1:"right-front-shoulder-joint",
            5:"right-front-knee-joint",

        }
        actuator_to_ctrl = {name: i for i, name in enumerate(self.actuator_names)}
        #print("Actuator to ctrl mapping:", actuator_to_ctrl)
        e = 10000
        error_vec = 100*np.ones(8)
        prev_error_vec = np.zeros(8)
        int_error_vec = np.zeros(8)
        
        num_joints = 8
        angle_holder = np.zeros((num_timesteps, num_joints))
        reference_holder = np.zeros((num_timesteps, num_joints))
        actuator_nums = [3,7,0,4,2,6,1,5]
        index = 0


        for i in range(num_timesteps):
            
            desired_angles = np.array([np.deg2rad(behavior[num]) for num in actuator_nums])
            
            
            # Calculate errors
            error_vec = desired_angles - np.array([self.data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums])
            int_error_vec += error_vec*dt
            de_dt_vec = (error_vec - prev_error_vec)/dt

            # PID control - single update per timestep
            for j, num in enumerate(actuator_nums):
                e = error_vec[j]
                de_dt = de_dt_vec[j]
                int_e = int_error_vec[j]

                ctrl = kp*e + ki*int_e + kd*de_dt
                #print("Control signal:", ctrl)

                # PID control with clipping
                if clipped_control == True:
                    ctrl = np.clip(ctrl,limits[0],limits[1])
                
                self.data.ctrl[actuator_to_ctrl[actuator_map[num]]] = ctrl
                prev_error_vec[j] = e 
            
            # Single simulation step
            mujoco.mj_step(self.model, self.data)
            viewer.sync()


            if i % 50 == 0:
                index = (index + 1) % len(behavior)

            
            # After convergence (or reaching max iterations), store final angles:
            angle_holder[i, :] = [self.data.qpos[joint_to_qpos[actuator_map[num]]] for num in actuator_nums]
            reference_holder[i, :] = desired_angles

            #self.data.ctrl[actuator_to_ctrl["left-back-knee-joint"]] = np.clip(np.deg2rad(wkf[index][7]) / np.pi, -1, 1)
            #index += 1
            #time.sleep(0.005)
            
        if plotty:  
            # Create time array
            time_array = np.arange(num_timesteps) * dt

            # Create subplots for all joints
            plt.figure(figsize=(15, 20))

            # Get joint names from actuator_map for titles
            joint_names = [actuator_map[num] for num in actuator_nums]

            for j in range(num_joints):
                plt.subplot(4, 2, j+1)
                plt.plot(time_array, angle_holder[:, j], label='Actual')
                plt.plot(time_array, reference_holder[:, j], '--', label='Reference')
                plt.xlabel('Time (s)')
                plt.ylabel('Angle (rad)')
                plt.title(f'{joint_names[j]} Trajectory')
                plt.grid(True)
                plt.legend()

            plt.tight_layout()
            plt.show()
            plt.figure(figsize=(15, 20))
