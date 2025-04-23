import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on your setup

class PID_Controller:
    def __init__(self, xml_path, dt, kp=1.0, ki=0.0, kd=0.1):
        # Load self.model and self.data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.t = 0
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.clipped_control = False
        self.limits = [0,0]

        self.targets = np.zeros((1,8))
        self.prev_error_vec = np.zeros(8)
        self.int_error_vec = np.zeros(8)

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
        self.actuator_nums = [3,7,0,4,2,6,1,5]
        self.jointPos_to_qpos = {}
        self.jointVel_to_qpos = {}
        for name in self.actuator_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.jointPos_to_qpos[name] = self.model.jnt_qposadr[joint_id]
            self.jointVel_to_qpos[name] = self.model.jnt_dofadr[joint_id]
        self.actuator_map = {
            3:"left-back-shoulder-joint",
            7:"left-back-knee-joint",
            0:"left-front-shoulder-joint",
            4:"left-front-knee-joint",
            2:"right-back-shoulder-joint",
            6:"right-back-knee-joint",
            1:"right-front-shoulder-joint",
            5:"right-front-knee-joint",
        }
        self.actuator_to_ctrl = {name: i for i, name in enumerate(self.actuator_names)}

    def get_angles(self, actuator_nums):
        # This is what this line is doing all at once:
        #   joint_names = [self.actuator_map[num] for num in actuator_nums]
        #   q_pos = [self.joint_to_qpos[name] for name in joint_names]
        #   joint_angles = [self.data.qpos[pos] for pos in q_pos]
        return np.array(self.data.qpos[[self.jointPos_to_qpos[self.actuator_map[num]] for num in actuator_nums]])

    def get_velocities(self, actuator_nums):
        # This is what this line is doing all at once:
        #   joint_names = [self.actuator_map[num] for num in actuator_nums]
        #   q_vel = [self.joint_to_qvel[name] for name in joint_names]
        #   joint_velocities = [self.data.qvel[pos] for pos in q_vel]
        return np.array(self.data.qvel[[self.jointVel_to_qpos[self.actuator_map[num]] for num in actuator_nums]])
    
    def set_targets(self, target):
        self.t = 0
        # # If target is list of lists, convert to numpy array
        # if isinstance(target, list) and all(isinstance(i, list) for i in target):
            
        # If target is 1d numpy array, convert to 2d
        if target.ndim == 1:
            target = target.reshape(1, -1)
        else:
            print("kazam")

        self.targets = target
        
    
    def step(self, viewer):
        #print("Actuator to ctrl mapping:", actuator_to_ctrl)
        e, de_dt, int_e = 10000, 100, 1
        error_vec = e*np.ones(8)
        #print(self.targets)
        # print(self.targets[self.t,0])
        
        desired_angles = np.array([np.deg2rad(self.targets[self.t,num]) for num in self.actuator_nums])
        self.t += 1
        #print(np.shape(self.targets))
        if self.t >= len(self.targets):
            self.t = 0
        #print(self.t)
        # Calculate errors
        error_vec = desired_angles - self.get_angles(self.actuator_nums)
        self.int_error_vec += error_vec*self.dt
        de_dt_vec = (error_vec - self.prev_error_vec)/self.dt
        num_timesteps = 500
        error_threshold = 0.01


        # PID control for each joint
        for j, num in enumerate(self.actuator_nums):
            e = error_vec[j]
            de_dt = de_dt_vec[j]
            int_e = self.int_error_vec[j]

            ctrl = self.kp*e + self.ki*int_e + self.kd*de_dt

            # PID control with clipping
            if self.clipped_control == True:
                ctrl = np.clip(ctrl,self.limits[0],self.limits[1])
            
            self.data.ctrl[self.actuator_to_ctrl[self.actuator_map[num]]] = ctrl
            self.prev_error_vec[j] = e
        
    
        # Single simulation step
        mujoco.mj_step(self.model, self.data)
        viewer.sync()
        return (np.abs(error_vec))
        # Exit loop if all errors are below threshold
        # if np.all(np.abs(error_vec) < error_threshold):
        #     break

    def execute(self, target, num_timesteps, dt, kp, ki, kd, viewer, clipped_control = False, limits = [0,0], plotty =False):
            #print("Actuator to ctrl mapping:", actuator_to_ctrl)
        e = 10000
        error_vec = 100*np.ones(8)
        prev_error_vec = np.zeros(8)
        int_error_vec = np.zeros(8)
        error_threshold = 0.03  # Threshold for convergence
        num_joints = 8
        angle_holder = np.zeros((num_timesteps, num_joints))
        reference_holder = np.zeros((num_timesteps, num_joints))
        actuator_nums = [3,7,0,4,2,6,1,5]
        index = 0
        time_to_get_there = 500
        for k in range(num_timesteps):
            for t in target:
                desired_angles = np.array([np.deg2rad(t[num]) for num in actuator_nums])
                for i in range(time_to_get_there):
                    # Calculate errors
                    error_vec = desired_angles - self.get_angles(actuator_nums)
                    int_error_vec += error_vec*dt
                    de_dt_vec = (error_vec - prev_error_vec)/dt

                    # PID control - single update per timestep
                    for j, num in enumerate(actuator_nums):
                        e = error_vec[j]
                        de_dt = de_dt_vec[j]
                        int_e = int_error_vec[j]

                        ctrl = kp*e + ki*int_e + kd*de_dt

                        # PID control with clipping
                        if clipped_control == True:
                            ctrl = np.clip(ctrl, limits[0], limits[1])
                        
                        self.data.ctrl[self.actuator_to_ctrl[self.actuator_map[num]]] = ctrl
                        prev_error_vec[j] = e 
                    
                    # Single simulation step
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                    if i % len(target) == 0:
                        index = (index + 1) % len(target)

                    
                    # After convergence (or reaching max iterations), store final angles:
                    # angle_holder[i, :] = self.get_angles(actuator_nums)
                    # reference_holder[i, :] = desired_angles

                    # Exit loop if all errors are below threshold
                    if np.all(np.abs(error_vec) < error_threshold):
                        break
            
        if plotty:  
            # Create time array
            time_array = np.arange(num_timesteps) * dt

            # Create subplots for all joints
            plt.figure(figsize=(15, 20))

            # Get joint names from actuator_map for titles
            joint_names = [self.actuator_map[num] for num in actuator_nums]

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
