import casadi as cs
import tinympc as MPC
import numpy as np
from .petoi_kinematics import PetoiKinematics

class MPCConfig:
    def __init__(self, N=10, dt=0.1):
        self.N = N
        self.dt = dt
        self.mpc = MPC.TinyMPC()
        self.setup_mpc()
        self.petoi = PetoiKinematics()
    
    def setup_mpc(self):
        """Configure MPC parameters, costs, and constraints"""
        # Basic MPC parameters
       
        self.mpc.N = self.N          # Prediction horizon
        self.mpc.nx = 22             # Number of states (same as your nx)
        self.mpc.nu = 8              # Number of controls (same as your nu)

        # Cost matrices (Q and R from your original code)
        self.Q = np.diag([
            1e4, 1e8, 1e1,       # [x, z, yaw]
            1e3, 1e3, 1e2,       # [dx, dz, dyaw]
            *[1e2]*8,            # Joint angles (q1-q8)
            *[1e-1]*8            # Joint velocities (dq1-dq8)
        ])
        
        self.R = np.diag([
            1e3 * 1.0,   # left-back-shoulder
            1e3 * 0.01,  # left-back-knee
            1e3 * 1.0,   # left-front-shoulder
            1e3 * 0.01,  # left-front-knee
            1e3 * 1.0,   # right-back-shoulder
            1e3 * 0.01,  # right-back-knee
            1e3 * 1.0,   # right-front-shoulder
            1e3 * 0.01   # right-front-knee
        ])
        
        self.mpc.Q = self.Q
        self.mpc.R = self.R
        self.mpc.Qf = 10 * self.Q  # Terminal cost

        # State constraints (time-invariant)
        self.mpc.x_min = -np.inf * np.ones(self.mpc.nx)
        self.mpc.x_max = np.inf * np.ones(self.mpc.nx)
        
        # Joint angle limits from XML
        joint_limits = np.array([
            [-1.5708, 1.22173],   # left-back-shoulder
            [-1.22173, 1.48353],  # left-back-knee
            [-1.5708, 1.22173],   # left-front-shoulder
            [-1.22173, 1.48353],  # left-front-knee
            [-1.5708, 1.22173],   # right-back-shoulder
            [-1.22173, 1.48353],  # right-back-knee
            [-1.5708, 1.22173],   # right-front-shoulder
            [-1.22173, 1.48353]   # right-front-knee
        ])
        self.joint_limits = joint_limits
        # Apply joint angle constraints (indices 6-13 in state vector)
        self.mpc.x_min[6:14] = joint_limits[:, 0]
        self.mpc.x_max[6:14] = joint_limits[:, 1]

        # Control constraints (torque limits from XML)
        self.mpc.u_min = -0.75 * np.ones(self.mpc.nu)
        self.mpc.u_max = 0.75 * np.ones(self.mpc.nu)
    
    def create_dynamics(self):
        """Full dynamics model matching XML parameters
        State (22 elements):
            [0:3] CoM position (x, z, yaw)
            [3:6] CoM velocity (dx, dz, dyaw)
            [6:14] Joint angles (8)
            [14:22] Joint velocities (8)
        Control (8 elements): Joint torques (limited to ±0.75 Nm)
        """
        petoi = PetoiKinematics()
        
        # State and control variables
        x = cs.MX.sym('x', 22)
        u = cs.MX.sym('u', 8)

        # XML-derived parameters
        total_mass = 0.165  # From root body inertial
        com_inertia = 0.001  # From root body diaginertia (izz)
        
        # Joint parameters from XML
        joint_inertias = np.array([
            # Shoulders (0.00044) and knees (0.00063) alternating
            0.00044, 0.00063, 0.00044, 0.00063,
            0.00044, 0.00063, 0.00044, 0.00063
        ])
        
        joint_damping = 0.01  # From <joint damping="0.01">
        torque_limit = 0.75   # From actuator forcerange

        # Unpack state variables
        com_pos = x[0:3]        # [x, z, theta]
        com_vel = x[3:6]        # [dx, dz, dtheta]
        joint_angles = x[6:14]  # q1-q8
        joint_vel = x[14:22]    # dq1-dq8

        # 1. CoM Dynamics ------------------------------------------
        foot_positions = []
        for i in range(4):
            alpha = joint_angles[i*2]
            beta = joint_angles[i*2+1]
            
            # XML-based leg geometry
            L = petoi.leg_length + petoi.foot_length * cs.sin(beta)
            x_foot = L * cs.sin(alpha)
            z_foot = -L * cs.cos(alpha)
            
            # Transform using XML-measured offsets
            if i in [0, 3]:  # front legs
                T = petoi.T01_front @ cs.vertcat(x_foot, z_foot, 1)
            else:  # back legs
                T = petoi.T01_back @ cs.vertcat(x_foot, z_foot, 1)
            foot_positions.append(T[:2])

        # Ground reaction model
        total_force = cs.MX.zeros(2)
        total_torque = 0.0
        k_ground = 1500
        c_ground = 75
        
        for fp in foot_positions:
            penetration = cs.fmax(0 - fp[1], 0)
            f_z = k_ground * penetration - c_ground * com_vel[1]
            f_x = -0.25 * f_z * cs.tanh(com_vel[0] * 15)
            lever_arm = fp - com_pos[:2]
            total_torque += lever_arm[0] * f_z - lever_arm[1] * f_x
            total_force += cs.vertcat(f_x, f_z)
        
        # CoM accelerations
        com_acc = cs.vertcat(
            total_force[0]/total_mass,
            total_force[1]/total_mass - 9.81,
            total_torque / com_inertia
        )

        # 2. Joint Dynamics ----------------------------------------
        # Element-wise operations for joint accelerations
        joint_acc = (cs.fmin(cs.fmax(u, -torque_limit), torque_limit)  # Apply torque limits
                    - joint_damping * joint_vel) / joint_inertias

        # 3. Full State Derivative ---------------------------------
        dxdt = cs.vertcat(
            com_vel[0],        # dx/dt
            com_vel[1],        # dz/dt
            com_vel[2],        # dtheta/dt
            com_acc[0],        # ddx/dt
            com_acc[1],        # ddz/dt
            com_acc[2],        # ddtheta/dt
            joint_vel,         # dq/dt
            joint_acc          # ddq/dt
        )
        
        return cs.Function('dynamics', [x, u], [dxdt], ['x', 'u'], ['dxdt'])
   
    def linearize_dynamics(f, nx=22, nu=8):
        x = cs.MX.sym("x", nx)
        u = cs.MX.sym("u", nu)

        # Evaluate dynamics symbolically
        dx = f(x, u)

        # Compute Jacobians
        A = cs.jacobian(dx, x)
        B = cs.jacobian(dx, u)

        # Turn them into CasADi functions
        A_func = cs.Function("A", [x, u], [A])
        B_func = cs.Function("B", [x, u], [B])

        return A_func, B_func
    
  
