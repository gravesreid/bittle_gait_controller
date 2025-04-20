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
        joint_angles_sym = cs.MX.sym('joint_angles', 8)
        tau_sym = cs.MX.sym('tau', 8)

        grf_expr = self.joints_to_GRF(joint_angles_sym, tau_sym)

        self.joints_to_GRF_func = cs.Function('joints_to_GRF_func', [joint_angles_sym, tau_sym], [grf_expr])
    def joints_to_GRF(self, joint_angles, tau):
        grf = cs.MX.zeros((4, 2))  # Initialize GRFs array for 4 legs

        for i in range(4):
            alpha = joint_angles[2*i]
            beta = joint_angles[2*i + 1]

            # Get Jacobian function for the leg
            J_func = self.leg_jacobian(cs.MX.sym('alpha'), cs.MX.sym('beta'))

            # Evaluate Jacobian at current joint angles
            J_current = J_func(alpha, beta)

            # Extract joint torques for this leg
            tau_leg = cs.vertcat(tau[2*i], tau[2*i + 1])

            # Compute pseudoinverse of Jacobian transpose
            Jt_pinv = cs.pinv(J_current.T)

            # Compute GRF for the leg: F = (J^T)^dagger * tau
            F_leg = cs.mtimes(Jt_pinv, tau_leg)

            # Store computed GRF
            grf[i, :] = F_leg.T

        return grf

    def leg_jacobian(self, alpha, beta):
        petoi = PetoiKinematics()
        """Symbolic Jacobian for a single leg"""
        L1 = petoi.upper_length
        L2 = petoi.lower_length
        
        # Foot position equations
        x_foot = L1*cs.sin(alpha) + L2*cs.sin(alpha + beta)
        y_foot = -L1*cs.cos(alpha) - L2*cs.cos(alpha + beta)
        
        # Create Jacobian matrix
        J = cs.jacobian(cs.vertcat(x_foot, y_foot), cs.vertcat(alpha, beta))
        return cs.Function('leg_jacobian', [alpha, beta], [J])
    def setup_mpc(self):
        """Configure MPC parameters, costs, and constraints"""
        # Basic MPC parameters
       
        self.mpc.N = self.N          # Prediction horizon
        self.mpc.nx = 22             # Number of states (same as your nx)
        self.mpc.nu = 8              # Number of controls (same as your nu)

        # In the MPC setup section, modify these parameters:
        self.Q = np.diag([1e3, 1e3, 1e2,  # COM x,z,yaw
                                1e1, 1e1, 1e0,   # COM velocities
                                1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8,  # Joint angles 
                                1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1]) # Joint velocities

        self.R = np.diag([1e-1]*8)  # Reduced control cost
        
        self.mpc.Q = self.Q
        self.R = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) 
        self.R[1::2] = 0.05  # fy components
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

        # GRF Constraints
        mu = 0.8  # Friction coefficient
        self.mpc.u_min = [-np.inf, 0, -np.inf, 0, -np.inf, 0, -np.inf, 0]  # fy >= 0
        self.mpc.u_max = [np.inf]*8

        # Add friction cone constraints |fx| <= μ*fy for each leg
        # Use linear inequality constraints: 
        #   fx - μ*fy <= 0
        #  -fx - μ*fy <= 0
        A_ineq = np.zeros((8, 8))
        b_ineq = np.zeros(8)
        for i in range(4):
            # fx_i <= μ*fy_i
            A_ineq[2*i, 2*i] = 1
            A_ineq[2*i, 2*i + 1] = -mu
            # -fx_i <= μ*fy_i
            A_ineq[2*i + 1, 2*i] = -1
            A_ineq[2*i + 1, 2*i + 1] = -mu

        self.mpc.A_ineq = A_ineq
        self.mpc.b_ineq = b_ineq

    def create_dynamics(self):
        """Dynamics model using GRFs as controls"""
        petoi = PetoiKinematics()
        x = cs.MX.sym('x', 22)
        u = cs.MX.sym('u', 8)  # GRFs: [fx1, fy1, ..., fx4, fy4]

        # XML-derived parameters
        total_mass = 0.165
        com_inertia = 0.001
        joint_inertias = np.array([0.00044, 0.00063, 0.00044, 0.00063,
                                0.00044, 0.00063, 0.00044, 0.00063])
        joint_damping = 0.01

        def continuous_dynamics(x, u):
            com_pos = x[0:3]
            com_vel = x[3:6]
            joint_angles = x[6:14]
            joint_vel = x[14:22]

            # 1. Compute joint torques from GRFs using Jacobian transpose
            grf = u.reshape((4, 2))  # Reshape to [fx1, fy1; ...; fx4, fy4]
            tau = cs.MX.zeros(8)

            for i in range(4):
                # Get current joint angles
                alpha = joint_angles[2*i]
                beta = joint_angles[2*i + 1]
                
                # Get precomputed Jacobian function
                J_func = self.leg_jacobian(cs.MX.sym('alpha'), cs.MX.sym('beta'))
                
                # Evaluate at current angles
                J_current = J_func(alpha, beta)
                
                # Compute torques: τ = J^T * F
                F_leg = grf[i, :]
                tau_leg = cs.mtimes(J_current.T, F_leg.T)
                tau[2*i] += tau_leg[0]
                tau[2*i + 1] += tau_leg[1]

            # 2. CoM dynamics from GRFs
            total_force = cs.sum1(grf)  # Sum all GRFs
            total_torque = 0.0

            # Compute torque from GRFs about CoM
            for i in range(4):
                alpha = joint_angles[2*i]
                beta = joint_angles[2*i + 1]
                L1 = petoi.upper_length
                L2 = petoi.lower_length
                x_foot = L1 * cs.sin(alpha) + L2 * cs.sin(alpha + beta)
                y_foot = -L1 * cs.cos(alpha) - L2 * cs.cos(alpha + beta)
                lever_arm = cs.vertcat(x_foot, y_foot)
                total_torque += lever_arm[0] * grf[i, 1] - lever_arm[1] * grf[i, 0]

            com_acc = cs.vertcat(
                total_force[0]/total_mass,
                total_force[1]/total_mass - 9.81,
                total_torque / com_inertia
            )

            # 3. Joint dynamics
            joint_acc = (tau - joint_damping * joint_vel) / joint_inertias

            return cs.vertcat(
                com_vel[0], com_vel[1], com_vel[2],
                com_acc[0], com_acc[1], com_acc[2],
                joint_vel,
                joint_acc
            )

        # Create continuous dynamics function
        f_continuous = cs.Function('cont_dyn', [x, u], [continuous_dynamics(x, u)])
        
        # RK4 Integration (0.1s timestep)
        dt = 0.1  # MPC timestep
        k1 = f_continuous(x, u)
        k2 = f_continuous(x + (dt/2)*k1, u)
        k3 = f_continuous(x + (dt/2)*k2, u)
        k4 = f_continuous(x + dt*k3, u)
        next_x = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        return cs.Function('discrete_dynamics', [x, u], [next_x], 
                        ['x', 'u'], ['next_x'])
    @staticmethod
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
    
  
