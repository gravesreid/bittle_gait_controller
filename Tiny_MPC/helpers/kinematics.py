import numpy as np
from .petoi_kinematics import PetoiKinematics

class KinematicsHelper:
    def __init__(self):
        self.petoi = PetoiKinematics(render_mode=None)
    
    def joints_to_com(self, joint_angles_rad):
        """
        Uses joint angles to estimate the CoM via PetoiKinematics and validates with leg_ik.
        Returns:
            np.array([x, z, yaw]) - estimated center of mass state
        """
        petoi = self.petoi
        # Split into shoulder and knee angles
        alphas = joint_angles_rad[[0, 2, 4, 6]]
        betas  = joint_angles_rad[[1, 3, 5, 7]]

        # Update kinematic model
        petoi.alphas = alphas
        petoi.betas = betas
        petoi.update_gamma_h()

        # Get estimated foot positions using current joint angles (forward kinematics)
        foot_positions = np.zeros((4, 2))
        for i in range(4):
            T = petoi.T01_front if i in [0, 3] else petoi.T01_back
            alpha = alphas[i]
            beta = betas[i]
            L = petoi.leg_length + petoi.foot_length * np.sin(beta)
            x = L * np.sin(alpha)
            z = -L * np.cos(alpha)
            foot_world = T @ np.array([x, z, 1])
            foot_positions[i] = foot_world[:2]

        # Use inverse kinematics to get back the joint angles from foot positions (optional validation)
        alphas_ik, betas_ik = petoi.leg_ik(foot_positions)

        # (Optional: Check closeness to original angles if needed for debug)
        # print("IK alpha error:", np.rad2deg(alphas - alphas_ik))
        # print("IK beta error:", np.rad2deg(betas - betas_ik))

        # Compute CoM x, z from midpoint of front and back frames
        x = (petoi.T01_front[0, 2] + petoi.T01_back[0, 2]) / 2
        z = (petoi.T01_front[1, 2] + petoi.T01_back[1, 2]) / 2
        yaw = petoi.gamma[0] if petoi.gamma.size > 0 else 0.0

        return np.array([x, z, yaw])
# Initialize TinyMPC
    
    def calculate_jacobians(self):
        """Calculate Jacobian matrices if needed"""
        # Implementation...