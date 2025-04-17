import numpy as np
from .petoi_kinematics import PetoiKinematics

class KinematicsHelper:
    def __init__(self):
        self.petoi = PetoiKinematics(render_mode=None)
    
    def joints_to_com(self, joint_angles_rad):
        """
        Uses forward kinematics to compute foot positions and estimate CoM.
        Returns:
            np.array([x, z, yaw]) - estimated center of mass state
        """
        # Split joint angles into legs (assuming order: FL, FR, BL, BR)
        leg_angles = joint_angles_rad.reshape(4, 2)
        
        # Compute foot positions in body frame using forward kinematics
        foot_positions = np.zeros((4, 2))  # [x,z] for each foot

        petoi_com = self.petoi.joints_to_com(leg_angles[:,0],leg_angles[:,1])
        
        # # Front left leg (index 0)
        # fl_pos = self.petoi.forward_kinematics_front(leg_angles[0])
        # foot_positions[0] = [float(fl_pos[0]), float(fl_pos[1])]
        
        # # Front right leg (index 1)
        # fr_pos = self.petoi.forward_kinematics_front(leg_angles[1])
        # foot_positions[1] = [float(fr_pos[0]), float(fr_pos[1])]
        
        # # Back left leg (index 2)
        # bl_pos = self.petoi.forward_kinematics_back(leg_angles[2])
        # foot_positions[2] = [float(bl_pos[0]), float(bl_pos[1])]
        
        # # Back right leg (index 3)
        # br_pos = self.petoi.forward_kinematics_back(leg_angles[3])
        # foot_positions[3] = [float(br_pos[0]), float(br_pos[1])]
        
        # # Simple CoM estimation:
        # # 1. Average of all foot positions (assuming legs bear equal weight)
        # # 2. Add body offset (assuming body CoM is at geometric center)
        # com_x = np.mean(foot_positions[:, 0])
        # com_z = np.mean(foot_positions[:, 1])
        
        # # Add body height offset (since foot positions are relative to shoulder)
        # com_z += self.petoi.h[0] if len(self.petoi.h) > 0 else 0.0
        
        # Body yaw angle
        yaw = self.petoi.gamma[0] if len(self.petoi.gamma) > 0 else 0.0

        com_x = petoi_com[0]
        com_z = petoi_com[1]
        
        return np.array([com_x, com_z, yaw])
