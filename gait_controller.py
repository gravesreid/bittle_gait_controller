import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on your setup
import os

from PID_Controller import PID_Controller
from petoi_kinematics import PetoiKinematics

class GaitController:
    def __init__(self, model_path="urdf/bittle.xml"):
        self.pid_controller = PID_Controller(model_path)
        self.kinematics = PetoiKinematics(render_mode='2d')
        
        # Note on limb ordering:
        # In the kinematics module: [left front (0), left back (1), right back (2), right front (3)]
        # In the PID controller: [left front (0), right front (1), right back (2), left back (3)]
        # We'll handle this mismatch in the convert_to_joint_angles method
        
        # Initialize standard gait patterns
        self.gaits = {
            'stand': np.array([
                # frame 0 - Ordering: [left front, left back, right back, right front]
                [
                    [-5., -5],  # left front
                    [5, -5],    # left back
                    [5, -5],    # right back
                    [-5, -5]    # right front
                ]
            ]),

            'step': np.array([
                # frame 0 - Ordering: [left front, left back, right back, right front]
                [
                    [-5, -5],   # left front
                    [5, -5],    # left back
                    [5, -6],    # right back
                    [-5, -6]    # right front
                ],
                # frame 1
                [
                    [-5, -5.5], # left front
                    [5, -5.5],  # left back
                    [5, -5.5],  # right back
                    [-5, -5.5]  # right front
                ],
                # frame 2
                [
                    [-5, -6],   # left front
                    [5, -6],    # left back
                    [5, -5],    # right back
                    [-5, -5]    # right front
                ],
                # frame 3
                [
                    [-5, -5.5], # left front
                    [5, -5.5],  # left back
                    [5, -5.5],  # right back
                    [-5, -5.5]  # right front
                ]
            ]),

            'walk': np.array([
                # frame 0 - Ordering: [left front, left back, right back, right front]
                [
                    [-5, -5],    # left front
                    [5, -5],     # left back
                    [6, -5.75],  # right back
                    [-4, -5.75]  # right front
                ],
                # frame 1
                [
                    [-6, -5.25], # left front
                    [4, -5.25],  # left back
                    [7, -6],     # right back
                    [-3, -6]     # right front
                ],
                # frame 2
                [
                    [-5, -5.5],  # left front
                    [5, -5.5],   # left back
                    [6, -5],     # right back
                    [-4, -5]     # right front
                ],
                # frame 3
                [
                    [-4, -5.75], # left front
                    [6, -5.75],  # left back
                    [5, -5],     # right back
                    [-5, -5]     # right front
                ],
                # frame 4
                [
                    [-3, -6],    # left front
                    [7, -6],     # left back
                    [4, -5.25],  # right back
                    [-6, -5.25]  # right front
                ],
                # frame 5
                [
                    [-4, -5],    # left front
                    [6, -5],     # left back
                    [5, -5.5],   # right back
                    [-5, -5.5]   # right front
                ]
            ])
        }
        
    def convert_to_joint_angles(self, gait_pattern):
        """
        Convert foot positions to joint angles
        
        Args:
            gait_pattern: Array of shape [num_frames, 4, 2] or [4, 2] for single frame
        
        Returns:
            Array of joint angles for each frame
        """
        # Ensure gait_pattern is 3D (add frame dimension if needed)
        if gait_pattern.ndim == 2:
            gait_pattern = gait_pattern.reshape(1, *gait_pattern.shape)
            
        num_frames = gait_pattern.shape[0]
        joint_angles = []
        
        for frame in range(num_frames):
            # Rearrange the gait pattern to match PID controller's convention
            # Kinematics: [left front (0), left back (1), right back (2), right front (3)]
            # PID: [left front (0), right front (1), right back (2), left back (3)]
            corrected_pattern = gait_pattern[frame].copy()
            
            # Create the mapping: kinematics index -> PID controller index
            # left front (0) -> left front (0) [stays the same]
            # left back (1) -> left back (3) [moves to end]
            # right back (2) -> right back (2) [stays the same]
            # right front (3) -> right front (1) [moves up]
            kinematics_order = np.array([0, 1, 2, 3])  # Original indices
            pid_order = np.array([0, 3, 2, 1])         # Where they should go in PID convention
            
            # Rearrange the pattern
            rearranged_pattern = corrected_pattern[kinematics_order]
            
            alphas, betas = self.kinematics.leg_ik(rearranged_pattern)
            
            # Convert to degrees and arrange for PID controller format
            # Each leg has 2 joints (shoulder and knee)
            frame_angles = []
            for i in range(4):
                print(f"Leg {i} angles (shoulder, knee): {alphas[i]}, {betas[i]}")
                shoulder_deg = np.rad2deg(alphas[i])
                print(f"Shoulder angle for leg {i}: {shoulder_deg} degrees")
                knee_deg = np.rad2deg(betas[i])
                print(f"Knee angle for leg {i}: {knee_deg} degrees")
                frame_angles.extend([shoulder_deg, knee_deg])
            
            joint_angles.append(frame_angles)
            
        return np.array(joint_angles)
    
    def run_gait(self, gait_name, num_cycles=3, hold_time=1.0, transition_time=0.5, 
                 kp=1e5, ki=50, kd=100, dt=0.01, visualize=True):
        """
        Execute a specific gait pattern
        
        Args:
            gait_name: Name of the gait to run ('stand', 'step', or 'walk')
            num_cycles: Number of times to repeat the gait pattern
            hold_time: Time to hold each pose (seconds)
            transition_time: Time to transition between poses (seconds)
            kp, ki, kd: PID controller parameters
            dt: Time step for simulation
            visualize: Whether to visualize the kinematics
        """
        if gait_name not in self.gaits:
            raise ValueError(f"Unknown gait: {gait_name}. Available gaits: {list(self.gaits.keys())}")
        
        gait_pattern = self.gaits[gait_name]
        joint_angles = self.convert_to_joint_angles(gait_pattern)
        print(f"Joint angles for gait '{gait_name}': {joint_angles}")
        
        # For standing, we just hold the pose
        if gait_name == 'stand':
            with mujoco.viewer.launch_passive(self.pid_controller.model, self.pid_controller.data) as viewer:
                if os.name == 'nt':
                    try:
                        import ctypes
                        hwnd = ctypes.windll.user32.GetForegroundWindow()
                        ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
                    except:
                        pass
                
                print(f"Executing {gait_name} gait")
                self.pid_controller.track_pose_sequence(joint_angles, hold_time=hold_time*3, 
                                                       dt=dt, kp=kp, ki=ki, kd=kd, viewer=viewer)
                return
        
        # For walking gaits, we repeat the pattern multiple times
        # Create a sequence of poses by repeating the joint angle sequence
        full_sequence = np.tile(joint_angles, (num_cycles, 1))
        
        with mujoco.viewer.launch_passive(self.pid_controller.model, self.pid_controller.data) as viewer:
            if os.name == 'nt':
                try:
                    import ctypes
                    hwnd = ctypes.windll.user32.GetForegroundWindow()
                    ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE = 3
                except:
                    pass
            
            print(f"Executing {gait_name} gait for {num_cycles} cycles")
            frames_per_second = int(1.0 / dt)
            hold_frames = int(hold_time * frames_per_second)
            
            # Run the simulation
            self.pid_controller.track_pose_sequence(full_sequence, hold_time=hold_time, 
                                                  dt=dt, kp=kp, ki=ki, kd=kd, viewer=viewer)
            
            # Optionally visualize the kinematics
            if visualize:
                print("Showing kinematics visualization")
                for frame in range(gait_pattern.shape[0]):
                    alphas, betas = self.kinematics.leg_ik(gait_pattern[frame])
                    self.kinematics.render(alphas, betas)
                    plt.pause(0.5)  # Pause to show each frame
    
    def add_custom_gait(self, name, gait_pattern):
        """
        Add a custom gait pattern to the available gaits
        
        Args:
            name: Name of the gait
            gait_pattern: Array of shape [num_frames, 4, 2] for multi-frame gait
                          or [4, 2] for single frame pose
        """
        # Ensure gait_pattern has the correct dimensions
        if gait_pattern.ndim == 2 and gait_pattern.shape == (4, 2):
            # Single frame pose - reshape to add frame dimension
            gait_pattern = gait_pattern.reshape(1, 4, 2)
        elif gait_pattern.ndim == 3 and gait_pattern.shape[1:] == (4, 2):
            # Already in correct format
            pass
        else:
            raise ValueError(f"Gait pattern must have shape [num_frames, 4, 2] or [4, 2], got {gait_pattern.shape}")
        
        self.gaits[name] = gait_pattern
        print(f"Added custom gait: {name}")

# Example usage
if __name__ == "__main__":
    controller = GaitController()
    
    # Test standing gait
    print("Testing standing gait")
    controller.run_gait('stand', hold_time=3.0, dt=0.01, kp=1e5, ki=50, kd=100)
    
    # Test stepping gait
    print("Testing stepping gait")
    controller.run_gait('step', num_cycles=2, hold_time=0.3, dt=0.01, kp=1e5, ki=50, kd=100)
    
    # Test walking gait
    print("Testing walking gait")
    controller.run_gait('walk', num_cycles=2, hold_time=0.2, dt=0.01, kp=1e5, ki=50, kd=100)
    
    # Example of adding a custom gait
    custom_trot = np.array([
        # frame 0 - diagonal legs up
        [
            [-5, -4],   [5, -6],    [5, -4],    [-5, -6]
        ],
        # frame 1 - all legs down
        [
            [-5, -5.5], [5, -5.5],  [5, -5.5],  [-5, -5.5] 
        ],
        # frame 2 - alternate diagonal legs up
        [
            [-5, -6],   [5, -4],    [5, -6],    [-5, -4]
        ],
        # frame 3 - all legs down
        [
            [-5, -5.5], [5, -5.5],  [5, -5.5],  [-5, -5.5] 
        ]
    ])
    
    controller.add_custom_gait('trot', custom_trot)
    controller.run_gait('trot', num_cycles=3, hold_time=0.15, dt=0.01, kp=1e5, ki=50, kd=100)