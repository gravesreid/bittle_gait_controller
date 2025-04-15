import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

from petoi_kinematics import PetoiKinematics
from gait_controller import GaitController

def main():
    # Create the gait controller
    controller = GaitController()
    
    # Define additional test gaits
    bouncing_gait = np.array([
        # frame 0 - extended pose (taller) - [left front, left back, right back, right front]
        [
            [-5, -4.5], # left front
            [5, -4.5],  # left back
            [5, -4.5],  # right back
            [-5, -4.5]  # right front
        ],
        # frame 1 - compressed pose (shorter)
        [
            [-5, -5.5], # left front
            [5, -5.5],  # left back
            [5, -5.5],  # right back
            [-5, -5.5]  # right front
        ],
    ])
    
    turn_left_gait = np.array([
        # frame 0 - right side legs forward/back - [left front, left back, right back, right front]
        [
            [-5, -5],  # left front
            [5, -5],   # left back
            [6, -5],   # right back
            [-6, -5]   # right front
        ],
        # frame 1 - all legs neutral
        [
            [-5, -5.5], # left front
            [5, -5.5],  # left back
            [5, -5.5],  # right back
            [-5, -5.5]  # right front
        ],
        # frame 2 - left side legs forward/back
        [
            [-4, -5],  # left front
            [4, -5],   # left back
            [5, -5],   # right back
            [-5, -5]   # right front
        ],
        # frame 3 - all legs neutral
        [
            [-5, -5.5], # left front
            [5, -5.5],  # left back
            [5, -5.5],  # right back
            [-5, -5.5]  # right front
        ],
    ])

    test_walk = np.array([
        # frame 0
        [
            [-6, -5],   [5, -6],    [5, -5],    [-6, -6]
        ],
        # frame 1
        [
            [-6, -5.5], [5, -5.5],  [5, -5.5],  [-6, -5.5] 
        ],
        # frame 2
        [
            [-6, -6],   [5, -5],    [5, -6],    [-6, -5]
        ],
        # frame 3
        [
            [-6, -5.5], [5, -5.5],  [5, -5.5],  [-6, -5.5] 
        ]
    ])
    
    # Add the custom gaits
    controller.add_custom_gait('bounce', bouncing_gait)
    controller.add_custom_gait('turn_left', turn_left_gait)
    controller.add_custom_gait('test_walk', test_walk)
    
    # Menu system for user to select gaits
    while True:
        print("\n==== Bittle Gait Testing System ====")
        print("Available gaits:")
        for i, gait_name in enumerate(controller.gaits.keys()):
            print(f"{i+1}. {gait_name}")
        print("0. Exit")
        
        choice = input("\nSelect a gait to test (0-{0}): ".format(len(controller.gaits)))
        
        try:
            choice = int(choice)
            if choice == 0:
                break
                
            if 1 <= choice <= len(controller.gaits):
                gait_name = list(controller.gaits.keys())[choice-1]
                
                # Get parameters for this gait
                num_cycles = int(input(f"Number of cycles (default 3): ") or 3)
                hold_time = float(input(f"Hold time per pose in seconds (default 0.3): ") or 0.3)
                kp = float(input(f"P gain (default 1e5): ") or 1e5)
                ki = float(input(f"I gain (default 50): ") or 50)
                kd = float(input(f"D gain (default 100): ") or 100)
                
                # Run the selected gait
                print(f"\nRunning {gait_name} gait for {num_cycles} cycles with hold time {hold_time}s")
                print("Press Escape in the Mujoco window to stop")
                
                # Show kinematics preview before running in Mujoco
                print("Showing kinematics preview (close window to continue to simulation)")
                kinematics = PetoiKinematics(render_mode='2d')
                kinematics.visualize_gait(controller.gaits[gait_name], delay=0.5)
                
                # Run the gait in Mujoco
                controller.run_gait(gait_name, num_cycles=num_cycles, hold_time=hold_time,
                                   kp=kp, ki=ki, kd=kd, dt=0.01, visualize=False)
                
            else:
                print("Invalid selection, please try again")
        except ValueError:
            print("Please enter a number")
        except Exception as e:
            print(f"Error: {e}")
            
    print("Exiting program")

def visualize_all_gaits():
    """
    Function to visualize all available gaits in the kinematics simulator
    """
    controller = GaitController()
    kinematics = PetoiKinematics(render_mode='2d')
    
    for gait_name, gait_pattern in controller.gaits.items():
        print(f"\nVisualizing {gait_name} gait")
        print("Press any key to continue to next gait, or Escape to exit")
        kinematics.visualize_gait(gait_pattern, delay=0.5)

if __name__ == "__main__":
    print("Bittle Robot Gait Testing System")
    print("================================")
    print("1. Run interactive gait testing menu")
    print("2. Visualize all gaits in kinematics simulator")
    
    choice = input("Select an option (1-2): ")
    
    if choice == "1":
        main()
    elif choice == "2":
        visualize_all_gaits()
    else:
        print("Invalid selection. Exiting.")