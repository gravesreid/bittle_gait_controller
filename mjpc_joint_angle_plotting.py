import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def visualize_joint_angles(csv_path):
    """
    Visualize joint angle history from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file with joint angle data
    """
    # Load the joint data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # convert joint angles from radians to degrees
    for col in df.columns:
        if 'shoulder' in col or 'knee' in col:
            df[col] = np.rad2deg(df[col])
    
    # Create a figure with subplots organized by leg
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 1, figure=fig, hspace=0.4)
    
    leg_names = ["Front Left (FL)", "Hind Left (HL)", "Front Right (FR)", "Hind Right (HR)"]
    leg_codes = ["FL", "HL", "FR", "HR"]
    
    for i, (name, code) in enumerate(zip(leg_names, leg_codes)):
        ax = fig.add_subplot(gs[i])
        
        # Plot shoulder joint angle
        ax.plot(df['time'], df[f'shoulder_{code}'], 'b-', label=f'Shoulder {code}')
        
        # Plot knee joint angle
        ax.plot(df['time'], df[f'knee_{code}'], 'r-', label=f'Knee {code}')
        
        ax.set_title(f'{name} Joint Angles')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.grid(True)
        ax.legend()
    
    plt.suptitle('Robot Joint Angle History', fontsize=16)
    plt.tight_layout()
    plt.show()

def extract_joint_sequence(csv_path, start_time, end_time, output_format='wkf'):
    """
    Extract a section of joint angle history for open loop commands.
    
    Args:
        csv_path (str): Path to the CSV file with joint angle data
        start_time (float): Start time for extraction
        end_time (float): End time for extraction
        output_format (str): Format for output ('wkf' or 'bk')
        
    Returns:
        List: Formatted joint angles for robot commands
    """
    # Load the joint data
    df = pd.read_csv(csv_path)

    # convert joint angles from radians to degrees
    for col in df.columns:
        if 'shoulder' in col or 'knee' in col:
            df[col] = np.rad2deg(df[col])
    
    # Filter data for the specified time range
    mask = (df['time'] >= start_time) & (df['time'] <= end_time)
    sequence_df = df.loc[mask].copy()
    
    if sequence_df.empty:
        print(f"No data found between time {start_time} and {end_time}")
        return []
    
    # Prepare the output format based on the examples in skills.py
    output_sequence = []
    
    if output_format == 'wkf':
        # Format: [FL_shoulder, FL_knee, HR_shoulder, HR_knee, FR_shoulder, FR_knee, HL_shoulder, HL_knee]
        for _, row in sequence_df.iterrows():
            formatted_row = [
                row['shoulder_FL'], row['knee_FL'], 
                row['shoulder_HR'], row['knee_HR'],
                row['shoulder_FR'], row['knee_FR'], 
                row['shoulder_HL'], row['knee_HL']
            ]
            output_sequence.append(formatted_row)
    
    elif output_format == 'bk':
        # Looking at the bk example, the format appears to be:
        # [FL_shoulder, FR_shoulder, HL_shoulder, HR_shoulder, FL_knee, FR_knee, HL_knee, HR_knee]
        for _, row in sequence_df.iterrows():
            formatted_row = [
                row['shoulder_FL'], row['shoulder_FR'], 
                row['shoulder_HL'], row['shoulder_HR'],
                row['knee_FL'], row['knee_FR'], 
                row['knee_HL'], row['knee_HR']
            ]
            output_sequence.append(formatted_row)
    
    else:
        print(f"Unknown output format: {output_format}")
        return []
    
    return output_sequence

def save_sequence_to_py(sequence, output_path, variable_name='custom_sequence'):
    """
    Save the extracted sequence to a Python file.
    
    Args:
        sequence (List): Formatted joint angles
        output_path (str): Path to save the Python file
        variable_name (str): Name of the variable in the Python file
    """
    with open(output_path, 'w') as f:
        f.write(f"{variable_name} = [\n")
        for row in sequence:
            formatted_row = [f"{val:3d}" if isinstance(val, int) else f"{val:5.1f}" for val in row]
            f.write(f"  [{', '.join(formatted_row)}],\n")
        f.write("]\n")
    
    print(f"Sequence saved to {output_path}")

# Example usage
if __name__ == "__main__":
    csv_path = "/home/reid/projects/optimal_control/joint_data.csv"
    
    # Visualize all joint angles
    visualize_joint_angles(csv_path)
    
    # Example: Extract a sequence between time 2.0 and 4.0 seconds
    # Adjust these values based on your specific needs
    sequence = extract_joint_sequence(csv_path, 3.0, 5.0, output_format='wkf')
    
    # Save the extracted sequence
    if sequence:
        save_sequence_to_py(sequence, "extracted_sequence.py", "my_robot_sequence")