import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# The reference WKF gait pattern
REFERENCE_WKF = [
 [45, 59, 47, 58, -10, -4, -5, -4],
 [47, 59, 47, 59, -5, -4, -5, -4],
 [50, 60, 48, 59, -5, -3, -5, -4],
 [50, 60, 48, 59, -5, -3, -5, -4],
 [50, 61, 49, 60, -5, -3, -5, -3],
 [51, 61, 49, 60, -5, -3, -5, -3],
 [51, 61, 50, 62, -5, -3, -5, -2],
 [52, 61, 50, 68, -5, -3, -5, -5],
 [52, 61, 50, 73, -5, -3, -5, -10],
 [52, 62, 51, 79, -5, -2, -5, -17],
 [52, 62, 51, 83, -5, -2, -5, -23],
 [52, 62, 52, 87, -5, -2, -5, -31],
 [53, 63, 52, 86, -5, -2, -5, -37],
 [53, 63, 52, 84, -5, -2, -5, -41],
 [54, 64, 52, 79, -5, 0, -5, -44],
 [54, 70, 53, 79, -5, -2, -5, -44],
 [54, 76, 53, 79, -5, -7, -5, -44],
 [55, 81, 53, 78, -5, -14, -5, -44],
 [55, 85, 54, 71, -5, -21, -5, -43],
 [56, 91, 54, 62, -5, -29, -5, -40],
 [56, 91, 55, 54, -5, -35, -5, -35],
 [56, 89, 55, 48, -5, -40, -5, -29],
 [56, 84, 56, 44, -4, -43, -5, -21],
 [56, 84, 56, 42, -4, -43, -5, -14],
 [57, 84, 56, 41, -4, -43, -4, -9],
 [57, 84, 56, 43, -4, -43, -4, -5],
 [58, 77, 56, 45, -4, -43, -4, -5],
 [58, 69, 57, 45, -4, -40, -4, -5],
 [58, 61, 57, 45, -4, -36, -4, -5],
 [58, 55, 58, 46, -4, -30, -4, -5],
 [58, 49, 58, 46, -4, -23, -4, -5],
 [59, 46, 58, 47, -4, -15, -4, -5],
 [59, 45, 58, 47, -4, -10, -4, -5],
 [59, 47, 59, 48, -4, -5, -4, -5],
 [60, 50, 59, 48, -3, -5, -4, -5],
 [60, 50, 59, 48, -3, -5, -4, -5],
 [61, 50, 60, 49, -3, -5, -3, -5],
 [61, 51, 60, 49, -3, -5, -3, -5],
 [61, 51, 62, 50, -3, -5, -2, -5],
 [61, 52, 68, 50, -3, -5, -5, -5],
 [61, 52, 73, 51, -3, -5, -10, -5],
 [62, 52, 79, 51, -2, -5, -17, -5],
 [62, 52, 83, 52, -2, -5, -23, -5],
 [63, 52, 87, 52, -2, -5, -31, -5],
 [63, 53, 86, 52, -2, -5, -37, -5],
 [63, 53, 84, 52, -1, -5, -41, -5],
 [67, 54, 79, 52, 0, -5, -44, -5],
 [73, 54, 79, 53, -4, -5, -44, -5],
 [78, 54, 79, 53, -10, -5, -44, -5],
 [83, 55, 78, 54, -17, -5, -44, -5],
 [89, 55, 71, 54, -25, -5, -43, -5],
 [90, 56, 62, 55, -31, -5, -40, -5],
 [91, 56, 54, 55, -37, -5, -35, -5],
 [87, 56, 48, 55, -41, -5, -29, -5],
 [84, 56, 44, 56, -43, -4, -21, -5],
 [84, 56, 42, 56, -43, -4, -14, -5],
 [84, 57, 41, 56, -43, -4, -9, -4],
 [81, 57, 43, 56, -43, -4, -5, -4],
 [73, 58, 45, 57, -42, -4, -5, -4],
 [64, 58, 45, 57, -39, -4, -5, -4],
 [57, 58, 45, 58, -33, -4, -5, -4],
 [52, 58, 46, 58, -26, -4, -5, -4],
 [48, 58, 46, 58, -18, -4, -5, -4],
 [46, 59, 47, 58, -12, -4, -5, -4],
]

def load_sequence_simple(file_path):
    """
    Very simple approach to load sequence data from a Python file.
    This method directly executes the file and extracts the sequence variable.
    
    Args:
        file_path (str): Path to the Python file
        
    Returns:
        list: The sequence as a list of lists
    """
    print(f"Loading sequence from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None
    
    try:
        # Create a namespace to execute the file in
        namespace = {}
        
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Execute the file content in the namespace
        exec(content, namespace)
        
        # Look for a sequence variable (usually 'extracted_sequence' or something similar)
        sequence = None
        for var_name, value in namespace.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                print(f"Found sequence variable: {var_name} with {len(value)} rows")
                sequence = value
                break
        
        if sequence is None:
            print(f"ERROR: Could not find any sequence variable in {file_path}")
            return None
        
        return sequence
        
    except Exception as e:
        print(f"ERROR: Failed to load sequence from {file_path}")
        print(f"Exception: {e}")
        return None

def resample_sequence(sequence, target_length):
    """
    Resample a sequence to a target length using linear interpolation
    
    Args:
        sequence (list): Original sequence
        target_length (int): Target length
        
    Returns:
        list: Resampled sequence
    """
    print(f"Resampling sequence from {len(sequence)} to {target_length} points...")
    
    if len(sequence) == target_length:
        return sequence
    
    # Convert to numpy array for easier manipulation
    sequence_array = np.array(sequence)
    
    # Create indices for interpolation
    orig_indices = np.arange(len(sequence))
    target_indices = np.linspace(0, len(sequence) - 1, target_length)
    
    # Create output array
    result = []
    
    for i in target_indices:
        # Linear interpolation
        i_floor = int(np.floor(i))
        i_ceil = min(int(np.ceil(i)), len(sequence) - 1)
        
        if i_floor == i_ceil:
            # Exact index
            result.append(sequence[i_floor])
        else:
            # Interpolate
            weight = i - i_floor
            row = []
            
            for j in range(len(sequence[0])):
                val = (1 - weight) * sequence[i_floor][j] + weight * sequence[i_ceil][j]
                row.append(val)
            
            result.append(row)
    
    return result

def compare_with_reference(sequence, reference=REFERENCE_WKF, seq_name="Custom Sequence"):
    """
    Compare a sequence with the reference WKF gait
    
    Args:
        sequence (list): Sequence to compare
        reference (list): Reference sequence (defaults to WKF gait)
        seq_name (str): Name of the sequence
    """
    if not sequence:
        print("ERROR: Cannot compare empty sequence")
        return
    
    print(f"Comparing '{seq_name}' with reference WKF gait...")
    
    # Basic info about the sequences
    print(f"\nSequence info:")
    print(f"  Reference WKF: {len(reference)} points, {len(reference[0])} joints")
    print(f"  {seq_name}: {len(sequence)} points, {len(sequence[0])} joints")
    
    # Calculate sequence length difference
    if len(sequence) != len(reference):
        print(f"  Note: Sequences have different lengths. Resampling for comparison.")
        # We'll resample during visualization
    
    # Check if all joints are present
    if len(sequence[0]) != len(reference[0]):
        print(f"  Warning: Joint count mismatch ({len(sequence[0])} vs {len(reference[0])})")
        print(f"  Only comparing common joints")
    
    # Convert to numpy arrays
    seq_array = np.array(sequence)
    ref_array = np.array(reference)
    
    # Calculate some statistics
    joint_names = [
        "FL Shoulder", "FL Knee", "HR Shoulder", "HR Knee",
        "FR Shoulder", "FR Knee", "HL Shoulder", "HL Knee"
    ]
    
    # Find common joint count
    common_joints = min(len(sequence[0]), len(reference[0]))
    
    print("\nJoint angle ranges:")
    for i in range(common_joints):
        joint = joint_names[i] if i < len(joint_names) else f"Joint {i}"
        seq_min = np.min(seq_array[:, i])
        seq_max = np.max(seq_array[:, i])
        ref_min = np.min(ref_array[:, i])
        ref_max = np.max(ref_array[:, i])
        
        print(f"  {joint}:")
        print(f"    {seq_name}: {seq_min:.1f} to {seq_max:.1f} degrees")
        print(f"    Reference: {ref_min:.1f} to {ref_max:.1f} degrees")
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Resample if needed
    if len(sequence) != len(reference):
        # Resample the shorter sequence to match the longer one
        if len(sequence) < len(reference):
            sequence = resample_sequence(sequence, len(reference))
            seq_array = np.array(sequence)
        else:
            reference = resample_sequence(reference, len(sequence))
            ref_array = np.array(reference)
    
    # Create time axes
    time_index = np.arange(len(sequence))
    
    # Create figure and subplots
    fig, axes = plt.subplots(common_joints, 1, figsize=(10, 2*common_joints), sharex=True)
    
    # Make sure axes is always a list
    if common_joints == 1:
        axes = [axes]
    
    # Plot each joint
    for i in range(common_joints):
        ax = axes[i]
        joint = joint_names[i] if i < len(joint_names) else f"Joint {i}"
        
        # Plot both sequences
        ax.plot(time_index, seq_array[:, i], 'b-', label=seq_name)
        ax.plot(time_index, ref_array[:, i], 'r-', label='Reference WKF')
        
        # Add labels and grid
        ax.set_ylabel(f"{joint} (deg)")
        ax.grid(True)
        
        # Add legend to first subplot
        if i == 0:
            ax.legend(loc='upper right')
    
    # Add title and x-label
    fig.suptitle(f"Joint Angle Comparison: {seq_name} vs Reference WKF Gait")
    axes[-1].set_xlabel("Time Steps")
    
    # Adjust layout and display
    plt.tight_layout()
    
    print("Showing plot... (close the window to continue)")
    plt.show()
    print("Plot closed.")

def save_sequence_to_py(sequence, output_path, variable_name='custom_sequence'):
    """
    Save a sequence to a Python file
    
    Args:
        sequence (list): Sequence to save
        output_path (str): Path to save the file
        variable_name (str): Name of the variable in the Python file
    """
    print(f"Saving sequence to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write(f"{variable_name} = [\n")
        
        for row in sequence:
            # Format each value
            formatted_values = []
            for val in row:
                if isinstance(val, int):
                    formatted_values.append(f"{val}")
                else:
                    formatted_values.append(f"{val:.1f}")
            
            # Write the row
            f.write(f" [{', '.join(formatted_values)}],\n")
        
        f.write("]\n")
    
    print(f"Sequence saved to {output_path}")

def main():
    """Main function"""
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python simple_gait_visualizer.py <sequence_file.py> [options]")
        print("Options:")
        print("  --resample <length>: Resample the sequence to a specific length")
        print("  --save-reference: Save the reference WKF gait to a file")
        return
    
    # Check if user wants to save the reference
    if sys.argv[1] == "--save-reference":
        save_sequence_to_py(REFERENCE_WKF, "reference_wkf.py", "reference_wkf")
        return
    
    # Load the sequence
    sequence_file = sys.argv[1]
    sequence = load_sequence_simple(sequence_file)
    
    if not sequence:
        print("Failed to load sequence. Exiting.")
        return
    
    # Process options
    if len(sys.argv) > 2 and sys.argv[2] == "--resample" and len(sys.argv) > 3:
        # Resample the sequence
        try:
            target_length = int(sys.argv[3])
            resampled = resample_sequence(sequence, target_length)
            
            # Save the resampled sequence
            output_file = sequence_file.replace(".py", f"_resampled_{target_length}.py")
            save_sequence_to_py(resampled, output_file, "resampled_sequence")
            
            # Compare with reference
            compare_with_reference(resampled, REFERENCE_WKF, f"Resampled ({target_length} points)")
            
            return
        except ValueError:
            print(f"Error: Invalid target length: {sys.argv[3]}")
            return
    
    # Default: compare the sequence with the reference
    compare_with_reference(sequence, REFERENCE_WKF, os.path.basename(sequence_file).replace(".py", ""))

if __name__ == "__main__":
    main()