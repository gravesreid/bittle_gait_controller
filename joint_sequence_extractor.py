import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import sys

class InteractiveJointExtractor:
    def __init__(self, csv_path):
        """
        Interactive tool to visualize and extract joint angle sequences
        
        Args:
            csv_path (str): Path to the CSV file with joint angle data
        """
        self.csv_path = csv_path
        self.load_data()
        
        # Initial time selection
        self.start_time = None
        self.end_time = None
        
        # Create the figure and plot
        self.create_plots()
        
    def load_data(self):
        """Load and process the joint data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df)} data points")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)
            
        # Create a column with combined knee + shoulder values to simplify overall movement visualization
        self.leg_codes = ["FL", "HL", "FR", "HR"]
        for code in self.leg_codes:
            self.df[f'combined_{code}'] = self.df[f'shoulder_{code}'] + self.df[f'knee_{code}']
            
    def create_plots(self):
        """Create the interactive visualization"""
        self.fig = plt.figure(figsize=(15, 12))
        
        # Create grid layout
        self.ax_main = plt.subplot2grid((5, 1), (0, 0), rowspan=2)  # Main selector plot
        self.ax_legs = [plt.subplot2grid((5, 1), (i+2, 0)) for i in range(3)]  # Individual leg plots
        
        # Setup the main selection plot with combined values
        for code in self.leg_codes:
            self.ax_main.plot(self.df['time'], self.df[f'combined_{code}'], 
                             label=f'{code} Combined')
        
        self.ax_main.set_title('Select Time Range (Click and Drag)')
        self.ax_main.legend(loc='upper right')
        self.ax_main.grid(True)
        
        # Create span selector for time range selection
        self.span = SpanSelector(
            self.ax_main, self.on_select, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='tab:blue'),
            interactive=True, drag_from_anywhere=True
        )
        
        # Setup the detailed leg plots (initially empty)
        self.leg_axes = {}
        self.leg_lines = {}
        
        # Placeholder for detailed views (will be updated on selection)
        for i, ax in enumerate(self.ax_legs):
            if i == 0:
                ax.set_title('Selected Section: Shoulder Joints')
                self.shoulder_ax = ax
            elif i == 1:
                ax.set_title('Selected Section: Knee Joints')
                self.knee_ax = ax
            else:
                ax.set_title('Preview: Extracted Data Points')
                self.preview_ax = ax
                
            ax.grid(True)
            
        plt.tight_layout()
        plt.show()
        
    def on_select(self, xmin, xmax):
        """Handle the selection of a time range"""
        self.start_time = xmin
        self.end_time = xmax
        
        # Clear previous plots
        self.shoulder_ax.clear()
        self.knee_ax.clear()
        self.preview_ax.clear()
        
        # Set titles
        self.shoulder_ax.set_title('Selected Section: Shoulder Joints')
        self.knee_ax.set_title('Selected Section: Knee Joints')
        self.preview_ax.set_title(f'Preview: Extracted Data Points ({xmin:.2f}s to {xmax:.2f}s)')
        
        # Filter data for the selected time range
        mask = (self.df['time'] >= xmin) & (self.df['time'] <= xmax)
        selected_df = self.df.loc[mask]
        
        if selected_df.empty:
            print("No data in selected range")
            return
        
        # Plot shoulder joints
        for code in self.leg_codes:
            self.shoulder_ax.plot(selected_df['time'], selected_df[f'shoulder_{code}'], 
                                 label=f'Shoulder {code}')
        
        # Plot knee joints
        for code in self.leg_codes:
            self.knee_ax.plot(selected_df['time'], selected_df[f'knee_{code}'], 
                             label=f'Knee {code}')
        
        # Show extraction preview
        times = selected_df['time'].values
        times_original = np.arange(len(times))
        times_resampled = np.linspace(0, len(times)-1, min(50, len(times)))
        indices_resampled = np.round(times_resampled).astype(int)
        
        for code in self.leg_codes:
            self.preview_ax.plot(times_resampled, 
                                selected_df[f'shoulder_{code}'].values[indices_resampled], 
                                'o-', markersize=4, label=f'Shoulder {code}')
        
        # Add legends
        self.shoulder_ax.legend(loc='upper right')
        self.knee_ax.legend(loc='upper right')
        self.preview_ax.legend(loc='upper right')
        
        # Add grid
        self.shoulder_ax.grid(True)
        self.knee_ax.grid(True)
        self.preview_ax.grid(True)
        
        # Print extraction info
        num_points = len(selected_df)
        duration = xmax - xmin
        print(f"Selected {num_points} points from {xmin:.2f}s to {xmax:.2f}s (duration: {duration:.2f}s)")
        print(f"Press 'e' to extract this sequence, 'q' to quit")
        
        # Refresh the plot
        self.fig.canvas.draw_idle()
        
        # Connect key press event
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def on_key_press(self, event):
        """Handle key press events"""
        if event.key == 'e' and self.start_time is not None and self.end_time is not None:
            self.extract_sequence()
        elif event.key == 'q':
            plt.close(self.fig)
            
    def extract_sequence(self):
        """Extract the selected sequence and save it"""
        if self.start_time is None or self.end_time is None:
            print("No time range selected")
            return
            
        # Extract sequence
        sequence = self.get_formatted_sequence('wkf')
        
        # Create a unique filename based on the time range
        filename = f"sequence_{self.start_time:.2f}_{self.end_time:.2f}.py"
        
        # Save the sequence
        self.save_sequence_to_py(sequence, filename)
        
    def get_formatted_sequence(self, output_format='wkf'):
        """
        Get the formatted joint sequence for the selected time range
        
        Args:
            output_format (str): Format for output ('wkf' or 'bk')
            
        Returns:
            List: Formatted joint angles for robot commands
        """
        mask = (self.df['time'] >= self.start_time) & (self.df['time'] <= self.end_time)
        sequence_df = self.df.loc[mask].copy()
        # Convert joint angles from radians to degrees
        for col in sequence_df.columns:
            if 'shoulder' in col or 'knee' in col:
                sequence_df[col] = np.rad2deg(sequence_df[col])
        
        if sequence_df.empty:
            print(f"No data found between time {self.start_time} and {self.end_time}")
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
            # Format: [FL_shoulder, FR_shoulder, HL_shoulder, HR_shoulder, FL_knee, FR_knee, HL_knee, HR_knee]
            for _, row in sequence_df.iterrows():
                formatted_row = [
                    row['shoulder_FL'], row['shoulder_FR'], 
                    row['shoulder_HL'], row['shoulder_HR'],
                    row['knee_FL'], row['knee_FR'], 
                    row['knee_HL'], row['knee_HR']
                ]
                output_sequence.append(formatted_row)
        
        return output_sequence
    
    def save_sequence_to_py(self, sequence, output_path, variable_name='extracted_sequence'):
        """
        Save the extracted sequence to a Python file
        
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
        print(f"Use the following to load your sequence:")
        print(f"from {output_path[:-3]} import {variable_name}")

# Run the tool
if __name__ == "__main__":
    csv_path = "/home/reid/projects/optimal_control/joint_data.csv"
    extractor = InteractiveJointExtractor(csv_path)
    
    print("USER GUIDE:")
    print("1. Click and drag in the top plot to select a time range")
    print("2. Press 'e' to extract the selected sequence")
    print("3. Press 'q' to quit")