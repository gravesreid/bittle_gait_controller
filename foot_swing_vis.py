import numpy as np
import matplotlib.pyplot as plt
from cheetahgym.controllers.foot_swing_trajectory import FootSwingTrajectory

# Parameters
swing_time = 0.5
n_points = 100
phases = np.linspace(0, 1, n_points)
height = 0.1

# Define initial and final positions for 4 feet: LF, RF, LB, RB
# Format: [x, y, z]
init_positions = [
    np.array([0.0,  0.1, 0.0]),  # Left Front
    np.array([0.0, -0.1, 0.0]),  # Right Front
    np.array([-0.1,  0.1, 0.0]), # Left Back
    np.array([-0.1, -0.1, 0.0])  # Right Back
]

final_positions = [
    np.array([0.2,  0.1, 0.0]),  # Left Front
    np.array([0.2, -0.1, 0.0]),  # Right Front
    np.array([0.1,  0.1, 0.0]),  # Left Back
    np.array([0.1, -0.1, 0.0])   # Right Back
]

labels = ['LF', 'RF', 'LB', 'RB']
colors = ['blue', 'orange', 'green', 'red']
phase_offsets = [0.0, 0.25, 0.5, 0.75]  # phase offsets for staggered gait

# Set up figure
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Iterate over 4 feet
for i in range(4):
    foot = FootSwingTrajectory()
    foot.setInitialPosition(init_positions[i])
    foot.setFinalPosition(final_positions[i])
    foot.setHeight(height)

    positions = []

    for phase in phases:
        # Wrap phase + offset around [0, 1]
        p = (phase + phase_offsets[i]) % 1.0
        foot.computeSwingTrajectoryBezier(p, swing_time)
        positions.append(foot.getPosition().copy())

    positions = np.array(positions)

    # Plot 3D trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=labels[i], color=colors[i])
    ax.scatter(*positions[0], color=colors[i], marker='o')  # start
    ax.scatter(*positions[-1], color=colors[i], marker='x')  # end

    # Plot z vs phase
    ax2.plot(phases, positions[:, 2], label=labels[i], color=colors[i])

# Customize 3D plot
ax.set_title("Foot Swing Trajectories (3D)")
ax.set_xlabel("X (Forward)")
ax.set_ylabel("Y (Lateral)")
ax.set_zlabel("Z (Vertical)")
ax.legend()
ax.grid(True)

# Customize Z-height plot
ax2.set_title("Foot Z Height vs Phase")
ax2.set_xlabel("Phase")
ax2.set_ylabel("Z (m)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
