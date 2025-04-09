import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from petoi_kinematics import PetoiKinematics
from support_polygon import visualize_support_polygon


def project_com_to_support_polygon(feet, com):
    v1 = feet[1] - feet[0]
    v2 = feet[2] - feet[0]
    normal = np.cross(v1, v2)

    if np.linalg.norm(normal) < 1e-6:
        return com, False

    u = com - feet[0]
    d = np.dot(normal, u) / np.dot(normal, normal)
    com_proj = com - d * normal

    def is_point_in_triangle(pt, v1, v2, v3):
        v0 = v3 - v1
        v1v = v2 - v1
        v2p = pt - v1

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1v)
        dot02 = np.dot(v0, v2p)
        dot11 = np.dot(v1v, v1v)
        dot12 = np.dot(v1v, v2p)

        denom = dot00 * dot11 - dot01 * dot01
        if denom == 0:
            return com, False

        invDenom = 1 / denom
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return (u >= 0) and (v >= 0) and (u + v <= 1)

    is_stable = is_point_in_triangle(com_proj[:2], feet[0][:2], feet[1][:2], feet[2][:2])
    return com_proj, is_stable


model = PetoiKinematics(render_mode=None)

# Set stance legs: 0, 1, 2, 3 corresponds to FL, RL, RR, FR
stance_legs = [1, 2, 3]

# Initial guess for foot positions (clipped to be within bounds)
initial_foot_guess = np.array([
    [-5.0, -5.0],  # FL
    [5.0, -2.0],   # RL (was 1.0, now clipped)
    [5.0, -2.0],   # RR
    [-5.0, -6.0]   # FR
])

# Objective function to maximize margin of stability
def margin_of_stability(feet, com_proj):
    def point_to_line_distance(p, a, b):
        ap = p - a
        ab = b - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    distances = [
        point_to_line_distance(com_proj[:2], feet[i][:2], feet[(i+1)%3][:2])
        for i in range(3)
    ]
    return min(distances)

def objective(xz_flat):
    xz_feet = initial_foot_guess.copy()
    xz_feet[stance_legs] = xz_flat.reshape(-1, 2)
    com = np.array([0, 0, 0])  # CoM in robot frame
    feet = []
    for i in stance_legs:
        y = -model.b/2 if i in [0, 1] else model.b/2
        feet.append([xz_feet[i][0], y, xz_feet[i][1]])
    feet = np.array(feet)
    print(f'feet position: x: {xz_feet[stance_legs][:, 0]}, z: {xz_feet[stance_legs][:, 1]}')
    com_proj, is_stable = project_com_to_support_polygon(feet, com)

    # Visualize every iteration for debugging
    #visualize_support_polygon(feet, com_proj, pause_duration=0.3)

    if is_stable:
        margin = margin_of_stability(feet, com_proj)
        return -margin
    else:
        return 1000

leg_bounds = {
    1: [(2, 4), (-6, -5)],   # RL (x_min, x_max), (z_min, z_max)
    2: [(2, 4), (-6, -5)],   # RR
    3: [(-4, -2), (-6, -5)]  # FR
}

bounds = []
for leg in stance_legs:
    bounds.extend(leg_bounds[leg])

res = minimize(objective, initial_foot_guess[stance_legs].flatten(), bounds=bounds, method='SLSQP')
print(f'\nOptimization result: {res}')

if res.success:
    optimized_xz = initial_foot_guess.copy()
    optimized_xz[stance_legs] = res.x.reshape(-1, 2)
    print(f'Optimized foot positions: {optimized_xz[stance_legs]}')

    # Step 3: Solve IK
    alphas, betas = model.leg_ik(optimized_xz)
    print(f'Leg angles: {np.degrees(alphas)}, {np.degrees(betas)}')

    feet = []
    for i in range(4):
        y = -model.b/2 if i in [0, 1] else model.b/2
        feet.append([optimized_xz[i][0], y, optimized_xz[i][1]])
    feet = np.array(feet)
    com_proj, _ = project_com_to_support_polygon(feet[stance_legs], np.array([0, 0, 0]))
    visualize_support_polygon(feet[stance_legs], com_proj)

    # Step 5: Print results
    print("\n✅ Optimization successful.")
    for i in stance_legs:
        print(f"Leg {i}: foot = {optimized_xz[i]}, hip = {np.degrees(alphas[i]):.2f}°, knee = {np.degrees(betas[i]):.2f}°")
else:
    print("\n❌ Optimization failed:", res.message)

