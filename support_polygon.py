import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def project_com_to_support_polygon(feet, com):
    """
    Projects the center of mass onto the plane defined by three feet and checks if
    the projection is within the support triangle.

    Parameters:
        feet: np.array of shape (3, 3) containing feet positions (x, y, z).
        com: np.array of shape (3,) containing the center of mass position (x, y, z).

    Returns:
        com_proj: The projected COM coordinates on the plane.
        is_stable: Boolean indicating if COM projection is within support polygon.
    """
    # Define vectors in the plane
    v1 = feet[1] - feet[0]
    v2 = feet[2] - feet[0]

    # Compute normal of the plane
    normal = np.cross(v1, v2)

    # Project COM onto the plane
    u = com - feet[0]
    d = np.dot(normal, u) / np.dot(normal, normal)
    com_proj = com - d * normal

    # Check if the projection lies within the triangle (2D check)
    def is_point_in_triangle(pt, v1, v2, v3):
        # Barycentric coordinate method (in xy-plane)
        v0 = v3 - v1
        v1v = v2 - v1
        v2p = pt - v1

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1v)
        dot02 = np.dot(v0, v2p)
        dot11 = np.dot(v1v, v1v)
        dot12 = np.dot(v1v, v2p)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return (u >= 0) and (v >= 0) and (u + v <= 1)

    is_stable = is_point_in_triangle(
        com_proj[:2], feet[0][:2], feet[1][:2], feet[2][:2])

    return com_proj, is_stable

def visualize_support_polygon(feet, com_proj):
    fig, ax = plt.subplots()
    polygon = np.vstack([feet[:, :2], feet[0, :2]])

    ax.plot(polygon[:, 0], polygon[:, 1], 'b-', label='Support Polygon')
    ax.plot(com_proj[0], com_proj[1], 'ro', label='Projected COM')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Support Polygon and Projected COM')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.show()

def visualize_support_polygon_3d(feet, com, com_proj):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot feet
    ax.scatter(feet[:,0], feet[:,1], feet[:,2], color='blue', label='Feet positions')

    # Plot support polygon plane
    poly3d = [feet]
    ax.add_collection3d(Poly3DCollection(poly3d, color='cyan', alpha=0.3))

    # Plot COM and projected COM
    ax.scatter(com[0], com[1], com[2], color='green', label='COM', s=50)
    ax.scatter(com_proj[0], com_proj[1], com_proj[2], color='red', label='Projected COM', s=50)

    # Plot normal vector
    centroid = np.mean(feet, axis=0)
    v1 = feet[1] - feet[0]
    v2 = feet[2] - feet[0]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], length=0.2, color='black', label='Plane Normal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Support Polygon and COM Projection')
    ax.legend()
    plt.show()


# Example usage:
feet = np.array([
    [-5.0, -5.0, -6.0], # Front Left
    [5.0, 5.0, -2.0], # back Right
    [5.0, -5.0, -1.0] # back Left
])
com = np.array([0, 0, 0])

com_proj, is_stable = project_com_to_support_polygon(feet, com)
print("Projected COM:", com_proj)
print("Stable?", is_stable)

visualize_support_polygon(feet, com_proj)
visualize_support_polygon_3d(feet, com, com_proj)