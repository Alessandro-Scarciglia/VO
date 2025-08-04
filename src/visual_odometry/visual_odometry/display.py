import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import cv2

def orthonormalize(R):
    """Ensure R is a proper rotation matrix."""
    U, _, Vt = svd(R)
    return U @ Vt

def set_axes_equal(ax):
    """Set equal scaling for 3D axes."""
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    centers = np.mean(extents, axis=1)
    max_range = np.max(extents[:,1] - extents[:,0]) / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - max_range, ctr + max_range)

def plot_trajectory(T_list, axis_length=0.1, show=True):
    """
    Plot trajectory from a list of 4x4 homogeneous transformation matrices.
    
    Parameters:
        T_list (List[np.ndarray]): List of 4x4 numpy arrays (poses).
        axis_length (float): Length of orientation axes (X, Y, Z).
        show (bool): Whether to call plt.show() at the end.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract positions
    positions = np.array([T[:3, 3] for T in T_list])
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'k-', label='trajectory')

    # Plot orientations
    step = max(1, len(T_list) // 50)  # Avoid clutter
    for i in range(0, len(T_list), step):
        T = T_list[i]
        origin = T[:3, 3]
        R = orthonormalize(T[:3, :3])  # Optional, safe
        ax.quiver(*origin, *(axis_length * R[:, 0]), color='r')  # X
        ax.quiver(*origin, *(axis_length * R[:, 1]), color='g')  # Y
        ax.quiver(*origin, *(axis_length * R[:, 2]), color='b')  # Z

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory with Orientation')
    ax.legend()
    ax.view_init(elev=30, azim=-135)
    set_axes_equal(ax)

    if show:
        plt.tight_layout()
        plt.show()


def draw_matches(img1, kp1, img2, kp2, matches):

    output_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    cv2.imshow("", output_img)
    cv2.waitKey(10)
