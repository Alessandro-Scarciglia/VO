# Import modules
import numpy as np


# Generate K from resolution and FOV
def calculate_intrinsic_matrix(fov: float,
                               resolution: tuple
                               ):
    """
    Calculate the intrinsic camera matrix K for an ideal pinhole camera.
    
    Parameters:
    - fov: Field of view in degrees (assumes symmetric FOV for both axes).
    - resolution: Tuple (width, height) of the image in pixels.
    
    Returns:
    - K: 3x3 intrinsic camera matrix.
    """

    # Image resolution
    width, height = resolution
    
    # Compute focal lengths
    fx = width / (2 * np.tan(fov / 2))
    fy = height / (2 * np.tan(fov / 2))
    
    # Compute the principal point (assumed to be the image center)
    cx = width / 2
    cy = height / 2
    
    # Construct the intrinsic matrix
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    return K