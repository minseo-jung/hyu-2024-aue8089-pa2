import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """

    # [TODO] get image coordinates
    image_coords = (K @ points_3d.T).T  # (Nx3)
    
    image_coords[:, 0] /= image_coords[:, 2]  # u = u/z
    image_coords[:, 1] /= image_coords[:, 2]  # v = v/z
    
    image_coords = image_coords[:, :2]  # (Nx2)
    
    # [TODO] apply distortion
    distorted_coords = distort_points(image_coords, D, K)

    # Define projected_points (distorted_coords를 반환할 값으로 설정)
    projected_points = distorted_coords
    
    return projected_points
