import numpy as np
import random 
import math


def _point_in_bounds(point, lower=0.0, upper=1.0, tol=1e-12):
    """Return True if point coordinates are inside [lower, upper] (with tolerance)."""
    x, y = point
    return (lower - tol <= x <= upper + tol) and (lower - tol <= y <= upper + tol)


def validate_segments_bounds(points_1, points_2, lower=0.0, upper=1.0, tol=1e-12):
    """Return a list of out-of-bounds segment endpoints for diagnostics."""
    violations = []
    for idx, (p1, p2) in enumerate(zip(points_1, points_2)):
        if (not _point_in_bounds(p1, lower=lower, upper=upper, tol=tol) or
                not _point_in_bounds(p2, lower=lower, upper=upper, tol=tol)):
            violations.append({
                "segment_index": idx,
                "point_1": p1,
                "point_2": p2,
            })
    return violations

def calculate_angle(point_1, point_2):
    """
    Calculate the angle of a line segment from point_1 to point_2.
    
    Parameters:
    - point_1: Starting point (x1, y1) as a tuple or list
    - point_2: Ending point (x2, y2) as a tuple or list
    
    Returns:
    - Angle in radians (range: -π to π)
    """
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    return math.atan2(dy, dx)

def calculate_angles(points_1, points_2):
    """
    Calculate angles for all line segments defined by corresponding points.
    
    Parameters:
    - points_1: List of starting points [(x1, y1), ...]
    - points_2: List of ending points [(x2, y2), ...]
    
    Returns:
    - List of angles in radians
    """
    angles = []
    for i in range(len(points_1)):
        angle = calculate_angle(points_1[i], points_2[i])
        angles.append(angle)
    return angles

def generate_points(N, points_1, points_2, angles, a,
                    enforce_bounds=True, max_attempts_per_segment=10000):
    if a < 0:
        raise ValueError("a must be non-negative")

    # In a unit square, no segment longer than the diagonal can fit completely inside.
    if enforce_bounds and a > math.sqrt(2):
        raise ValueError(
            "Impossible to keep sticks inside [0,1]x[0,1]: "
            f"a={a} is larger than sqrt(2)."
        )

    for i in range(N):
        attempts = 0

        while True:
            attempts += 1
            if attempts > max_attempts_per_segment:
                raise RuntimeError(
                    "Could not generate an in-bounds stick after "
                    f"{max_attempts_per_segment} attempts for segment {i}. "
                    "Try reducing a or increasing max_attempts_per_segment."
                )

            # Generate random (x1, y1) coordinates in the range [0, 1)
            x_1 = np.random.uniform(0, 1)
            y_1 = np.random.uniform(0, 1)

            # Generate a random angle value in the range [0, 2*pi)
            alpha_value = np.random.uniform(0, 2 * np.pi)

            # Calculate new (x2, y2) coordinates based on a and alpha_value
            x_2 = x_1 + a * np.cos(alpha_value)
            y_2 = y_1 + a * np.sin(alpha_value)

            p1 = (x_1, y_1)
            p2 = (x_2, y_2)
            if (not enforce_bounds) or (
                _point_in_bounds(p1) and _point_in_bounds(p2)
            ):
                points_1.append(p1)
                angles.append(alpha_value)
                points_2.append(p2)
                break