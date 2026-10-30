"""Random stick generation in the unit square (2D pipeline).

``generate_points`` draws the sticks of one network with the global NumPy
generator, so a seed set with ``np.random.seed`` before the call fixes the
network. ``clip_segment_to_box`` is shared with the 3D generator.
"""
import numpy as np
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

def clip_segment_to_box(p1, p2, lower, upper):
    """Clip a segment to an axis-aligned box (Liang-Barsky).

    Parameters
    ----------
    p1, p2 : array_like, shape (d,)
        Segment endpoints, in any dimension d.
    lower, upper : sequence of float, length d
        Box bounds per coordinate.

    Returns
    -------
    (q1, q2) : tuple of tuple of float, or None
        Endpoints of the part inside the box, ordered from p1 towards p2, or
        None when the segment misses the box or the inside part is shorter
        than 1e-12 in the segment parameter (touching a face or an edge).
    """
    p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
    d = p2 - p1
    t0, t1 = 0.0, 1.0
    for k in range(len(p1)):
        for sign, bound in ((-1.0, lower[k]), (1.0, upper[k])):
            # inside when sign*(p - bound) <= 0, i.e. p >= lower / p <= upper
            num = sign * (bound - p1[k]); den = sign * d[k]
            if den == 0.0:
                if num < 0.0:
                    return None
                continue
            r = num / den
            if den < 0.0:
                if r > t1: return None
                if r > t0: t0 = r
            else:
                if r < t0: return None
                if r < t1: t1 = r
    if t1 - t0 <= 1e-12:
        return None
    return tuple(p1 + t0 * d), tuple(p1 + t1 * d)


def generate_points(N, points_1, points_2, angles, a,
                    enforce_bounds=True, max_attempts_per_segment=10000,
                    boundary="clip"):
    """
    Place random sticks of length a in the unit square.

    boundary="reject" (the rule used until 2026-09-15): draw a start point
        uniformly in the square and an orientation uniformly in [0, 2pi), and
        redraw until both endpoints lie inside. Exactly N sticks are placed.
        A start point closer than a to an edge survives only for orientations
        that keep the second endpoint inside, so the accepted sticks are
        depleted within a distance of order a of every edge and biased there
        toward orientations parallel to it.
    boundary="clip" (current rule): the square is a window cut from a
        larger film. Stick centres are placed uniformly in the enlarged box
        [-a/2, 1 + a/2]^2 with uniform orientation, round(N (1 + a)^2) of them
        so that the density of centres is N per unit area, and each stick is
        clipped to the square; sticks that miss the square are dropped. The
        stick material inside the square is then statistically uniform and
        isotropic up to the edge, the number of sticks actually placed is
        random with mean close to N, and clipped sticks are shorter than a.
        N is a density (expected number of stick centres per unit area), and
        N a^2 keeps its meaning as the excluded-area variable.

    The lists points_1, points_2, angles are extended in place. angles[i] is
    the direction from points_1[i] to points_2[i].

    Parameters
    ----------
    N : int
        Number of sticks ("reject") or expected number of stick centres per
        unit area ("clip").
    points_1, points_2 : list
        Receive the stick endpoints as (x, y) tuples.
    angles : list
        Receives the stick orientations, in radians in [0, 2 pi).
    a : float
        Stick length, non-negative.
    enforce_bounds : bool, optional
        "reject" only: redraw sticks with an endpoint outside the square.
        Raises ValueError if a > sqrt(2), when no stick can fit.
    max_attempts_per_segment : int, optional
        "reject" only: redraws allowed per stick before RuntimeError.
    boundary : {"clip", "reject"}, optional
        Boundary rule, as described above.

    Notes
    -----
    Each candidate stick consumes three uniform draws from ``np.random`` (two
    coordinates, one angle). Under "reject" the number of draws per network
    is random, so changing `a` changes every later draw in the process.
    """
    if a < 0:
        raise ValueError("a must be non-negative")
    if boundary not in ("reject", "clip"):
        raise ValueError(f"boundary must be 'reject' or 'clip', got {boundary!r}")

    if boundary == "clip":
        n_window = int(round(N * (1.0 + a) ** 2))
        lower = (0.0, 0.0); upper = (1.0, 1.0)
        for _ in range(n_window):
            cx = np.random.uniform(-a / 2.0, 1.0 + a / 2.0)
            cy = np.random.uniform(-a / 2.0, 1.0 + a / 2.0)
            alpha = np.random.uniform(0, 2 * np.pi)
            hx, hy = 0.5 * a * np.cos(alpha), 0.5 * a * np.sin(alpha)
            clipped = clip_segment_to_box((cx - hx, cy - hy), (cx + hx, cy + hy), lower, upper)
            if clipped is None:
                continue
            q1, q2 = clipped
            points_1.append(q1); points_2.append(q2); angles.append(alpha)
        return

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