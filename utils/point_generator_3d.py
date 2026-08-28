"""
3D sibling of point_generator.py: generates random sticks (line segments) in
a box (the unit cube [0,1]^3 by default, or an anisotropic box via
`box_dims` -- e.g. a thin film [0,1]x[0,1]x[0,d]) instead of the unit
square. Mirrors generate_points' retry-until-in-bounds structure and
calling convention (appends into caller-provided lists) so it slots into a
pipeline the same way.

The one substantive difference beyond "one more coordinate" is direction
sampling: a stick's orientation must be a uniformly random point on the unit
sphere, NOT two independent uniform angles (theta, phi) -- that biases
orientation toward the poles, since the area element on a sphere scales with
sin(theta). Uniformity is achieved here by normalizing a 3D standard-normal
vector, which is exactly spherically symmetric.
"""
import numpy as np
import math


def _point_in_bounds_3d(point, lower=0.0, upper=1.0, tol=1e-12):
    """
    Return True if point coordinates are inside the box [lower, upper]^3
    (with tolerance). `upper` may also be a 3-tuple (Ux, Uy, Uz) for an
    anisotropic box (e.g. a thin film of thickness Uz), in which case
    `lower` still applies uniformly to all three axes (every box used in
    this project starts at the origin).
    """
    x, y, z = point
    ux, uy, uz = upper if isinstance(upper, (tuple, list, np.ndarray)) else (upper, upper, upper)
    return (lower - tol <= x <= ux + tol and
            lower - tol <= y <= uy + tol and
            lower - tol <= z <= uz + tol)


def validate_segments_bounds_3d(points_1, points_2, lower=0.0, upper=1.0, tol=1e-12):
    """Return a list of out-of-bounds segment endpoints for diagnostics."""
    violations = []
    for idx, (p1, p2) in enumerate(zip(points_1, points_2)):
        if (not _point_in_bounds_3d(p1, lower=lower, upper=upper, tol=tol) or
                not _point_in_bounds_3d(p2, lower=lower, upper=upper, tol=tol)):
            violations.append({
                "segment_index": idx,
                "point_1": p1,
                "point_2": p2,
            })
    return violations


def random_unit_vector_3d():
    """
    Draw a uniformly random direction on the unit sphere.

    Normalizing a vector of 3 iid standard-normal samples is exact (not an
    approximation): the multivariate standard normal is spherically
    symmetric, so its direction is already uniform on S^2 before
    normalizing -- no reweighting needed.
    """
    while True:
        g = np.random.normal(size=3)
        norm = np.linalg.norm(g)
        if norm > 1e-12:  # guard against the near-zero-probability underflow case
            return g / norm


def generate_points_3d(N, points_1, points_2, directions, a, box_dims=(1.0, 1.0, 1.0),
                       enforce_bounds=True, max_attempts_per_segment=10000):
    """
    3D analogue of generate_points: draws N random sticks of length `a` in
    the box [0, box_dims[0]] x [0, box_dims[1]] x [0, box_dims[2]], retrying
    until both endpoints are in-bounds.

    `box_dims` defaults to the unit cube (1.0, 1.0, 1.0), reproducing the
    original behavior exactly. Passing an anisotropic box, e.g.
    `(1.0, 1.0, d)` with `d < 1`, confines sticks to a slab of thickness d
    -- the film-thickness generalization used by
    `effective_resistance_vs_d.ipynb`. Stick position (x1,y1,z1) is drawn
    uniformly over the box; orientation is still drawn uniformly on the
    unit sphere (random_unit_vector_3d) regardless of box shape, so a thin
    box simply rejects more attempts for sticks that would poke out of the
    thin dimension -- it does not itself bias surviving sticks' in-plane vs
    out-of-plane orientation beyond what "must fit in a thin slab" implies.

    Parameters otherwise mirror generate_points exactly, except `directions`
    takes the place of `angles`: each entry is a 3-vector unit direction
    instead of a scalar angle (a single angle can't parameterize an
    arbitrary direction in 3D).
    """
    if a < 0:
        raise ValueError("a must be non-negative")

    box_dims = tuple(box_dims)
    diagonal = math.sqrt(sum(L ** 2 for L in box_dims))
    # No segment longer than the box's space diagonal can fit completely inside it.
    if enforce_bounds and a > diagonal:
        raise ValueError(
            f"Impossible to keep sticks inside a box of dims {box_dims}: "
            f"a={a} is larger than the box diagonal ({diagonal:.4f})."
        )

    for i in range(N):
        attempts = 0

        while True:
            attempts += 1
            if attempts > max_attempts_per_segment:
                raise RuntimeError(
                    "Could not generate an in-bounds stick after "
                    f"{max_attempts_per_segment} attempts for segment {i}. "
                    "Try reducing a, increasing box_dims, or increasing max_attempts_per_segment."
                )

            # Generate random (x1, y1, z1) coordinates uniformly inside the box
            p1_arr = np.random.uniform(0, 1, size=3) * np.asarray(box_dims)

            # Generate a uniformly random 3D direction
            direction = random_unit_vector_3d()

            # Calculate new (x2, y2, z2) coordinates based on a and direction
            p2_arr = p1_arr + a * direction

            p1 = tuple(p1_arr)
            p2 = tuple(p2_arr)
            if (not enforce_bounds) or (
                _point_in_bounds_3d(p1, upper=box_dims) and _point_in_bounds_3d(p2, upper=box_dims)
            ):
                points_1.append(p1)
                directions.append(tuple(direction))
                points_2.append(p2)
                break
