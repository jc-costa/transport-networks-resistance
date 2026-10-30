"""
3D contact-detection module: replaces intersection_utils.py's exact
line-intersection algorithm, which has no 3D equivalent (two random finite
line segments in 3D are generically skew and never intersect -- the
probability of an exact crossing is zero).

Instead, two sticks are considered electrically connected when the closest
distance between their two (finite, clamped) segments is <= contact_distance
-- the standard continuum-percolation "bonding criterion" for cylindrical
sticks/fibers (Pike & Seager 1974; Balberg, Binenbaum & Wagner 1984), where
contact_distance stands in for twice the stick radius (2r).

Each contact becomes ONE graph node, placed at the midpoint of the two
segments' closest-approach points. This mirrors identify_intersections'
output shape exactly -- `graph[k] = (stick_i, stick_j)`, the two sticks
meeting at node k -- so downstream code (find_path, discover_consecutive_edges,
pruning, merge_boundary_nodes, ImpedanceCalculator) consumes it unchanged,
with no knowledge that the coordinates are 3D or that "intersection" is now
an approximate contact rather than an exact crossing.
"""
import numpy as np


def closest_points_between_segments(a0, a1, b0, b1, epsilon=1e-12):
    """
    Closest points and distance between two finite 3D line segments
    (a0->a1 and b0->b1), handling the general skew case as well as the
    degenerate point-segment and parallel-segment cases.

    Standard clamped-parametric algorithm (Ericson, "Real-Time Collision
    Detection", section 5.1.9 / the algorithm commonly attributed to
    Lumelsky 1985). `s`/`t` in [0, 1] parameterize position along each
    segment (0 = first endpoint, 1 = second endpoint).

    Parameters
    ----------
    a0, a1 : array_like, shape (3,)
        Endpoints of segment A.
    b0, b1 : array_like, shape (3,)
        Endpoints of segment B.
    epsilon : float, optional
        Absolute threshold on squared lengths and on the parallelism term
        a*e - b^2. A segment with squared length <= epsilon is treated as a
        point, so the threshold depends on the length units.

    Returns
    -------
    point_on_a, point_on_b : np.ndarray, shape (3,)
    distance : float
    """
    a0 = np.asarray(a0, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    b1 = np.asarray(b1, dtype=float)

    d1 = a1 - a0  # direction of segment A
    d2 = b1 - b0  # direction of segment B
    r = a0 - b0

    a = np.dot(d1, d1)  # squared length of segment A
    e = np.dot(d2, d2)  # squared length of segment B
    f = np.dot(d2, r)

    if a <= epsilon and e <= epsilon:
        # both segments degenerate to points
        s, t = 0.0, 0.0
    elif a <= epsilon:
        # segment A degenerates to a point
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= epsilon:
            # segment B degenerates to a point
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            # general nondegenerate case
            b = np.dot(d1, d2)
            denom = a * e - b * b  # always >= 0 by Cauchy-Schwarz; ~0 means parallel
            if denom > epsilon:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                # parallel/near-parallel: any s solves the unconstrained
                # line-line problem equally well, so anchor s=0 and let the
                # t-clamp-and-refit below (shared with the general case)
                # find the true closest points on the finite segments
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    point_on_a = a0 + s * d1
    point_on_b = b0 + t * d2
    distance = float(np.linalg.norm(point_on_a - point_on_b))
    return point_on_a, point_on_b, distance


def identify_contacts_3d(points_1, points_2, contact_distance):
    """
    Find all stick-stick contacts in a 3D network: an O(N^2) pairwise scan
    (same complexity class as identify_intersections), connecting two sticks
    whenever their closest-approach distance is <= contact_distance.

    Parameters
    ----------
    points_1, points_2 : sequence of (x, y, z)
        Start/end points of each stick, length N.
    contact_distance : float
        Sticks closer than this (centerline-to-centerline) are electrically
        connected -- stands in for 2 * stick_radius.

    Returns
    -------
    intersections_points : list of tuple
        One 3-tuple per contact node, at the midpoint of the two sticks'
        closest-approach points.
    graph : list of tuple
        graph[k] = (stick_i, stick_j), the two sticks meeting at node k --
        same shape as identify_intersections's `graph` output.
    contact_gaps : list of float
        Actual closest-approach distance for each node (<= contact_distance).
        Unused by the resistance calculation today; kept so a future
        junction/contact-resistance model doesn't need to recompute geometry.
    """
    N = len(points_1)
    intersections_points = []
    graph = []
    contact_gaps = []

    for i in range(N):
        a0, a1 = points_1[i], points_2[i]
        for j in range(i + 1, N):
            b0, b1 = points_1[j], points_2[j]
            pt_a, pt_b, dist = closest_points_between_segments(a0, a1, b0, b1)
            if dist <= contact_distance:
                midpoint = 0.5 * (pt_a + pt_b)
                intersections_points.append(tuple(midpoint))
                graph.append((i, j))
                contact_gaps.append(dist)

    return intersections_points, graph, contact_gaps
