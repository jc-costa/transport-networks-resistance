"""Exact pairwise intersections of 2D sticks.

For each pair (i, j), stick j is moved into the frame of stick i: stick i
starts at the origin, lies along the x axis, and is then shifted to the line
y = a. The ``region_checks`` predicates decide from the transformed start
point and angle of stick j whether the two sticks can cross; the crossing of
stick j's line with y = a is mapped back to the original frame and kept only
if it lies within both segments' bounding boxes.

The frame uses the nominal length `a` for both sticks. Under the ``clip``
boundary rule sticks can be shorter, which is safe: a length-a stick from
the same start point contains the clipped one, and the final bounding-box
test uses the real endpoints.
"""
import math
from region_checks import *

def point_within_segment(px, py, x1, y1, x2, y2, tolerance=1e-9):
    """
    Check if point (px, py) lies within the bounding box of segment from (x1,y1) to (x2,y2).
    
    This validates that an intersection point is actually within the finite segment,
    not just on the infinite line extension.
    
    Parameters:
    - px, py: The point to check
    - x1, y1: Start point of segment
    - x2, y2: End point of segment
    - tolerance: Small tolerance for floating point comparisons
    
    Returns:
    - True if point is within segment bounds, False otherwise
    """
    min_x, max_x = min(x1, x2) - tolerance, max(x1, x2) + tolerance
    min_y, max_y = min(y1, y2) - tolerance, max(y1, y2) + tolerance
    return min_x <= px <= max_x and min_y <= py <= max_y

def rotate_the_xy_axis(x2i, x1i, y2i, y1i, theta1, theta2, a):
    """Express the start point of stick j in the frame of stick i.

    Parameters
    ----------
    x2i, y2i : float
        Start point of stick j.
    x1i, y1i : float
        Start point of stick i, mapped to the origin.
    theta1, theta2 : float
        Orientations of sticks i and j, in radians.
    a : float
        Offset added to y, which places stick i on the line y = a.

    Returns
    -------
    x, y : float
        Start point of stick j in the new frame.
    theta : float
        Orientation of stick j relative to stick i (theta2 - theta1).
    """
   # Shift: Reference Point 1 to origin
    xs = x2i - x1i
    ys = y2i - y1i
    # Rotation by -theta1 to bring the reference line 1 to the x-axis
    theta = theta2 - theta1
    # New coordinates
    x = xs*math.cos(-theta1) - ys*math.sin(-theta1)
    y = xs*math.sin(-theta1) + ys*math.cos(-theta1)
    # Final shift
    y = y + a

    return x, y, theta
    
def back_to_the_origial_coordinate_system(x, x1i, y, y1i, theta1, theta, a):
    """Inverse of :func:`rotate_the_xy_axis`: map a point back to the original frame.

    Returns the original coordinates and the absolute angle theta + theta1.
    """
    # Final shift
    y = y - a
    # New coordinates
    xs = x*math.cos(theta1) - y*math.sin(theta1)
    ys = x*math.sin(theta1) + y*math.cos(theta1)

    x2i = xs + x1i
    y2i = ys + y1i

    theta2 = theta + theta1

    return x2i, y2i, theta2
    
def find_intersection_point(x2, y2, a, angle):
    """Crossing of the line through (x2, y2) at `angle` with the line y = a.

    Returns ``(x, y)``, or None (after printing a message) when tan(angle)
    is exactly zero. Near-parallel lines give a finite but very distant
    point, which the bounding-box test in ``identify_intersections`` then
    rejects.
    """
    try:
        x = x2 + ((a - y2) / math.tan(angle))
        y = y2 + math.tan(angle)*(x - x2)
        return x, y
    except ZeroDivisionError:
        print("Error: Tan(phi) cannot be zero.")
        return None

def identify_intersections(points_1, points_2, intersections_points, intersected_line_1, intersected_line_2, graph, angles, a, N):
    """Find every crossing between the first N sticks.

    Parameters
    ----------
    points_1, points_2 : sequence of (float, float)
        Stick endpoints; ``points_1[i]`` is the start of stick i.
    intersections_points, intersected_line_1, intersected_line_2, graph : list
        Output lists, extended in place and also returned.
    angles : sequence of float
        Direction from ``points_1[i]`` to ``points_2[i]``, in radians.
    a : float
        Nominal stick length used to build each pair's frame.
    N : int
        Number of sticks to scan; must not exceed ``len(points_1)``. Callers
        using the ``clip`` rule should pass ``len(points_1)``.

    Returns
    -------
    intersections_points : list of (float, float)
        One crossing per intersecting pair, in pair order (i < j).
    intersected_line_1, intersected_line_2 : list
        Start and end points of both sticks of each crossing, two entries per
        crossing (stick i, then stick j).
    graph : list of (int, int)
        ``graph[n] = (i, j)``: the two sticks meeting at crossing n.

    Notes
    -----
    The scan is O(N^2). Each pair is tried in the frame of stick i and, if
    no crossing is found, in the frame of stick j, because the region tests
    are not symmetric. The bounding-box test uses a tolerance of 1e-9.
    Parallel sticks (tan = 0 in the relative frame) are reported as not
    crossing, with a printed message.
    """
    def add_intersection_if_valid(x, y, theta, x1i, y1i, x1f, y1f, x2i, y2i, x2f, y2f, theta1, i, j):
        """
        Helper function to compute intersection point and add it only if it lies
        within the bounds of BOTH segments.
        
        Returns True if a valid intersection was added, False otherwise.
        """
        result = find_intersection_point(x, y, a, theta)
        if result is None:
            return False  # tan(theta) == 0 → parallel/horizontal sticks, no unique intersection
        x_new, y_new = result
        x_old, y_old, theta_old = back_to_the_origial_coordinate_system(x_new, x1i, y_new, y1i, theta1, theta, a)
        
        # Validate that intersection point is within BOTH segment bounds
        in_seg_i = point_within_segment(x_old, y_old, x1i, y1i, x1f, y1f)
        in_seg_j = point_within_segment(x_old, y_old, x2i, y2i, x2f, y2f)
        
        if in_seg_i and in_seg_j:
            intersected_line_1.append((x1i, y1i))
            intersected_line_2.append((x1f, y1f))
            intersected_line_1.append((x2i, y2i))
            intersected_line_2.append((x2f, y2f))
            intersections_points.append((x_old, y_old))
            graph.append((i, j))
            return True
        return False

    def try_all_regions(x, y, theta, x1i, y1i, x1f, y1f, x2i, y2i, x2f, y2f, theta1, i, j):
        """Check all 4 regions; return True as soon as a valid intersection is found."""
        for region_check in [is_point_inside_region_1, is_point_inside_region_2,
                             is_point_inside_region_3, is_point_inside_region_4]:
            if region_check(x, y, theta, a):
                if add_intersection_if_valid(x, y, theta, x1i, y1i, x1f, y1f,
                                             x2i, y2i, x2f, y2f, theta1, i, j):
                    return True
        return False

    for i in range(N):
        x1i, y1i = points_1[i]
        x1f, y1f = points_2[i]
        theta1 = angles[i]
        for j in range(i + 1, N):
            x2i, y2i = points_1[j]
            x2f, y2f = points_2[j]
            theta2 = angles[j]

            # Direction 1: rotate stick j into stick i's coordinate system
            x, y, theta = rotate_the_xy_axis(x2i, x1i, y2i, y1i, theta1, theta2, a)
            found = try_all_regions(x, y, theta, x1i, y1i, x1f, y1f,
                                    x2i, y2i, x2f, y2f, theta1, i, j)

            # Direction 2 (fallback): rotate stick i into stick j's coordinate system
            # Covers cases where the region check is asymmetric
            if not found:
                x, y, theta = rotate_the_xy_axis(x1i, x2i, y1i, y2i, theta2, theta1, a)
                try_all_regions(x, y, theta, x2i, y2i, x2f, y2f,
                                x1i, y1i, x1f, y1f, theta2, i, j)
    
    return intersections_points, intersected_line_1, intersected_line_2, graph