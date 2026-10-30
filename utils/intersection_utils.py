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
    try:
        x = x2 + ((a - y2) / math.tan(angle))
        y = y2 + math.tan(angle)*(x - x2)
        return x, y
    except ZeroDivisionError:
        print("Error: Tan(phi) cannot be zero.")
        return None

def identify_intersections(points_1, points_2, intersections_points, intersected_line_1, intersected_line_2, graph, angles, a, N):
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