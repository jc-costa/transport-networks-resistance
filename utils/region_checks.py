import math

def is_point_inside_region_1_inferior_1(x, y, theta, a):
    # Check if the point (x, y) is within the specified region
    if x >= 0 and x <= a/2:
        # Calculate the lower limit of y based on the equation of a circle
        lower_limit_y = a - math.sqrt(a**2 - (x - a)**2)

        # Check if y is within the defined range
        if y >= lower_limit_y and y <= a:
            # Calculate two angles, atan_1 and atan_2
            atan_1 = math.atan2((a - y), (a - x))
            atan_2 = math.pi/2 + math.atan2(x, (a - y))

            # Check if the given theta is within the angle range
            # defined by atan_1 and atan_2
            if theta >= atan_1 and theta <= atan_2:
                # If all conditions are met, return True,
                # indicating that the point is inside the region
                return True

    # If any condition is not met, return False,
    # indicating that the point is outside the region
    return False

def is_point_inside_region_1_inferior_2(x, y, theta, a):
    if x >= a/2 and x <= a:
        lower_limit_y = a - math.sqrt(a**2 - x**2)

        if y >= lower_limit_y and y <= a:
            atan_1 = math.atan2((a - y), (a - x))
            atan_2 = math.pi/2 + math.atan2(x, (a - y))

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_1_superior_1(x, y, theta, a):
    if x >= 0 and x <= a/2:
        lower_limit_y = a
        upper_limit_y = a + math.sqrt(a**2 - (x - a)**2)

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = math.pi + math.atan2((y - a), x )
            atan_2 = 3*math.pi/2 + math.atan2((a - x), (y - a))

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_1_superior_2(x, y, theta, a):
    if x >= a/2 and x <= a:
        lower_limit_y = a
        upper_limit_y = a + math.sqrt(a**2 - x**2)

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = math.pi + math.atan2((y-a), x )
            atan_2 = 3*math.pi/2 + math.atan2((a-x), (y - a))

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_1(x, y, theta, a):
    # Check if the point (x, y) is inside any of the four subregions of region 1
    region_1_inferior_1 = is_point_inside_region_1_inferior_1(x, y, theta, a)
    region_1_inferior_2 = is_point_inside_region_1_inferior_2(x, y, theta, a)
    region_1_superior_1 = is_point_inside_region_1_superior_1(x, y, theta, a)
    region_1_superior_2 = is_point_inside_region_1_superior_2(x, y, theta, a)

    # Return True if the point is inside any of the four subregions, otherwise, return False
    return region_1_inferior_1 or region_1_inferior_2 or region_1_superior_1 or region_1_superior_2

def is_point_inside_region_2_inferior_1(x, y, theta, a):
    if x >= 0 and x <= a/2:
        square_root_1 = a - math.sqrt(a**2 - x**2)
        square_root_2 = a - math.sqrt(a**2 - (x - a)**2)

        if y >= square_root_1 and y <= square_root_2:
            atan_1 = math.asin((a - y) / a)
            atan_2 = math.pi/2 + math.atan2(x, (a - y))

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_2_inferior_2(x, y, theta, a):
    if x >= a/2 and x <= a:
        square_root_1 = a - math.sqrt(a**2 - (x - a)**2)
        square_root_2 = a - math.sqrt(a**2 - x**2)

        if y >= square_root_1 and y <= square_root_2:
            atan_1 = math.atan2((a - y), (a - x))
            atan_2 = math.pi/2 + math.acos((a - y) / a)

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_2_superior_1(x, y, theta, a):
    if x >= 0 and x <= a/2:
        lower_limit_y = a + math.sqrt(a**2 - (x - a)**2)
        upper_limit_y = a + math.sqrt(a**2 - x**2)

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = math.pi + math.atan2((y - a), x)
            atan_2 = 3*math.pi/2 + math.acos((y - a) / a)

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_2_superior_2(x, y, theta, a):
    if x >= a/2 and x <= a:
        lower_limit_y = a + math.sqrt(a**2 - x**2)
        upper_limit_y = a + math.sqrt(a**2 - (x - a)**2)

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = math.pi + math.asin((y - a) / a)
            atan_2 = 3*math.pi/2 + math.atan2((a - x), (y - a)) #Check it again

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_2(x, y, theta, a):
    region_2_inferior_1 = is_point_inside_region_2_inferior_1(x, y, theta, a)
    region_2_inferior_2 = is_point_inside_region_2_inferior_2(x, y, theta, a)
    region_2_superior_1 = is_point_inside_region_2_superior_1(x, y, theta, a)
    region_2_superior_2 = is_point_inside_region_2_superior_2(x, y, theta, a)

    return region_2_inferior_1 or region_2_inferior_2 or region_2_superior_1 or region_2_superior_2

def is_point_inside_region_2(x, y, theta, a):
    # Check if the point (x, y) is inside any of the four subregions of region 2
    region_2_inferior_1 = is_point_inside_region_2_inferior_1(x, y, theta, a)
    region_2_inferior_2 = is_point_inside_region_2_inferior_2(x, y, theta, a)
    region_2_superior_1 = is_point_inside_region_2_superior_1(x, y, theta, a)
    region_2_superior_2 = is_point_inside_region_2_superior_2(x, y, theta, a)

    # Return True if the point is inside any of the four subregions, otherwise, return False
    return region_2_inferior_1 or region_2_inferior_2 or region_2_superior_1 or region_2_superior_2

def is_point_inside_region_3_inferior_1(x, y, theta, a):
    if x >= 0 and x <= a/2:
        upper_limit_y = a - math.sqrt(a**2 - x**2)

        if y >= 0 and y <= upper_limit_y:
            atan_1 = math.asin((a - y) / a)
            atan_2 = math.pi/2 + math.acos((a - y) / a)

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_3_inferior_2(x, y, theta, a):
    if x >= a/2 and x <= a:
        upper_limit_y = a - math.sqrt(a**2 - (x - a)**2)

        if y >= 0 and y <= upper_limit_y:
            atan_1 = math.asin((a - y) / a)
            atan_2 = math.pi/2 + math.acos((a - y) / a)

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_3_superior_1(x, y, theta, a):
    if x >= 0 and x <= a/2:
        lower_limit_y = a + math.sqrt(a**2 - x**2)
        upper_limit_y = 2*a

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = math.pi + math.asin((y - a) / a)
            atan_2 = 3*math.pi/2 + math.acos((y - a) / a)

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_3_superior_2(x, y, theta, a):
    if x >= a/2 and x <= a:
        lower_limit_y = a + math.sqrt(a**2 - (x-a)**2)
        upper_limit_y = 2*a

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = math.pi + math.asin((y - a) / a)
            atan_2 = 3*math.pi/2 + math.acos((y - a) / a)

            if theta >= atan_1 and theta <= atan_2:
                #print("R3Sup2")
                return True

    return False

def is_point_inside_region_3(x, y, theta, a):
    region_3_inferior_1 = is_point_inside_region_3_inferior_1(x, y, theta, a)
    region_3_inferior_2 = is_point_inside_region_3_inferior_2(x, y, theta, a)
    region_3_superior_1 = is_point_inside_region_3_superior_1(x, y, theta, a)
    region_3_superior_2 = is_point_inside_region_3_superior_2(x, y, theta, a)

    return region_3_inferior_1 or region_3_inferior_2 or region_3_superior_1 or region_3_superior_2

def is_point_inside_region_4_inferior_1(x, y, theta, a):
    if x >= -a and x <= 0:
        lower_limit_y = a - math.sqrt(a**2 - x**2)
        upper_limit_y = a

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = math.asin((a - y) / a)
            atan_2 = math.atan2((a - y), (0-x))

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_4_inferior_2(x, y, theta, a):
    if x >= a and x <= 2*a:
        lower_limit_y = a - math.sqrt(a**2 - (x - a)**2)
        upper_limit_y = a

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = math.pi - math.atan2((a - y), (x - a))
            atan_2 = math.pi - math.asin((a - y) / a)

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_4_superior_1(x, y, theta, a):
    if x >= -a and x <= 0:
        lower_limit_y = a
        upper_limit_y = a + math.sqrt(a**2 - x**2)

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = 3*math.pi/2 + math.atan2((0-x), (y - a))
            atan_2 = 3*math.pi/2 + math.acos((y - a) / a)

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_4_superior_2(x, y, theta, a):
    if x >= a and x <= 2*a:
        lower_limit_y = a
        upper_limit_y = a + math.sqrt(a**2 - (x - a)**2)

        if y >= lower_limit_y and y <= upper_limit_y:
            atan_1 = 3*math.pi/2 - math.acos((y - a) / a)
            atan_2 = 3*math.pi/2 - math.atan2((x - a), (y - a))

            if theta >= atan_1 and theta <= atan_2:
                return True

    return False

def is_point_inside_region_4(x, y, theta, a):
    region_4_inferior_1 = is_point_inside_region_4_inferior_1(x, y, theta, a)
    region_4_inferior_2 = is_point_inside_region_4_inferior_2(x, y, theta, a)
    region_4_superior_1 = is_point_inside_region_4_superior_1(x, y, theta, a)
    region_4_superior_2 = is_point_inside_region_4_superior_2(x, y, theta, a)

    return region_4_inferior_1 or region_4_inferior_2 or region_4_superior_1 or region_4_superior_2
