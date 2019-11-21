import enum


class Axis(enum.IntEnum):
    """
    Axis of the 2d bspline.
    """
    u = 0
    v = 1

class IsecCurvSurfPoint:
    """
    Point as the result of intersection with corresponding coordinates on both surfaces

    """
    def __init__(self, curve_point, surf_point, xyz):
        """
        TODO: variable paramaterers - curve point / surface point
        :param surface_point_a: surface point
        :param surface_point_b: surface point
        :param xyz: array of global coordinates as numpy array 3x1
        """


        self.curve_point = curve_point
        self.surface_point = surf_point
        self.tol = 1  # have to be implemented
        self.xyz = xyz
        self.connected = 0
