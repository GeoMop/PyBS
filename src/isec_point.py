
import numpy as np
import enum
import surface_point as SP

class Axis(enum.IntEnum):
    """
    Axis of the 2d bspline.
    """
    u = 0
    v = 1

class IsecPoint:
    """
    Point as the result of intersection with corresponding coordinates on both surfaces

    """
    def __init__(self, own_point, other_point, xyz):
        """
        TODO: variable paramaterers - curve point / surface point
        :param own_point: surface point
        :param other_point: surface point
        :param xyz: array of global coordinates as numpy array 3x1
        """

        #self.surface_point = []
        #self.surface_point.append(surface_point_a)
        #self.surface_point.append(surface_point_b)
        #self.surface_point = tuple(self.surface_point)
        self.duplicite_with = None
        self.own_point = own_point
        self.other_point = other_point
        self.tol = 1  # have to be implemented
        self.xyz = xyz
        self.connected = 0


