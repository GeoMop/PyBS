import numpy as np


class CurvePoint:
    """
    Class to represent a point on the surface including the patch coordinates.

    """
    def __init__(self, curv, it, t):
        """
        :param surf: surface
        :param iuv: numpy array 2x1, corresponds to the position of the patch on the surface (defined by knot vectors )
        :param uv: numpy array 2x1 of local coordinates

        """
        self.curv = curv
        self.it = []
        self.it.append(it)
        self.t = t

        self.interface_flag = 0
        self.boundary_flag = 0
        self._curve_boundary_intersection()
        self.extend_patches()

    def extend_patches(self):
        """
        add reference to the neighboring patches (if point lies on patches interface)
        :return:
        """

        if np.logical_and(self.curve_boundary_flag == 0, self.interface_flag != 0):
            it = self.it[0] + self.interface_flag
            self.it.append(it)


    def curve_boundary_intersection(self):
        """

        #:param t: parameter value as double
        #:param ti: parameter interval as numpy array (2x1)
        :return:
        interface_flag as   "-1" corresponds to lower bound of the curve
                            "1" corresponds to upper bound of the curve
                            "0" corresponds to the interior points of the curve
        boundary_flag as    "1" if parameter corresponds to the surface boundary, i.e, equal to 0 or 1
                            "0" otherwise

        """

        t0 = self.surf.basis.knots[self.it[0] + 2]
        t1 = self.surf.basis.knots[self.it[0] + 3]

        if abs(t0 - self.t) == 0:
            self.interface_flag = -1
        elif abs(t1 - self.t) == 0:
            self.interface_flag = 1

        if np.logical_or(self.t == 0, self.t == 1):
            self.boundary_flag = 1






