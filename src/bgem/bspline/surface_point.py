import numpy as np

class SurfacePoint:
    """
    Class to represent a point on the surface including the patch coordinates.

    """
    def __init__(self, surf, iuv, uv):
        """
        :param surf: surface
        :param iuv: numpy array 2x1, corresponds to the position of the patch on the surface (defined by knot vectors )
        :param uv: numpy array 2x1 of local coordinates

        """
        self.surf = surf
        self.iuv = []
        self.iuv.append(iuv)
        self.uv = uv

        self.interface_flag = np.zeros([2], 'int')
        self.boundary_flag = np.zeros([2], 'int')

        self.curve_interface_intersection()

        self.surface_boundary_flag = 0
        if np.logical_or(self.boundary_flag[0] == 1, self.boundary_flag[1] == 1):
            self.surface_boundary_flag = 1

        self.extend_patches()

    def extend_patches(self):
        """
        add reference to the neighboring patches (if point lies on patches interface)
        :return:
        """

        n = 0

        if np.logical_and(self.boundary_flag[0] == 0, self.interface_flag[0] != 0):
            iuv = np.zeros([2], dtype=int)
            iuv[0] = self.iuv[0][0] + self.interface_flag[0]
            iuv[1] = self.iuv[0][1]
            self.iuv.append(iuv)
            n = n + 1

        if np.logical_and(self.boundary_flag[1] == 0, self.interface_flag[1] != 0):
            iuv = np.zeros([2], dtype=int)
            iuv[0] = self.iuv[0][0]
            iuv[1] = self.iuv[0][1] + self.interface_flag[1]
            self.iuv.append(iuv)
            n = n + 1

        if n == 2:
            iuv = np.zeros([2], dtype=int)
            iuv[0] = self.iuv[0][0] + self.interface_flag[0]
            iuv[1] = self.iuv[0][1] + self.interface_flag[1]
            self.iuv.append(iuv)

    def curve_interface_intersection(self):
        """
        :param t: parameter value as double
        :param ti: parameter interval as numpy array (2x1)
        :return:
        interface_flag as   "-1" corresponds to lower bound of the curve
                            "1" corresponds to upper bound of the curve
                            "0" corresponds to the interior points of the curve
        boundary_flag as    "1" if parameter corresponds to the surface boundary, i.e, equal to 0 or 1
                            "0" otherwise

        """

        ti = np.zeros((2, 2))

        ti[0, 0:2] = self.surf.u_basis.knots[self.iuv[0][0] + 2:self.iuv[0][0] + 4]
        ti[1, 0:2] = self.surf.v_basis.knots[self.iuv[0][1] + 2:self.iuv[0][1] + 4]

        for i in range(2):
            if abs(ti[i, 0] - self.uv[i]) == 0:
                self.interface_flag[i] = int(-1)
            elif abs(ti[i, 1] - self.uv[i]) == 0:
                self.interface_flag[i] = int(1)

            # max/ min parameter value instead of 0 / 1
            if np.logical_or(self.uv[i] == 0, self.uv[i] == 1):
                self.boundary_flag[i] = 1

    # def patch_id(self):
    #     """
    #     Determines ID's of all neighboring patches
    #     :param surf_point: as surface_point
    #     :return: as numpy array of integers
    #     """
    #
    #     k = len(self.iuv)
    #     patch_id = np.zeros(k, dtype=int)
    #     #patch_id = set()
    #
    #
    #     patch_id = {self.surf.patch_pos2id(iu, iv) for iu,iv in self.iuv}
    #
    #     for i in range(k):
    #         iu = self.iuv[i][0]
    #         iv = self.iuv[i][1]
    #         patch_id[i] = self.surf.patch_pos2id(iu, iv)
    #
    #     return patch_id

    def patch_id(self):
        """
        Determines ID's of all neighboring patches
        :param surf_point: as surface_point
        :return: as numpy array of integers
        """
        patch_id = set()
        patch_id = {self.surf.patch_pos2id(iu, iv) for iu, iv in self.iuv}

        return patch_id
