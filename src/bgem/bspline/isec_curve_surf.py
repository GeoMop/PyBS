
import numpy as np
import numpy.linalg as la

from . import curve_point as CP, surface_point as SP, isec_curv_surf_point as ICSP


class IsecCurveSurf:
    """
    Class which provides intersection of the given surface and curve
    """

    def __init__(self, surf, curv):
        self.surf = surf
        self.curv = curv

    def _compute_jacobian_and_coordinates(self, uvt, iuvt):
        """
        Computes Jacobian matrix and xyz coordinates, for given local parameters uvt
        and corresponding surface and curve
        :param uvt: vector of local coordinates [u,v,t] (array 3x1)
        :param iuvt: index of the knot intervals for uvt point (array 3x1)
        :return: J: jacobian matrix (array 3x3) , deltaXYZ: vector of deltas in R^3 space (array 3x1)
        """

        surf = self.surf
        curv = self.curv
        iu, iv, it = iuvt
        surf_poles = surf.poles[iu:iu + 3, iv:iv + 3, :]
        t_poles = curv.poles[it:it + 3, :]

        uf = surf.u_basis.eval_vector(iu, uvt[0])
        vf = surf.v_basis.eval_vector(iv, uvt[1])
        ufd = surf.u_basis.eval_diff_vector(iu, uvt[0])
        vfd = surf.v_basis.eval_diff_vector(iv, uvt[1])
        tf = curv.basis.eval_vector(it, uvt[2])
        tfd = curv.basis.eval_diff_vector(it, uvt[2])

        dxyz2t = t_poles.T @ tfd
        # surf_poles have shape (Nu, Nv, 3)
        dxyz1u = surf_poles.T @ ufd @ vf
        dxyz1v = surf_poles.T @ uf @ vfd
        J = np.column_stack((dxyz1u, dxyz1v, -dxyz2t))
        xyz1 = surf_poles.T  @ uf @ vf
        xyz2 = t_poles.T @ tf

        return J, xyz1, xyz2

    def get_intersection(self, iu, iv, it, max_it, rel_tol, abs_tol):
        """
        Newton iteration loop for solving intersection point
        TODO: Say what the method does.
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :param max_it: maximum number of iteration
        :param rel_tol: relative tolerance (absolute in parametric space)
        :param abs_tol: absolute tolerance (in R3 space)
        :return:
            uvt: vector of initial guess of local coordinates [u,v,t] (array 3x1),
            conv: as "0" if the method does not achieve desired accuracy
                    "1" if the method achieve desired accuracy
            xyz: as coordinates in R3
            direction: where "0" corresponds to the intersection on the lover bound of the local interval
                        where "1" corresponds to the intersection on the upper bound of the local interval
                        where "-1" corresponds to the intersection inside local interval
        """

        min_bounds = np.array([self.surf.u_basis.knots[iu + 2], self.surf.v_basis.knots[iv + 2], self.curv.basis.knots[it + 2]])
        max_bounds = np.array([self.surf.u_basis.knots[iu + 3], self.surf.v_basis.knots[iv + 3], self.curv.basis.knots[it + 3]])
        uvt = (min_bounds + max_bounds)/2

        iuvt = (iu, iv, it)
        #uvt_basis = [self.surf.u_basis, self.surf.v_basis, self.curv.basis]
        #bounds = [basis.knot_interval_bounds(iuvt[axis]) for axis, basis in enumerate(uvt_basis)]
        #bounds = np.array(bounds).T  # shape (2, 3)
        #uvt = np.average(bounds, axis=0)

        for i in range(max_it):
            J, xyz1, xyz2 = self._compute_jacobian_and_coordinates(uvt, iuvt)
            delta_xyz = xyz1 - xyz2
            conv = (la.norm(delta_xyz) <= abs_tol)
            if la.norm(delta_xyz) < abs_tol:
                break

            delta_xyz = delta_xyz.flatten()
            uvt = uvt - la.solve(J, delta_xyz)
            uvt = np.maximum(uvt, min_bounds)
            uvt = np.minimum(uvt, max_bounds)

        xyz = (xyz1 + xyz2) / 2

        return uvt, conv, xyz


    def get_intersections(self, surf, curv, tree):
        """
        Tries to compute intersection of the main curves from surface1 and patches of the surface2 which have a
         non-empty intersection of corresponding bonding boxes
        :param surf1: Surface used to construction of the main threads
        :param surf2: Intersected surface
        :param tree2: Bih tree of the patches of the surface 2
        :return: point_list as list of points of intersection
        """

        point_list = []
        crossing = np.zeros([curv.basis.n_intervals + 1])

        for it in range(curv.basis.n_intervals):
            if self._already_found(crossing, it) == 1:  # ?
                print('continue')
                continue
            intersectioned_patches2 = tree.find_box(curv.boxes[it])
            for ipatch2 in intersectioned_patches2:
                iu2, iv2 = surf.patch_id2pos(ipatch2)
                uvt,  conv, xyz = self.get_intersection(iu2, iv2, it, self.max_it, self.rel_tol, self.abs_tol)
                if conv == 1:
                    # Point A
                    t_a = uvt[2]
                    it_a = it
                    curv_point = CP.CurvePoint(curv, it_a, t_a)

                    # Point B
                    uv_b = uvt[0:2]
                    iuv_b = np.array([iu2, iv2])
                    surf_point = SP.SurfacePoint(surf, iuv_b, uv_b)

                    point = ICSP.IsecCurvSurfPoint(curv_point, surf_point, xyz)
                    point_list.append(point)

                    if curv_point.interface_flag != 0:
                        ind = it + int(0.5 * (curv_point.interface_flag + 1))
                        crossing[tuple(ind)] = 1

        return point_list

    @staticmethod
    def _already_found(crossing, it):

        found = 0
        if np.logical_or(crossing[it] == 1, crossing[it +1] == 1):
            found = 1
        return found
