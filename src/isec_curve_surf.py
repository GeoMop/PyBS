
import numpy as np
import numpy.linalg as la

class IsecCurveSurf:
    """
    Class which provides intersection of the given surface and curve
    """

    def __init__(self, surf, curv):
        self.surf = surf
        self.curv = curv


    def _compute_jacobian_and_delta(self, uvt, iuvt):
        """
        Computes Jacobian matrix and delta vector of the function
        TODO: better description, what is delta, what is function.

        :param uvt: vector of local coordinates [u,v,t] (array 3x1)
        :param iuvt: index of the knot intervals for uvt point
        :return: J: jacobian matrix (array 3x3) , deltaXYZ: vector of deltas in R^3 space (array 3x1)
        """

        surf = self.surf
        curv = self.curv
        iu,iv,it = iuvt
        surf_poles = surf.poles[iu:iu + 3, iv:iv + 3, :]
        t_poles = curv.poles[it:it + 3, :]


        uf = surf.u_basis.eval_vector(iu, uvt[0])
        vf = surf.v_basis.eval_vector(iv, uvt[1])
        ufd = surf.u_basis.eval_diff_vector(iu, uvt[0])
        vfd = surf.v_basis.eval_diff_vector(iv, uvt[1])
        tf = curv.basis.eval_vector(it, uvt[2])
        tfd = curv.basis.eval_diff_vector(it, uvt[2])

        dXYZt = np.tensordot(tfd, t_poles, axes=([0], [0]))
        #print(dXYZt)
        dXYZu1 = self._energetic_inner_product(ufd, vf, surf_poles)
        dXYZv1 = self._energetic_inner_product(uf, vfd, surf_poles)
        J = np.column_stack((dXYZu1, dXYZv1, -dXYZt))

        XYZ1 = self._energetic_inner_product(uf, vf, surf_poles)
        XYZ2 = np.tensordot(tf, t_poles, axes=([0], [0]))
        #print(XYZ2)
        #return
        XYZ2 = XYZ2[:]
        XYZ1 = XYZ1[:]

        deltaXYZ = XYZ1 - XYZ2

        return J, deltaXYZ

    def get_intersection(self, iu, iv, it, max_it, rel_tol, abs_tol):
        """
        Main solving method for solving
        TODO: Say what the method does.
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :param max_it: maximum number of iteration
        :param rel_tol: relative tolerance (absolute in parametric space)
        :param abs_tol: absolute tolerance (in R3 space)
        :return:
            uvt: vector of initial guess of local coordinates [u,v,t] (array 3x1),
            TODO: here and everywhere use bool instead of int for flags
            conv as "0" if the methog does not achive desired accuracy
                    "1" if the methog achive desired accuracy
            flag as intersection specification
            XYZ
        """
        flag = -np.ones([3], 'int')
        iuvt  = (iu, iv, it)

        # patch bounds
        # TODO: remove after curve_boundary_intersection is moved into SurfacePoint
        ui = self.surf.u_basis.knots[iu + 2:iu + 4]
        vi = self.surf.v_basis.knots[iv + 2:iv + 4]
        ti = self.curv.basis.knots[it + 2:it + 4]

        uvt_basis = [self.surf.u_basis, self.surf.v_basis, self.curv.basis]
        bounds = [basis.knot_interval_bounds(iuvt[axis]) for axis, basis in enumerate(uvt_basis)]
        bounds = np.array(bounds).T     # shape (2, 3)
        uvt = np.average(bounds, axis=0)

        for i in range(max_it):
            J, delta_xyz = self._compute_jacobian_and_delta(uvt, iuvt)
            if la.norm(delta_xyz) < abs_tol:
                break


            delta_xyz = delta_xyz
            uvt = uvt - la.solve(J, delta_xyz)
            # project to patch bounds
            uvt = np.maximum(uvt, min_bounds)
            uvt = np.minimum(uvt, max_bounds)

        conv, xyz = self._test_intesection_tolerance(uvt, iuvt, abs_tol)

        if conv:
            flag[0] = self._curve_boundary_intersection(uvt[0], ui, rel_tol)
            flag[1] = self._curve_boundary_intersection(uvt[1], vi, rel_tol)
            flag[2] = self._curve_boundary_intersection(uvt[2], ti, rel_tol)


        return uvt, conv, flag, xyz

    @staticmethod
    def _curve_boundary_intersection(t,ti,rel_tol):
        """

        :param t: parameter value
        :param it: interval array (2x1)
        :return:
        flag as "0" corresponds to lower bound of the curve
                "1" corresponds to upper bound of the curve
                "-1" corresponds to the interior points of the curve
        """
        # interior boundary

        #flag = -1

        if abs(ti[0] - t) < rel_tol:
            flag = 0
        elif abs(ti[1] - t) < rel_tol:
            flag = 1
        else:
            flag = -1

        return flag

    def _test_intesection_tolerance(self, uvt, iuvt, abs_tol):
        """
        Test of the tolerance of the intersections in R3
        :param uvt: vector of local coordinates [u,v,t] (array 3x1)
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :param abs_tol: given absolute tolerance in R^3 space
        :return:
        conv - is_converged
        xyz - average intersection point
        """
        surf_r3 = self.surf.eval_local(uvt[0], uvt[1], iuvt[0], iuvt[1])
        curv_r3 = self.curv.eval_local(uvt[2], iuvt[2])
        xyz = (surf_r3 + curv_r3)/2
        dist = la.norm(surf_r3 - curv_r3)
        #print('distance =', dist)

        conv =  (dist <= abs_tol)
        return conv, xyz

    # @staticmethod
    # def _get_mean_value(knots, int):
    #     """
    #     Computes mean value of the local coordinate in the given interval
    #     :param knots: knot vector
    #     :param idx: index of the patch
    #     :return: mean
    #     """
    #     mean = (knots[int + 2] + knots[int + 3])/2
    #
    #     return mean

    @staticmethod
    def _energetic_inner_product(u, v, surf_poles):
        """
        Computes energetic inner product u^T X v
        :param u: vector of nonzero basis function in u
        :param v: vector of nonzero basis function in v
        :param X: tensor of poles in x,y,z
        :return: xyz as array (3x1)

        TODO: replace function call
        """
        #uX = np.tensordot(u, surf_poles, axes=([0], [0]))
        #xyz = np.tensordot(uX, v, axes=([0], [0]))
        # surf_poles have shape (Nu, Nv, 3)
        xyz = u @ surf_poles.T @ v
        return xyz



#class IsecCurvSurf:
#    '''
#    Class for calculation and representation of the intersection of
#    B-spline curve and B-spline surface.
#    Currently only (multi)-point intersections are assumed.
#    '''
#    def __init__(self, curve, surface):
#        pass