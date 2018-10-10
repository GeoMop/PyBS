import sys

build_path = "/home/jiri/Soft/Geomop/Intersections/external/bih/build"
sys.path += [build_path]
print(sys.path)

import bih
import numpy as np
import numpy.linalg as la

import bspline as bs


class IsecPoint:
    """
    Point as the result of intersection with correspondind coordinates on both surfaces
    """
    def __init__(self, surf1, iu1, iv1, uv1, boundary_flag, flag ,sum_idx, surf2, iu2, iv2, uv2,XYZ):

        self.surf1 = surf1
        self.iu1 = iu1
        self.iv1 = iv1
        self.uv1 = uv1
        self.boundary_flag = boundary_flag


        self.surf2 = surf2
        self.iu2 = iu2
        self.iv2 = iv2
        self.uv2 = uv2


        self.tol = 1 # implement
        self.R3_coor = XYZ

        self.connected = 0


        if boundary_flag == 1:
            self.add_patches(sum_idx,flag[2])

        if np.logical_or(flag[0]!=-1,flag[1]!=-1):
            self.extend_patches( flag[0:2])

        # flag

    def belongs_to(self,iu,iv,isurf):


        belongs = 0
        if isurf == 0:
            for i in range(self.iu1.__len__):
                if (np.logical_and(self.iu1[i] == iu),np.logical_and(self.iv1[i] == iv)):
                    belongs = 1
                    break

        return belongs

    #def duplicite(self,point):
    #    duplicite = 0

    #    if point.iu1.__len__

    #    return duplicite


    def connect(self):
        self.connected = 1

    def extend_patches(self, flag):

        if flag[0]!=-1:
            direction_u = 2 * flag[0] - 1
            self.iu2.append(self.iu2[0] + direction_u)
            self.iv2.append(self.iv2[0])

        if flag[1]!=-1:
            direction_v = 2 * flag[1] - 1
            self.iu2.append(self.iu2[0])
            self.iv2.append(self.iv2[0]+ direction_v)

        if np.logical_and(flag[0]!=-1,flag[1]!=-1):
            self.iu2.append(self.iu2[0]+ direction_u)
            self.iv2.append(self.iv2[0]+ direction_v)


    def add_patches(self, sum_idx,flag):

        n = self.iu1.__len__()

        direction = 2*flag -1

        if sum_idx == 0:
            for i in range(n):
                self.iu1.append(self.iu1[0]+direction)
                self.iv1.append(self.iv1[i])
        elif sum_idx == 1:
            for i in range(n):
                self.iu1.append(self.iu1[i])
                self.iv1.append(self.iv1[0]+direction)

class IsecCurveSurf:
    """
    Class which provides intersection of the given surface and curve
    """

    def __init__(self, surf, curv):
        self.surf = surf
        self.curv = curv


    def _compute_jacobian_and_delta(self, uvt, iu, iv, it):
        """
        Computes Jacobian matrix and delta vector of the function
        :param uvt: vector of local coordinates [u,v,t] (array 3x1)
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :return: J: jacobian matrix (array 3x3) , deltaXYZ: vector of deltas in R^3 space (array 3x1)
        """

        surf = self.surf
        curv = self.curv

        surf_poles = surf.poles[iu:iu + 3, iv:iv + 3, :]
        t_poles = curv.poles[it:it + 3, :]


        uf = surf.u_basis.eval_vector(iu, uvt[0, 0])
        vf = surf.v_basis.eval_vector(iv, uvt[1, 0])
        ufd = surf.u_basis.eval_diff_vector(iu, uvt[0, 0])
        vfd = surf.v_basis.eval_diff_vector(iv, uvt[1, 0])
        tf = curv.basis.eval_vector(it, uvt[2, 0])
        tfd = curv.basis.eval_diff_vector(it, uvt[2, 0])

        dXYZt = np.tensordot(tfd, t_poles, axes=([0], [0]))
        #print(dXYZt)
        dXYZu1 = self._energetic_inner_product(ufd, vf, surf_poles)
        dXYZv1 = self._energetic_inner_product(uf, vfd, surf_poles)
        J = np.column_stack((dXYZu1, dXYZv1, -dXYZt))

        XYZ1 = self._energetic_inner_product(uf, vf, surf_poles)
        XYZ2 = np.tensordot(tf, t_poles, axes=([0], [0]))
        #print(XYZ2)
        #return
        XYZ2 = XYZ2[:, None]
        XYZ1 = XYZ1[:, None]

        deltaXYZ = XYZ1 - XYZ2

        return J, deltaXYZ

    def get_intersection(self, uvt, iu, iv, it, max_it, rel_tol, abs_tol):
        """
        Main solving method for solving
        :param uvt: vector of initial guess of local coordinates [u,v,t] (array 3x1)
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :param max_it: maximum number of iteration
        :param rel_tol: relative tolerance (absolute in parametric space)
        :param abs_tol: absolute tolerance (in R3 space)
        :return: uvt: vector of initial guess of local coordinates [u,v,t] (array 3x1),
        conv as "0" if the methog does not achive desired accuracy
                "1" if the methog achive desired accuracy
        flag as intersection specification
        """
        conv = 0
        #flag = -1
        flag = -np.ones([3],'int')

        ui = self._compute_bounds(self.surf.u_basis.knots, iu)
        vi = self._compute_bounds(self.surf.v_basis.knots, iv)
        ti = self._compute_bounds(self.curv.basis.knots, it)

        for i in range(max_it):
            J, deltaXYZ = self._compute_jacobian_and_delta(uvt, iu, iv, it)
            if la.norm(deltaXYZ) < abs_tol:
                break
            uvt = uvt - la.solve(J, deltaXYZ)
            uvt = self._range_test(uvt, ui, vi, ti, 0.0)

        conv, XYZ = self._test_intesection_tolerance(uvt, iu, iv, it, abs_tol)

        if conv == 1:
            flag[0] = self._curve_boundary_intersection(uvt[0, 0], ui, rel_tol)
            flag[1] = self._curve_boundary_intersection(uvt[1, 0], vi, rel_tol)
            flag[2] = self._curve_boundary_intersection(uvt[2, 0], ti, rel_tol)


        return uvt, conv, flag, XYZ

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

        flag = -1

        if abs(ti[0] - t) < rel_tol:
            flag = 0
        elif abs(ti[1] - t) < rel_tol:
            flag = 1
        #else:
        #    flag = -1

        return flag

    # @staticmethod
    # def _patch_boundary_intersection(t,ti,rel_tol):
    #     """
    #
    #     :param t: parameter value
    #     :param it: interval array (2x1)
    #     :return:
    #     flag as "-1" corresponds to lower bound of the curve
    #             "1" corresponds to upper bound of the curve
    #             "0" corresponds to the boundary points of the curve
    #     """
    #     # interior boundary
    #
    #     flag = -1
    #
    #     if ti[0] != 0:
    #         if abs(ti[0] - t) < rel_tol:
    #             flag = 0
    #     elif ti[1] != 1:
    #         if abs(ti[1] - t) < rel_tol:
    #             flag = 1
    #     #else:
    #     #    flag = -1
    #
    #     return flag


    @staticmethod
    def _compute_bounds(knots, idx):
        """
        Computes boundary of the given area (lower and upper) in parametric space
        :param knots: knot vector
        :param idx: index of the interval
        :return: bounds as (array 2x1) (lower bound, upper bound)
        """
        s = knots[idx + 2]
        e = knots[idx + 3]
        bounds = np.array([s, e])
        return bounds

    @staticmethod
    def _range_test(uvt, ui, vi, ti, rel_tol):
        """
        Test of the entries of uvt, lies in given intervals
        with a given tolerance
        :param uvt: vector of local coordinates [u,v,t] (array 3x1)
        :param ui: knot interval of the coordinate u (array 2x1)
        :param vi: knot interval of the coordinate v (array 2x1)
        :param ti: knot interval of the coordinate t (array 2x1)
        :param rel_tol: given relative tolerance (absolute in [u,v,t])
        :return: uvt: vector of local coordinates [u,v,t] (array 3x1)
        """

        test = 0

        du = np.array([ ui[0] - uvt[0, 0],uvt[0, 0] - ui[1]])
        dv = np.array([ vi[0] - uvt[1, 0],uvt[1, 0] - vi[1]])
        dt = np.array([ ti[0] - uvt[2, 0],uvt[2, 0] - ti[1]])

        for i in range(0, 2):
            if (du[i] > rel_tol):
                uvt[0, 0] = ui[i]

        for i in range(0, 2):
            if (dv[i] > rel_tol):
                uvt[1, 0] = vi[i]

        for i in range(0, 2):
            if (dt[i] > rel_tol):
                uvt[2, 0] = ti[i]

        #if np.logical_and(uvt[0, 0] >= ui[0], uvt[0, 0] <= ui[1]):
        #    if np.logical_and(uvt[1, 0] >= vi[0], uvt[1, 0] <= vi[1]):
        #        if np.logical_and(uvt[2, 0] >= ti[0], uvt[2, 0] <= ti[1]):
        #            test = 1

        return uvt

    def _test_intesection_tolerance(self, uvt, iu, iv, it, abs_tol):
        """
        Test of the tolerance of the intersections in R3
        :param uvt: vector of local coordinates [u,v,t] (array 3x1)
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :param abs_tol: given absolute tolerance in R^3 space
        :return: conv as "0" if the methog does not achive desired accuracy
         or "1" if the methog achive desired accuracy
        """

        conv = 0

        surf_R3 = self.surf.eval_local(uvt[0, 0], uvt[1, 0], iu, iv)
        curv_R3 = self.curv.eval_local(uvt[2, 0], it)
        XYZ = (surf_R3 + curv_R3)/2
        dist = la.norm(surf_R3 - curv_R3)
        #print('distance =', dist)
        if dist <= abs_tol:
            conv = 1

        return conv, XYZ

    def get_initial_condition(self, iu, iv, it):
        """
        Computes initial guess as mean value of the considered area
        :param iu: index of the knot interval of the coordinate u
        :param iv: index of the knot interval of the coordinate v
        :param it: index of the knot interval of the coordinate t
        :return: uvt as array (3x1)
        """

        uvt = np.zeros([3, 1])

        uvt[0, 0] = self._get_mean_value(self.surf.u_basis.knots, iu)
        uvt[1, 0] = self._get_mean_value(self.surf.v_basis.knots, iv)
        uvt[2, 0] = self._get_mean_value(self.curv.basis.knots, it)

        return uvt

    @staticmethod
    def _get_mean_value(knots, int):
        """
        Computes mean value of the local coordinate in the given interval
        :param knots: knot vector
        :param idx: index of the patch
        :return: mean
        """
        mean = (knots[int + 2] + knots[int + 3])/2

        return mean

    @staticmethod
    def _energetic_inner_product(u, v, surf_poles):
        """
        Computes energetic inner product u^T X v
        :param u: vector of nonzero basis function in u
        :param v: vector of nonzero basis function in v
        :param X: tensor of poles in x,y,z
        :return: xyz as array (3x1)
        """
        uX = np.tensordot(u, surf_poles, axes=([0], [0]))
        xyz = np.tensordot(uX, v, axes=([0], [0]))
        return xyz

class IsecSurfSurf:
    def __init__(self, surf1, surf2, nt=2, max_it=10, rel_tol = 1e-16, abs_tol = 1e-14):
        self.surf1 = surf1
        self.surf2 = surf2
        self.box1, self.tree1 = self.bounding_boxes(self.surf1)
        self.box2, self.tree2 = self.bounding_boxes(self.surf2)
        self.nt = nt
        self.max_it = max_it
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        self._ipoint_list = []  # append
        # tolerance

    def bounding_boxes(self, surf):
        """
        Compute bounding boxes and construct BIH tree for a given surface
        :param surf:
        :return:
        """
        tree = bih.BIH()
        n_patch = (surf.u_basis.n_intervals) * (surf.v_basis.n_intervals)

        patch_poles = np.zeros([9, 3, n_patch])
        i_patch = 0
        for iu in range(surf.u_basis.n_intervals):
            for iv in range(surf.v_basis.n_intervals):
                n_points = 0
                for i in range(0, 3):
                    for j in range(0, 3):
                        patch_poles[n_points, :, i_patch] = surf.poles[iu + i, iv + j, :]
                        n_points += 1
                #assert i_patch == (iu * surf.v_basis.n_intervals + iv)
                assert i_patch == self._patch_pos2id(surf,iu,iv)
                i_patch += 1

        boxes = [bih.AABB(patch_poles[:, :, p].tolist()) for p in range(n_patch)]
        # print(patch_poles[:, :, 0])
        tree.add_boxes(boxes)
        tree.construct()
        # print(boxes)
        # for b in boxes:
        #    print(b.min()[0:2],b.max()[0:2])
        return boxes, tree

    def get_intersection(self):
        """
        Main method to get intersection points
        :return:
        """


        point_list1 = self.get_intersections(self.surf1, self.surf2, self.tree2)  # patches of surf 2 with respect threads of the surface 1
        point_list2 = self.get_intersections(self.surf2, self.surf1, self.tree1) # patches of surf 1 with respect threads of the surface 2

        print('points')
        for point in point_list1:
            print(point.uv1)
        print('points2')
        for point in point_list2:
            print(point.uv2)
        print(point_list1.__len__(),point_list2.__len__())

        assert point_list1.__len__() == 9
        assert point_list2.__len__() == 9

        return point_list1, point_list2

        connected_points = self._connect_points(point_list1,point_list2)


    def _connect_points(self,point_list1,point_list2):

        patch_point_list1, boundary_points1 = self.make_patch_point_list(point_list1,point_list2)
        patch_point_list2, boundary_points2 = self.make_patch_point_list(point_list2,point_list1)

        print(boundary_points2)

        patch_point_list = [[],[]]
        boundary_points = [[],[]]

        patch_point_list[0] = patch_point_list1  # append
        patch_point_list[1] = patch_point_list2
        boundary_points[0] = boundary_points1
        boundary_points[1] = boundary_points2


        #line = self._make_orderings(patch_point_list, boundary_points)





        return patch_point_list2


    def _make_orderings(self,patch_point_list, boundary_points):

        n_curves = 0

        line = []

        n_intersections = patch_point_list[0].__len__() + patch_point_list[1].__len__()
        #get_start
        #connect
        #test end

#        while n_intersections != 0:
#            for i in range(0,2):
#                for patchpoint in boundary_points[i]:
#                    if patchpoint.connected == 0:
#                        patchpoint.connect()
#                        n_intersections -= 1
#                        line[n_curves].append(patchpoint)
#                        while end_found == 0:
#                            #id  = self._patch_pos2id(patchpoint.surf2,patchpoint.iu2[k],patchpoint.iv2[k])
#                            pt = np.zeros([patchpoint.iu1.___len___])
#                            for k in range(patchpoint.iu1.___len___):
#                                ids = self._patch_pos2id(patchpoint.surf1,patchpoint.iu1[k],patchpoint.iv1[k])
#                                for pp in patch_point_list[1-i][ids]:
#                                    if pp.connected == 0:
#                                        pt[k] += 1
#                            if np.sum(pt) > 1:
#                                print('problem')
#                            else:
#
#            #found_start
#            end_found = 0
#            while end_found == 0:
#                #search next
#                check_duplicities



        while n_intersections != 0:
            for i in range(0,2):
                for patchpoint in patch_point_list[i]:
                    if np.logical_and(patchpoint.connected == 0, patchpoint.boundary_flag == 1 ):
                        end_found = 0
                        patchpoint.connect()
                        n_intersections -= 1
                        line[n_curves].append(patchpoint)
                        while end_found == 0:
                            #id  = self._patch_pos2id(patchpoint.surf2,patchpoint.iu2[k],patchpoint.iv2[k])
                            pt = np.zeros([patchpoint.iu1.___len___])
                            for k in range(patchpoint.iu1.___len___):
                                ids = self._patch_pos2id(patchpoint.surf1,patchpoint.iu1[k],patchpoint.iv1[k])
                                for pp in patch_point_list[1-i][ids]:
                                    if pp.connected == 0:
                                        pt[k] += 1
                                        id = ids
                            if np.sum(pt) > 1:
                                print('problem')
                            else:
                                end_point = patch_point_list[1-i][id]



            #found_start
#        end_found = 0
            while end_found == 0:
                #search next
                check_duplicities




            n_curves += 1



        return line

    #@staticmethod
    def make_patch_point_list(self,point_list,point_list2):

        surf1 = point_list[0].surf1

        list_len = surf1.u_basis.n_intervals * surf1.v_basis.n_intervals
        patch_points = []
        boundary_points = []

        boundary_points.append([[],[]])


        for i in range(list_len):
            patch_points.append([[],[]])

        idp = -1
        for point in point_list:
            idp += 1
            for i_patch in range(point.iu1.__len__()):
                id = self._patch_pos2id(surf1,point.iu1[i_patch],point.iv1[i_patch])
                patch_points[id][0].append(idp)
            if point.boundary_flag == 1:
                boundary_points[0].append(idp)

        idp = -1
        for point in point_list2:
            idp += 1
            for i_patch in range(point.iu2.__len__()):
                id = self._patch_pos2id(surf1, point.iu2[i_patch], point.iv2[i_patch])
                patch_points[id][1].append(idp)
            if point.boundary_flag == 1:
                boundary_points[1].append(idp)

        return patch_points, boundary_points

    @staticmethod
    def _main_threads(surf,sum_idx):
        """
        Construction of the main threads
        :param surf: surface which is used to construction of the main threads
        :param sum_idx: sum_idx == 0 --> u fixed, sum_idx == 1 --> v fixed
        :return: curves as list of curves, w_val as list of value of the fixed local coordinates , patches as list of neighbour patches
        """

        poles = surf.poles

        if sum_idx == 0:
            fix_basis = surf.u_basis
            curv_basis = surf.v_basis
        elif sum_idx == 1:
            fix_basis = surf.v_basis
            curv_basis = surf.u_basis

        curves = []
        w_val = []
        patches = []

        if sum_idx == 0:
            curv_pol = poles[0, :, :]
        elif sum_idx == 1:
            curv_pol = poles[:, 0, :]

        w_val.append(0.0)
        patches.append([0])

        curv = bs.Curve(curv_basis, curv_pol)
        curves.append(curv)

        for iw in range(1,fix_basis.n_intervals):
            w1f = fix_basis.eval_vector(iw, fix_basis.knots[iw + 2])
            if sum_idx == 0:
                surf_pol = poles[iw:iw + 3, :, :]
                curv_pol = np.tensordot(w1f, surf_pol, axes=([0], [sum_idx]))
            elif sum_idx == 1:
                surf_pol = poles[:,iw:iw + 3, :]
                curv_pol = np.tensordot(w1f, surf_pol, axes=([0], [sum_idx]))
            w_val.append(fix_basis.knots[iw + 2])
            patches.append([iw-1, iw])

            curv = bs.Curve(curv_basis, curv_pol)
            curves.append(curv)

        if sum_idx == 0:
            curv_pol = poles[poles.shape[sum_idx]-1,:, :]
            #print(poles.shape[sum_idx]-1,fix_basis.n_intervals - 1)
        elif sum_idx == 1:
            curv_pol = poles[:,poles.shape[sum_idx]-1 , :]  # fix_basis.n_intervals - 1
            #print(poles.shape[sum_idx] - 1,fix_basis.n_intervals - 1)

        w_val.append(1.0)
        patches.append([fix_basis.n_intervals-1])

        curv = bs.Curve(curv_basis, curv_pol)
        curves.append(curv)

        return curves, w_val, patches

    @staticmethod
    def _patch_pos2id(surf,iu,iv):

        id = iu * surf.v_basis.n_intervals + iv
        return id

    @staticmethod
    def _patch_id2pos(surf,id):

        iu = int(np.floor(id / surf.v_basis.n_intervals))
        iv = int(id - (iu * surf.v_basis.n_intervals))

        return iu, iv

    def get_intersections(self,surf1,surf2,tree2):
        """
        Tries to compute intersection of the main curves from surface1 and patches of the surface2 which have a
         non-empty intersection of corresponding bonding boxes
        :param surf1: Surface used to construction of the main threads
        :param surf2: Intersected surface
        :param tree2: Bih tree of the patches of the surface 2
        :return: point_list as list of points of intersection
        """

        point_list = []
        crossing  = np.zeros([surf1.u_basis.n_intervals+1,surf1.v_basis.n_intervals+1])

        for sum_idx in range(2):
            if sum_idx == 1:
                print(crossing)
            curves, w_val, patches = self._main_threads(surf1, sum_idx)
            curve_id = -1
            for curve in curves:
                curve_id += 1
                interval_id = -1
                for it in range(curve.basis.n_intervals):
                    interval_id += 1
                    if self._already_found(crossing,interval_id,curve_id,sum_idx) == 1:
                        #print('continue')
                        #print(crossing)
                        #print(point.R3_coor)
                        continue
                    curv_surf_isec = IsecCurveSurf(surf2, curve)
                    boxes = bih.AABB(curve.poles[it:it+3, :].tolist())
                    uv1 = np.zeros([2,1])
                    if np.logical_or(it == 0, it == curve.basis.n_intervals - 1):
                        interval_list = [it]
                    else:
                        interval_list = [it, it]
                    uv1[sum_idx,0] = w_val[curve_id]
                    intersectioned_patches2 = tree2.find_box(boxes)
                    #print("curve_id=", curve_id)
                    #print("sum_idx=", sum_idx)
                    #print("intersectioned_patches2=",intersectioned_patches2)
                    for ipatch2 in intersectioned_patches2:
                        iu2, iv2 = self._patch_id2pos(surf2,ipatch2)
                        uvt = curv_surf_isec.get_initial_condition(iu2, iv2, it)
                        uvt,  conv, flag, XYZ  = curv_surf_isec.get_intersection(uvt, iu2, iv2, it, self.max_it,
                                                                                 self.rel_tol, self.abs_tol)
                        if conv == 1:
                            #check second surf
                            #else break
                            if np.logical_or(np.logical_and(flag[2] == 0, it == 0),
                                             np.logical_and(flag[2] == 1, it == curve.basis.n_intervals - 1)):
                                boundary_flag = 1
                            else:
                                boundary_flag = 0

                            uv1[1 - sum_idx, 0] = uvt[2, 0]
                            if sum_idx == 0:
                                point = IsecPoint(surf1, patches[curve_id], interval_list, uv1, boundary_flag, flag,
                                                  sum_idx, surf2, [iu2], [iv2], uvt[0:2, :], XYZ)
                            elif sum_idx == 1:
                                point = IsecPoint(surf1, interval_list, patches[curve_id], uv1, boundary_flag, flag,
                                                  sum_idx,surf2, [iu2], [iv2], uvt[0:2, :], XYZ)

                            point_list.append(point)

                            #if boundary_flag == 0:
                            #    point_list[- 1].add_patches(sum_idx, flag[2]) # move to constructor
                            #print('flag')
                            #print(flag.shape)
                            #print(flag[2])
                            if np.logical_or(flag[2] == 0,flag[2] == 1):
                                if sum_idx == 0:
                                    crossing[curve_id, interval_id + flag[2]] = 1
                                elif sum_idx == 1:
                                    crossing[interval_id + flag[2],curve_id] = 1 # xxx
                                break



        return point_list

    @staticmethod
    def _already_found(crossing,interval_id,curve_id,sum_idx):

        found = 0


        if sum_idx == 0:
            for i in range(2):
                if crossing[interval_id + i,curve_id] == 1:
                    found = 1
        elif sum_idx == 1:
            for i in range(2):
                if crossing[curve_id, interval_id + i] == 1:
                    found = 1

        return found



# np.logical_not(np.logical_and(crossing[interval_id+flag,curve_id] == 1,sum_idx == 1))
    '''
    Calculation and representation of intersection of two B-spline surfaces.

    Result is set of B-spline segments approximating the intersection branches.
    Every segment consists of: a 3D curve and two 2D curves in parametric UV space of the surfaces.
    '''
