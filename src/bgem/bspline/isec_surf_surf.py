
from . import bspline as bs, isec_curve_surf as ICS, isec_point as IP, surface_point as SP
import numpy.linalg as la
import numpy as np
import copy

class Patch:
    # Auxiliary object to collect Intersection points on one patch.
    def __init__(self, own, other):
        own_surf_point, other_surf_point = own[0]
        self.own_surf = own_surf_point.surf
        self.other_surf = other_surf_point.surf
        self.main_curve_points = own
        # Intersection points on main curves of the surface of the patch
        # Results of get_intersections(own_surf, other_surf).
        self.other_points = other
        # Intersection points on main curves of the other surface


class IsecSurfSurf:
    def __init__(self, surf1, surf2, nt=2, max_it=10, rel_tol = 1e-16, abs_tol = 1e-14):
        self.surf1 = surf1
        self.surf2 = surf2
        self.nt = nt
        self.max_it = max_it
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        # Intersection curves reconstruction (sequence of the points)
        self.curve_max_id = -1
        self.curve = []
        self.curve_own_neighbours = []
        self.curve_other_neighbours = []
        self.curve_surf = []
        self.curve_loop = []


    def get_intersection(self):
        """
        Main method to get intersection points
        :return: point_list1, point_list2 as the lists of the intersection points
        """
        point_list1 = self.get_intersections(self.surf1, self.surf2)  # patches of surf 2 with respect threads of the surface 1
        point_list2 = self.get_intersections(self.surf2, self.surf1) # patches of surf 1 with respect threads of the surface 2

        self._connect_points(point_list1, point_list2)
        print("point_list1=", len(point_list1))
        print("point_list2=", len(point_list2))

        return point_list1, point_list2

    @staticmethod
    def _main_curves(surf, axis):
        """
        Construction of the main curves, i.e.,
        :param surf: surface which is used to construction of the main threads
        :param axis: sum_idx == 0 --> u fixed, sum_idx == 1 --> v fixed
        :return: curves as list of curves, w_val as list of value of the fixed local coordinates ,
        patches as list of neighbour patches
        """

        poles = surf.poles

        if axis == IP.Axis.u:
            fix_basis = surf.u_basis
            curv_basis = surf.v_basis
        elif axis == IP.Axis.v:
            fix_basis = surf.v_basis
            curv_basis = surf.u_basis

        curves = []
        w_val = []
        patch = []

        patch.append(0)
        for iw in range(0, fix_basis.n_intervals):
            patch.append(iw)

        for iw in range(0, fix_basis.n_intervals+1):
            w1f = fix_basis.eval_vector(patch[iw], fix_basis.knots[iw + 2])
            #ind = [slice(0, surf.u_basis.size), slice(0, surf.v_basis.size), slice(0, 3)]
            # slice(None) !!!!!  np.s_[]
            ind = [slice(None), slice(None), slice(None)]
            ind[axis] = slice(patch[iw], patch[iw] + 3)
            surf_pol = poles[tuple(ind)]
            curv_pol = np.tensordot(w1f, surf_pol, axes=([0], [axis]))
            w_val.append(fix_basis.knots[iw + 2])
            curv = bs.Curve(curv_basis, curv_pol)
            curves.append(curv)

        return curves, w_val, patch

    def get_intersections(self, own_surf, other_surf):
        """
        Tries to compute intersection of the main curves from surface1 and patches of the surface2 which have a
         non-empty intersection of corresponding bonding boxes
        :param own_surf: Surface used to construction of the main threads
        :param other_surf: Intersected surface
        :return: point_list as list of isec_points
        """

        tree2 = other_surf.tree

        point_list = []
        crossing = np.zeros([own_surf.u_basis.n_intervals + 1, own_surf.v_basis.n_intervals + 1])

        for axis in [IP.Axis.u, IP.Axis.v]:
            curves, w_val, patch = self._main_curves(own_surf, axis)
            curve_id = -1
            for curve in curves:
                curve_id += 1
                #interval_intersections = 0
                for it in range(curve.basis.n_intervals):
                    if self._already_found(crossing, it, curve_id, axis) == 1: #?
                        print('continue')
                        continue
                    curv_surf_isec = ICS.IsecCurveSurf(other_surf, curve)
                    intersectioned_patches2 = tree2.find_box(curve.boxes[it])
                    for ipatch2 in intersectioned_patches2:
                        iu2, iv2 = other_surf.patch_id2pos(ipatch2)
                        uvt,  conv, xyz = curv_surf_isec.get_intersection(iu2, iv2, it, self.max_it,
                                                                            self.rel_tol, self.abs_tol)
                        if conv == 1:
                            # Point A
                            uv_a = np.zeros([2])
                            uv_a[axis] = w_val[curve_id]
                            uv_a[1 - axis] = uvt[2]
                            iuv_a = np.zeros([2], dtype=int)
                            iuv_a[axis] = patch[curve_id]
                            iuv_a[1 - axis] = it
                            own_point = SP.SurfacePoint(own_surf, iuv_a, uv_a)

                            # Point B
                            uv_b = uvt[0:2]
                            iuv_b = np.array([iu2, iv2])
                            other_point = SP.SurfacePoint(other_surf, iuv_b, uv_b)

                            point = IP.IsecPoint(own_point, other_point, xyz)

                            point_list.append(point)

                            #if interval_intersections == 0:
                            #    point_list.append(point)
                            #    interval_intersections += 1
                            #else:
                            #    a = 1
                                #check

                            direction = own_point.interface_flag[1-axis]
                            if direction != 0:
                                ind = [curve_id, curve_id]
                                ind[1-axis] = it + int(0.5 * (direction + 1))
                                crossing[tuple(ind)] = 1
                            #break  # we consider only one point as an intersection of segment of a curve and a patch
                            #check duplicities

        return point_list

    @staticmethod
    def _already_found(crossing, interval_id, curve_id, axis):

        found = 0
        ind1 = [curve_id, curve_id]
        ind2 = [curve_id, curve_id]
        ind1[1-axis] = interval_id
        ind2[1-axis] = interval_id + 1

        if np.logical_or(crossing[tuple(ind1)] == 1, crossing[tuple(ind2)] == 1):
            found = 1

        return found

    ##########################
    # Connecting of the points
    ##########################

    def _connect_points(self, point_list1, point_list2):
        """
        builds new data structures in order to connection algorithm may work efficient & call connection algorithm
        :param point_list1: as the list of the isec_points
        :param point_list2: as the list of the isec_points
        :return:
        """

        patch_point1 = self.make_patch_point_list(point_list1, point_list2)
        patch_point2 = self.make_patch_point_list(point_list2, point_list1)

        patch_point = []
        point_list = []

        point_list.append(point_list1)
        point_list.append(point_list2)
        patch_point.append(patch_point1)
        patch_point.append(patch_point2)

        self._make_point_orderings(point_list, patch_point)

        ### summary of the intersection points (DEBUG)
        #for point_lists in point_list:
        #    print(len(point_lists))
        #    for points in point_lists:
        #        print(points.xyz)

        ### summary of the curves (DEBUG)
        print("n_curves=", self.curve_max_id+1)
        #k = -1
        #for lines in self.line:
        #    k += 1
        #    print("line size =", len(lines))
        #    i = -1
        #    for points in lines:
        #        i  += 1
        #        if self.line_surf[k][i] == 0:
        #            print(points.xyz, self.line_own_info[k][i], self.line_other_info[k][i], self.line_surf[k][i], points.own_point.patch_id(), points.other_point.patch_id())
        #        elif self.line_surf[k][i] == 1:
        #            print(points.xyz, self.line_own_info[k][i], self.line_other_info[k][i], self.line_surf[k][i], points.other_point.patch_id(), points.own_point.patch_id())

    @staticmethod
    def make_patch_point_list(own_isec_points, other_isec_points):
        """
        collects the topological information, in order to simplify access to the all isec_points which lies on every patch
        :param own_isec_points: as list of isec_points appropriate to the main curves of the surface corresponding
        to the surface_point equal to own_point
        :param other_isec_points: as list of isec_points appropriate to the general position on the surface
        corresponding to the surface_point equal to other_point
        :return: list of of the lists of the lists of the isec_points, such that
        patch_points[own/other][patch_ID][list of the intersection points]
        """

        surf = own_isec_points[0].own_point.surf

        list_len = surf.u_basis.n_intervals * surf.v_basis.n_intervals
        patch_points_own = []
        patch_points_other = []
        patch_points = []

        # initialize of the lists
        for i in range(list_len):
            patch_points_own.append([])
            patch_points_other.append([])

        # add links to the own_points
        for point in own_isec_points:
            patch_id = point.own_point.patch_id()
            for patch in patch_id:
                patch_points_own[patch].append(point)

        # add links to the other_points
        for point in other_isec_points:
            patch_id = point.other_point.patch_id()
            for patch in patch_id:
                patch_points_other[patch].append(point)

        # joint both lists
        patch_points.append(patch_points_own)
        patch_points.append(patch_points_other)

        return patch_points

    def _find_neighbours(self, isec_point, i_surf, patch_point_list):
        """
        :param point_list: list of all points
        :param patch_point_list:
        :param i_surf: index of the surface "0" or "1"
        :param patch_id: as numpy array of integers
        :return:
        """
        own_patches = isec_point.own_point.patch_id()
        other_patches = isec_point.other_point.patch_id()
        own = 0
        other = 1
        own_unconnected = []
        other_unconnected = []

        # find all unconnected own points and remove duplicities (they shouldn't be present - crossing)
        for pid in own_patches:
            own_isec_points = patch_point_list[i_surf][own][pid]
            for own_isec_point in own_isec_points:
                if own_isec_point.connected == 0:
                    if self.check_duplicities(own_isec_point.own_point, isec_point.own_point) < 0.00001:
                        own_isec_point.connected = 1
                        #print('duplicita1') # ASSERT
                        #print("vyhazuji:",own_isec_point.own_point.patch_id())
                    else:
                        own_unconnected.append(own_isec_point)
                        #print("pridavam:", own_isec_point.own_point.patch_id())

        own_list = own_unconnected.copy()
        own_list.append(isec_point)

        # find all unconnected other points and remove all duplicities
        # (it may occur, e.g., for two surfaces which having the same patch interfaces)
        for pid in own_patches:
            other_isec_points = patch_point_list[i_surf][other][pid]
            for other_isec_point in other_isec_points:
                if other_isec_point.connected == 1:
                    continue
                other_point = other_isec_point.other_point
                other_patches2 = other_isec_point.own_point.patch_id()
                intersect = len(other_patches & other_patches2)  # necessary condition
                if intersect > 0:
                    for own_isec_point in own_list:
                        own_point = own_isec_point.own_point
                        if self.check_duplicities(own_point, other_point) < 0.00001:
                            other_isec_point.connected = 1
                            #print('duplicita2') (may occur)
                            #own_isec_point.duplicite_with = other_isec_point
                            #other_isec_point.duplicite_with = own_isec_point
                    if other_isec_point.connected == 0:
                        other_unconnected.append(other_isec_point)

        return own_unconnected, other_unconnected

    def check_duplicities(self, surfpoint1, surfpoint2):
        """
        performs the test on distance of the points (iff appropriate patches are the same)
        :param surfpoint1: as surface_point
        :param surfpoint2: as surface_point
        :return: distance in parametric space (or 1 if distance is not relevant)
        """
        pid1 = surfpoint1.patch_id()
        pid2 = surfpoint2.patch_id()

        pid = pid1 - pid2 # TODO BETTER

        dist = 1
        if len(pid) == 0:
            dist = la.norm(surfpoint1.uv - surfpoint2.uv)

        return dist

    def reverse_last_curve(self):
        """
        performs reverse on all the lists corresponding to the last curve in order to move boundary point (and all
        corresponding data) of the curve to the first position of the lists
        :return:
        """

        self.curve[self.curve_max_id].reverse()
        self.curve_own_neighbours[self.curve_max_id].reverse()
        self.curve_other_neighbours[self.curve_max_id].reverse()
        self.curve_surf[self.curve_max_id].reverse()

    def add_point(self, point, i_surf, own_info, other_info):
        """
        connects the point to the last curve
        :param point: as isec_point
        :param i_surf: as integer, defines ID of the surface [0/1]
        :param own_info: as integer, number candidates for the next point from own_isec_points
        :param other_info: as integer, number candidates for the next point from other_isec_points
        :return:
        """
        point.connected = 1
        self.curve[self.curve_max_id].append(point)
        self.curve_own_neighbours[self.curve_max_id].append(own_info)
        self.curve_other_neighbours[self.curve_max_id].append(other_info)
        self.curve_surf[self.curve_max_id].append(i_surf)

    def loop_check(self):
        """
        detects closed curves, i.e., the first and the last points (of the curve) can be found on at least one common
        patch_id (on both surfaces)
        :return:
        """

        first_isec_point = self.curve[self.curve_max_id][0]
        last_isec_point = self.curve[self.curve_max_id][-1]
        point1_surf = self.curve_surf[self.curve_max_id][0]
        point2_surf = self.curve_surf[self.curve_max_id][1]

        if point1_surf == point2_surf:
            n1 = len(first_isec_point.own_point.patch_id() & last_isec_point.own_point.patch_id())
            n2 = len(first_isec_point.other_point.patch_id() & last_isec_point.other_point.patch_id())
        else:
            n1 = len(first_isec_point.own_point.patch_id() & last_isec_point.other_point.patch_id())
            n2 = len(first_isec_point.other_point.patch_id() & last_isec_point.own_point.patch_id())

        if np.logical_and(n1 > 0, n2 > 0):
            loop_detected = 1
        else:
            loop_detected = 0

        self.curve_loop.append(loop_detected)

    def _make_point_orderings(self, point_list, patch_points):
        """
        main connection algorithm, sorted sequences od the isec_points append into lists
        -1) starts from the first unconnected point
        -2) choose one of the possible directions
        -3) subsequent points are connected
        -4) when end of the curve is achieved, corresponding lists are reversed,
        -5) connects the points in the opposite direction
        :param point_list: as list of the list of the isec_points (used to find unconnected points - start points)
        :param patch_points: as list of the lists of the lists of the isec_points (used for connection algorithm)
        :return:
        """

        """
                assert point[0].surface_point[0].surf == self.surf[surf_id]
                assert point[0].own_point.surf == self.surf[surf_id] 
        """
        for n_surf in range(0, 2):
            for point in point_list[n_surf]:
                if point.connected == 1:
                    continue

                # unconnected point will be used as start point

                self.init_new_curve()
                end_found = np.zeros([2])

                i_surf = n_surf
                self.add_point(point, i_surf, -1, -1)  # "n_addepts  = 0" should be rewritten after reverse

                while end_found[1] == 0:

                    # search all patches where the last point live in
                    own_isec_point = self.curve[self.curve_max_id][-1]
                    i_surf = self.curve_surf[self.curve_max_id][-1]

                    own_isec_points, other_isec_points = self._find_neighbours(own_isec_point, i_surf, patch_points)

                    n_own_points = len(own_isec_points)
                    n_other_points = len(other_isec_points)

                    print(n_own_points, n_other_points)
                    if np.logical_and(n_own_points == 0, n_other_points == 0):
                        if end_found[0] == 0:
                            end_found[0] = 1
                            self.reverse_last_curve()
                            continue
                        else:
                            end_found[1] = 1
                            self.loop_check()
                            break

                    if len(other_isec_points) > 0:
                        point = other_isec_points[0]
                        i_surf = 1 - i_surf
                    elif len(own_isec_points) > 0:
                        point = own_isec_points[0]

                    self.add_point(point, i_surf, n_own_points, n_other_points)




    def init_new_curve(self):
        """
        initialize data structures for the new curve
        """
        self.curve.append([])
        self.curve_own_neighbours.append([])
        self.curve_other_neighbours.append([])
        self.curve_surf.append([])
        self.curve_max_id += 1
