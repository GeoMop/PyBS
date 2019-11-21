
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

        # Intersection curves reconstruction
        self.curve_max_id = -1
        self.line = []
        self.line_info = []
        self.line_surf = []


        self._ipoint_list = []  # append
        # tolerance


    def get_intersection(self):
        """
        Main method to get intersection points
        :return:
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
        Construction of the main threads
        Todo: what is "thread", describe.
        :param surf: surface which is used to construction of the main threads
        :param axis: sum_idx == 0 --> u fixed, sum_idx == 1 --> v fixed
        :return: curves as list of curves, w_val as list of value of the fixed local coordinates , patches as list of neighbour patches
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
        :return: point_list as list of points of intersection
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

    # Connecting of the points

    class Patch:
        def __init__(self, own, other):
            own_surf_point, other_surf_point = own[0]
            self.own_surf = own_surf_point.surf
            self.other_surf = other_surf_point.surf
            self.main_curve_points = own
            # Intersection points on main curves of the surface of the patch
            # Results of get_intersections(own_surf, other_surf).
            self.other_points = other
            # Intersection points on main curves of the other surface



    def _connect_points(self, point_list1, point_list2):

        patch_point1 = self.make_patch_point_list(point_list1, point_list2)
        patch_point2 = self.make_patch_point_list(point_list2, point_list1)

        patch_point = []
        point_list = []

        point_list.append(point_list1)
        point_list.append(point_list2)
        patch_point.append(patch_point1)
        patch_point.append(patch_point2)

        self._make_point_orderings(point_list, patch_point)

        #print(line)
        #curve_from_grid
        # test funkce pro ruzne parametry - moznosti volby S, kontrola tolerance - jak? task? pocet knotu

        for point_lists in point_list:
            print(len(point_lists))
            for points in point_lists:
                print(points.xyz)

        print("n_lines=", self.curve_max_id+1)

        for lines in self.line:
            print("line size =", len(lines))
            for points in lines:
                print(points.xyz)


    @staticmethod
    def make_patch_point_list(own_point_list, other_point_list):
        """
        creates conversion list such that patch_points[patch_ID] give list of positions to point_list,
        :param point_list: as list of intersection points
        :param surf_id:  0 or 1
        :return:
        """

        surf = own_point_list[0].own_point.surf

        list_len = surf.u_basis.n_intervals * surf.v_basis.n_intervals
        patch_points_own = []
        patch_points_other = []
        patch_points = []

        for i in range(list_len):
            patch_points_own.append([])
            patch_points_other.append([])

        for point in own_point_list:
            patch_id = point.own_point.patch_id()
            for patch in patch_id:
                patch_points_own[patch].append(point)

        for point in other_point_list:
            patch_id = point.other_point.patch_id()
            for patch in patch_id:
                patch_points_own[patch].append(point)

        patch_points.append(patch_points_own)
        patch_points.append(patch_points_other)

        return patch_points

    @staticmethod
    def _get_start_point(boundary_points, point_list):
        """
        Returns first unconnected boundary point from the list
        :param boundary_points: list of the id's of the boundary points
        :param point_list: list of the intersection points
        :return: intersection point which lies on the boundary of the surface & id of the point_list
        """
        # obtain start point of the curve
        for i in range(0, 2):
            for id_point in boundary_points[i]:
                point = point_list[i][id_point]
                if point.connected == 0:
                    i_surf = i
                    return point, i_surf

    def _find_neighbours(self, isec_point, i_surf, patch_point_list):
        """
        :param point_list: list of all points
        :param patch_point_list:
        :param i_surf: index of the surface "0" or "1"
        :param patch_id: as numpy array of integers
        :return:
        """
        patch_ids = isec_point.own_point.patch_id()
        patch2_ids = isec_point.other_point.patch_id()

        own = 0
        #other = 1
        #isec_point z volani zaradit do duplicit

        own_unconnected = []
        other_unconnected = []

        # find all unconnected own points
        for pid in patch_ids:
            own_isec_points = patch_point_list[i_surf][own][pid]
            for own_isec_point in own_isec_points:
                if own_isec_point.connected == 0:
                    if self.check_duplicities(own_isec_point.own_point, isec_point.own_point) < 0.00001:
                        own_unconnected.append(own_isec_point)
                    else:
                        own_isec_point.connected = 1
                        # ASSERT


        own_list = own_unconnected.copy()
        own_list.append(isec_point)

        # find all unconnected other points and remove all duplicities
        # (it may occur, e.g., for two surfaces which having the same patch interfaces)
        for pid2 in patch2_ids:
            other_isec_points = patch_point_list[1-i_surf][own][pid2] # -> other
            for other_isec_point in other_isec_points:
                if len(other_isec_point.other_point.patch_id() & isec_point.own_point.patch_id()) > 0:
                    if other_isec_point.connected == 1:
                        continue
                    other_point = other_isec_point.other_point
                    for own_isec_point in own_list:
                        own_point = own_isec_point.own_point
                        if self.check_duplicities(own_point, other_point) < 0.00001:
                            other_isec_point.duplicite_with = own_isec_point
                            other_isec_point.connected = 1
                            own_isec_point.duplicite_with = other_isec_point


                    if other_isec_point.connected == 0:
                        other_unconnected.append(other_isec_point)

        return own_unconnected, other_unconnected

    def check_duplicities(self, surfpoint1, surfpoint2):
        """
        :param surfpoint1:
        :param surfpoint2:
        :return:
        """
        pid1 = surfpoint1.patch_id()
        pid2 = surfpoint2.patch_id()

        pid = pid1 - pid2 # TODO BETTER

        dist = 1
        if len(pid) == 0:
            dist = la.norm(surfpoint1.uv - surfpoint2.uv)
            print(dist)

        return dist

        #return -1

    def add_point(self, point, i_surf, info):
        """
        :param point:
        :param i_surf:
        :param info:
        :return:
        """

        point.connected = 1
        self.line[self.curve_max_id].append(point)
        self.line_info[self.curve_max_id].append(info)
        self.line_surf[self.curve_max_id].append(i_surf)


    def _make_point_orderings(self, point_list, patch_points):
        """
        TODO: split into smaller functions.
        :param point_list:
        :param patch_points:
        :return:
        """

        """
        for surf_id, surf_points in enumerate(point_list):
            for point in surf_points:
                assert point[0].surface_point[0].surf == self.surf[surf_id]
                assert point[0].own_point.surf == self.surf[surf_id]
                ... 
        """

        for i_surf in range(0, 2):
            for point in point_list[i_surf]:
                if point.connected == 1:
                    continue

                # point will be uses as start point

                # initialization of the curve
                self.line.append([])
                self.line_info.append([])
                self.line_surf.append([])
                end_found = np.zeros([2])
                self.curve_max_id += 1

                self.add_point(point, i_surf, 0)  # "n_addepts  = 0" should be rewritten after reverse

                while end_found[1] == 0:

                    # search all patches where the last point live in
                    own_isec_point = self.line[self.curve_max_id][-1]
                    own_isec_points, other_isec_points = self._find_neighbours(own_isec_point, i_surf, patch_points)

                    print(len(other_isec_points), len(own_isec_points))

                    if np.logical_and(len(other_isec_points) == 0, len(own_isec_points) == 0):
                        if end_found[0] == 0:
                            end_found[0] = 1
                            self.line[self.curve_max_id].reverse()
                            self.line_info[self.curve_max_id].reverse()
                            self.line_surf[self.curve_max_id].reverse()
                            continue
                        else:
                            end_found[1] = 1
                            break

                    if len(other_isec_points) > 0:
                        point = other_isec_points[0]
                        i_surf = 1 - i_surf
                        n_points = len(other_isec_points)
                    elif len(own_isec_points) > 0:
                        point = own_isec_points[0]
                        n_points = len(own_isec_points)

                    self.add_point(point, i_surf, n_points)
