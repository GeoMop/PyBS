import numpy as np
import bspline as bs
import isec_point as IP
import surface_point as SP
import isec_curve_surf as ICS
import numpy.linalg as la


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

        self._ipoint_list = []  # append
        # tolerance


    def get_intersection(self):
        """
        Main method to get intersection points
        :return:
        """
        point_list1 = self.get_intersections(self.surf1, self.surf2)  # patches of surf 2 with respect threads of the surface 1
        point_list2 = self.get_intersections(self.surf2, self.surf1) # patches of surf 1 with respect threads of the surface 2

        #print('points')
        #for point in point_list1:
        #    print(point.uv1)
        #print('points2')
        #for point in point_list2:
        #    print(point.uv2)
        #print(point_list1.__len__(),point_list2.__len__())

        #assert point_list1.__len__() == 9
        #assert point_list2.__len__() == 9

        connected_points = self._connect_points(point_list1, point_list2)
        print(len(point_list1))
        print(len(point_list2))

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

    def get_intersections(self, surf1, surf2):
        """
        Tries to compute intersection of the main curves from surface1 and patches of the surface2 which have a
         non-empty intersection of corresponding bonding boxes
        :param surf1: Surface used to construction of the main threads
        :param surf2: Intersected surface
        :param tree2: Bih tree of the patches of the surface 2
        :return: point_list as list of points of intersection
        """

        tree2 = surf2.tree

        point_list = []
        crossing = np.zeros([surf1.u_basis.n_intervals + 1, surf1.v_basis.n_intervals + 1])

        for axis in [IP.Axis.u, IP.Axis.v]:
            curves, w_val, patch = self._main_curves(surf1, axis)
            curve_id = -1
            for curve in curves:
                curve_id += 1
                #interval_intersections = 0
                for it in range(curve.basis.n_intervals):
                    if self._already_found(crossing, it, curve_id, axis) == 1: #?
                        print('continue')
                        continue
                    curv_surf_isec = ICS.IsecCurveSurf(surf2, curve)
                    intersectioned_patches2 = tree2.find_box(curve.boxes[it])
                    for ipatch2 in intersectioned_patches2:
                        iu2, iv2 = surf2.patch_id2pos(ipatch2)
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
                            surf_point_a = SP.SurfacePoint(surf1, iuv_a, uv_a)

                            # Point B
                            uv_b = uvt[0:2]
                            iuv_b = np.array([iu2, iv2])
                            surf_point_b = SP.SurfacePoint(surf2, iuv_b, uv_b)

                            point = IP.IsecPoint(surf_point_a, surf_point_b, xyz)

                            point_list.append(point)

                            #if interval_intersections == 0:
                            #    point_list.append(point)
                            #    interval_intersections += 1
                            #else:
                            #    a = 1
                                #check


                            direction = surf_point_a.interface_flag[1-axis]
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

    def _connect_points(self, point_list1, point_list2):

        patch_point_list1, boundary_points1 = self.make_patch_point_list(point_list1, 0)
        patch_point_list2, boundary_points2 = self.make_patch_point_list(point_list2, 0)
        patch_point_list1i, boundary_points1i = self.make_patch_point_list(point_list2, 1)
        patch_point_list2i, boundary_points2i = self.make_patch_point_list(point_list1, 1)

        # print(boundary_points2)

        patch_point_list = []
        patch_point_listi = []
        boundary_points = []
        point_list = []

        #  boundary_points[0].append(1)
        #  print(boundary_points )
        #  print(boundary_points[0] )
        #  print(boundary_points[1] )
        #  print('done')

        point_list.append(point_list1)
        point_list.append(point_list2)
        patch_point_list.append(patch_point_list1)
        patch_point_list.append(patch_point_list2)
        patch_point_listi.append(patch_point_list1i)
        patch_point_listi.append(patch_point_list2i)
        boundary_points.append(boundary_points1)
        boundary_points.append(boundary_points2)


        line = []
        #line = self._make_orderings(point_list,patch_point_list, boundary_points)
        line = self._make_point_orderings(point_list, patch_point_list, patch_point_listi, boundary_points)

        #print(line)
        #curve_from_grid
        # test funkce pro ruzne parametry - moznosti volby S, kontrola tolerance - jak? task? pocet knotu



        for point_lists in point_list:
            print(len(point_lists))
            for points in point_lists:
                print(points.xyz)
        for lines in line:
            print(len(lines))
            print("line")
            for points in lines:
                print(points.xyz)


        return line

    # @staticmethod
    # def make_patch_point_list(point_list, surf_id):
    #     """
    #     creates conversion list such that patch_points[patch_ID] give list of positions to point_list,
    #     :param point_list: as list of intersection points
    #     :param surf_id:  0 or 1
    #     :return:
    #     """
    #
    #     surf = point_list[0].surface_point[surf_id].surf
    #
    #     list_len = surf.u_basis.n_intervals * surf.v_basis.n_intervals
    #     patch_points = []
    #     boundary_points = []
    #
    #     for i in range(list_len):
    #         patch_points.append([])
    #
    #     point_pos = -1
    #     for point in point_list:
    #         point_pos += 1
    #         patch_id = point.surface_point[surf_id].patch_id()
    #         for patch in patch_id:
    #             patch_points[patch].append(point_pos)
    #         if point.surface_point[0].surface_boundary_flag == 1:
    #             boundary_points.append(point_pos)
    #
    #     return patch_points, boundary_points

    @staticmethod
    def make_patch_point_list(point_list, surf_id):
        """
        creates conversion list such that patch_points[patch_ID] give list of positions to point_list,
        :param point_list: as list of intersection points
        :param surf_id:  0 or 1
        :return:
        """

        surf = point_list[0].surface_point[surf_id].surf

        list_len = surf.u_basis.n_intervals * surf.v_basis.n_intervals
        patch_points = []
        boundary_points = []

        for i in range(list_len):
            patch_points.append(set())

        point_pos = int(-1)
        for point in point_list:
            point_pos += 1
            patch_id = point.surface_point[surf_id].patch_id()
            for patch in patch_id:
                patch_points[patch].add(point_pos)
            if point.surface_point[0].surface_boundary_flag == 1:
                boundary_points.append(point_pos)

        return patch_points, boundary_points


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




    @staticmethod
    def _unconnected_patch_points(point_list, patch_point_list, i_surf, patch_id):
        """

        :param point_list: list of all points
        :param patch_point_list:
        :param i_surf: index of the surface "0" or "1"
        :param patch_id: as numpy array of integers
        :return:
        """

        #next_point = []
        next_point = set()
        for p_id in patch_id:
            next_point.union(patch_point_list[i_surf][p_id])

        connected = set()
        for pos in next_point:
             if point_list[i_surf][pos].connected == 1:
                connected.add(pos)

        next_point -= connected

        return list(next_point)

        #     for point_pos in patch_point_list[i_surf][p_id]:
        #
        #
        #         if point_list[i_surf][point_pos].connected == 0:
        #             found = 0
        #             for pids in next_point:
        #                 if pids == point_pos:
        #                     found == 1
        #             if found == 0:
        #                 next_point.append(point_pos)
        #
        # return next_point

    @staticmethod
    def _get_inside_points(point_list, patch_point_list, i_surf, patch_id):
        """

        :param point_list: list of all points
        :param patch_point_list:
        :param i_surf: index of the surface "0" or "1"
        :param patch_id: as numpy array of integers
        :return:
        """

        next_point = []
        for p_id in patch_id:
            for point_pos in patch_point_list[i_surf][p_id]:
                if point_list[i_surf][point_pos].connected == 0:
                    found = 0
                    for pids in next_point:
                        if pids == point_pos:
                            found == 1
                    if found == 0:
                        next_point.append(point_pos)

        return next_point

    @staticmethod
    def _free_point(self, point_list):

        i = -1
        for point in point_list:
            i += 1
            if point.connected == 0:
                return i
        return -1




    def check_duplicities(self, surfpoint1, surfpoint2):
        """
        :param surfpoint1:
        :param surfpoint2:
        :return:
        """
        pid1 = surfpoint1.patch_id()
        pid2 = surfpoint2.patch_id()

        pid = pid1 & pid2

        if len(pid) > 0:
            return la.norm(surfpoint1.uv - surfpoint2.uv)

        return -1


    def _make_point_orderings(self, point_list, patch_point_list, patch_point_listi,boundary_points):
        """
        TODO: split into smaller functions.
        :param point_list:
        :param patch_point_list:
        :param boundary_points:
        :return:
        """
        n_curves = 0
        i_surf = 0
        line = []
        line_info = []
        line_surf = []

        """
        for surf_id, surf_points in enumerate(point_list):
            for point in surf_points:
                assert point[0].surface_point[0].surf == self.surf[surf_id]
                assert point[0].own_point.surf == self.surf[surf_id]
                ... 
        """
        while self._free_point(point_list[i_surf]) != -1:

            line.append([])
            line_info.append([])
            line_surf.append([])


             # obtain start point
            pos = self._free_point(point_list[i_surf])
            if len(pos != 1):
                end_found = np.zeros([2])
            else:
                return

            point = point_list[i_surf][pos]
            point.connected = 1
            line[n_curves].append(point)
            line_info[n_curves].append(0)
            line_surf[n_curves].append(i_surf)

            while end_found[1] == 0:

                p1id = line[n_curves][-1].surface_point[0].patch_id()
                p2id = line[n_curves][-1].surface_point[1].patch_id()
                p1pos = self._unconnected_patch_points(point_list, patch_point_list, i_surf, p1id)
                point1 = line[n_curves][-1]
                surfpoint1 = line[n_curves][-1].surface_point[0]

                # points with respect second surface
                p2pos  = []
                point2_pos = self._unconnected_patch_points(point_list, patch_point_list, 1-i_surf, p2id)
                for pos2 in point2_pos:
                    point2 = point_list[1 - i_surf][pos2]

                    # !!! Do surface_point indexovat pouze 0 nebo 1 jako own nebo other
                    # asi vhodne do IsecPoint zavest dva samostatne atrbuty own_point a other_point
                    patch1_id_temp = point2.surface_point[1-i_surf].patch_id()
                    if len(patch1_id_temp & p1id) > 0:
                        if self.check_duplicities(surfpoint1,point2.surface_point[1-i_surf]) > 0.00001:
                            p2pos.append(pos2)
                        else:
                            point2.duplicite_with = point1
                            point2.connected = 1
                            point1.duplicite_with = point2


                if len(p2pos) == 0:
                    if len(p1pos) == 0:
                        if end_found[0] == 0:
                            end_found[0] = 1
                            line[n_curves].reverse()
                            line_info[n_curves].reverse()
                            line_surf[n_curves].reverse()
                        elif end_found[0] == 1:
                            end_found[1] = 1
                    else:

                    #zustavam na povrchu
                elif len(p2pos) == 1:
                    point2 = point_list[1 - i_surf][p2pos[0]]
                    point2.connected = 1
                    line[n_curves].append(point2)
                    line_info[n_curves].append(1)
                    line_surf[n_curves].append(1-i_surf)
                elif len(p2pos) == 2:
                    print("problem")

                last_point = line[n_curves][-1]
                pid2 = last_point.surface_point[1].patch_id()

                if len(p2pos > 0):

                inside_point_pos = self._get_inside_points(point_list, patch_point_listi, i_surf, pid)
                n_points = len(p1pos)

                print(n_points)


                line_info[n_curves].append(n_points)
                if n_points == 0:
                    if end_found[0] == 0:
                        end_found[0] = 1
                        line[n_curves].reverse()
                        line_info[n_curves].reverse()
                    elif end_found[0] == 1:
                        end_found[1] = 1
                elif n_points == 1:
                    point = point_list[i_surf][p1pos[0]]
                    point.connected = 1
                    line[n_curves].append(point)
                # out
                elif (np.logical_and(n_points >= 2, n_points <= 4) == 1):
                    k = -1
                    #print("pid")
                    #print(pid)
                    #print("p1pos")
                    #print(p1pos)
                    for pos in p1pos:
                        patch_id = point_list[i_surf][pos].surface_point[0].patch_id()
                        #print("patch ID")
                        #print(patch_id)
                        k += 1
                        found = np.zeros([len(p1pos)])
                        for patches in patch_id:
                            for patches_prev in pid:
                                if patches == patches_prev:
                                    found[k] += 1

                    # interface curve
                    already_found = 0
                    #for i in range(2): # mistake?
                    for i in range(n_points):
                        if found[i] == 2:
                            point = point_list[i_surf][p1pos[i]]
                            if np.logical_and(point.surface_point[0].surface_boundary_flag == 0, len(point.surface_point[0].patch_id()) == 2):
                                point.connected = 1
                                line[n_curves].append(point)
                                #found = np.zeros([len(p1pos)])
                                already_found = 1
                                break

                    if  already_found == 0:
                        for i in range(2):
                            if found[i] == 1:
                                point = point_list[i_surf][p1pos[i]]
                                point.connected = 1
                                line[n_curves].append(point)
                                break
                    #break

            n_curves += 1
            print('start')
            for points in line[n_curves-1]:
                print(points.surface_point[1].surface_boundary_flag)
            print('stop')
            print('start')
            for points in line_info[n_curves-1]:
                print(points)
            print('stop')

        return line


    def _make_point_orderingsx(self, point_list, patch_point_list, boundary_points):
        """
        TODO: split into smaller functions.
        :param point_list:
        :param patch_point_list:
        :param boundary_points:
        :return:
        """
        n_curves = 0
        i_surf = 0
        line = []
        line_info = []



        while self._free_point(point_list[i_surf]) != -1:

            line.append([])
            line_info.append([])
            end_found = np.zeros([2])
             # obtain start point
            pos = self._free_point(point_list[i_surf])
            point = point_list[i_surf][pos]

            point.connected = 1
            line[n_curves].append(point)
            line_info[n_curves].append(0)



            while end_found[1] == 0:

                pid = line[n_curves][-1].surface_point[0].patch_id()
                #pid = point.surface_point[0].patch_id()
                p1pos = self._get_out_points(point_list, patch_point_list, i_surf, pid)
                n_points = len(p1pos)
                print(n_points)
                line_info[n_curves].append(n_points)
                if n_points == 0:
                    if end_found[0] == 0:
                        end_found[0] = 1
                        line[n_curves].reverse()
                        line_info[n_curves].reverse()
                    elif end_found[0] == 1:
                        end_found[1] = 1
                elif n_points == 1:
                    point = point_list[i_surf][p1pos[0]]
                    point.connected = 1
                    line[n_curves].append(point)
                # out
                elif (np.logical_and(n_points >= 2, n_points <= 4) == 1):
                    k = -1
                    #print("pid")
                    #print(pid)
                    #print("p1pos")
                    #print(p1pos)
                    for pos in p1pos:
                        patch_id = point_list[i_surf][pos].surface_point[0].patch_id()
                        #print("patch ID")
                        #print(patch_id)
                        k += 1
                        found = np.zeros([len(p1pos)])
                        for patches in patch_id:
                            for patches_prev in pid:
                                if patches == patches_prev:
                                    found[k] += 1

                    # interface curve
                    already_found = 0
                    #for i in range(2): # mistake?
                    for i in range(n_points):
                        if found[i] == 2:
                            point = point_list[i_surf][p1pos[i]]
                            if np.logical_and(point.surface_point[0].surface_boundary_flag == 0, len(point.surface_point[0].patch_id()) == 2):
                                point.connected = 1
                                line[n_curves].append(point)
                                #found = np.zeros([len(p1pos)])
                                already_found = 1
                                break

                    if  already_found == 0:
                        for i in range(2):
                            if found[i] == 1:
                                point = point_list[i_surf][p1pos[i]]
                                point.connected = 1
                                line[n_curves].append(point)
                                break
                    #break

            n_curves += 1
            print('start')
            for points in line[n_curves-1]:
                print(points.surface_point[1].surface_boundary_flag)
            print('stop')
            print('start')
            for points in line_info[n_curves-1]:
                print(points)
            print('stop')

        return line


    # @staticmethod


    '''
    Calculation and representation of intersection of two B-spline surfaces.

    Result is set of B-spline segments approximating the intersection branches.
    Every segment consists of: a 3D curve and two 2D curves in parametric UV space of the surfaces.
    '''
