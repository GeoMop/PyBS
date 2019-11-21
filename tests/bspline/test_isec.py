from bgem.bspline import bspline as bs, isec_surf_surf as iss, bspline_plot as bp, bspline_approx as bsa
import numpy as np
import math


class xTestSurface:

    def plot_extrude(self):
        #fig1 = plt.figure()

        #ax1 = fig1.gca(projection='3d')



        def function(x):
            return math.sin(x[0]*4) * math.cos(x[1] * 4)

        def function2(x):
            return math.cos(x[0]*4) * math.sin(x[1] * 4)

        def function3(x):
            return (-x[0] + x[1] + 4 + 3 + math.cos(3 * x[0]))

        def function4(x):
            return (2 * x[0] - x[1] + 3 + math.cos(3 * x[0]))

        u1_int = 4
        v1_int = 4
        u2_int = 4
        v2_int = 4

        u_basis = bs.SplineBasis.make_equidistant(2, u1_int) #10
        v_basis = bs.SplineBasis.make_equidistant(2, v1_int) #15
        u2_basis = bs.SplineBasis.make_equidistant(2, u2_int) #10
        v2_basis = bs.SplineBasis.make_equidistant(2, v2_int) #15
        poles = bs.make_function_grid(function, u1_int + 2, v1_int + 2) #12, 17
        surface_extrude = bs.Surface((u_basis, v_basis), poles)

        myplot = bp.Plotting((bp.PlottingPlotly()))
        #myplot.plot_surface_3d(surface_extrude, poles = False)
        poles2 = bs.make_function_grid(function2,  u2_int + 2, v2_int + 2) #12, 17
        surface_extrude2 = bs.Surface((u2_basis, v2_basis), poles2)
        #myplot.plot_surface_3d(surface_extrude2, poles=False)

        m = 100
        fc = np.zeros([m * m, 3])
        fc2 = np.empty([m * m, 3])
        a = 5
        b = 7
        #print(fc)

        for i in range(m):
            for j in range(m):
                #print([i,j])
                x = i / m * a
                y = j / m * b
                z = function3([x, y])
                z2 = function4([x, y])
                fc[i + j * m, :] = [x, y, z]
                fc2[i + j * m, :] = [x, y, z2]

        #print(fc)

        #gs = bs.GridSurface(fc.reshape(-1, 3))
        #gs.transform(xy_mat, z_mat)
        #approx = bsa.SurfaceApprox.approx_from_grid_surface(gs)




        approx = bsa.SurfaceApprox(fc)
        approx2 = bsa.SurfaceApprox(fc2)
        surfz = approx.compute_approximation(nuv=np.array([11, 26]))
        surfz2 = approx2.compute_approximation(nuv=np.array([20, 16]))
        #surfz = approx.compute_approximation(nuv=np.array([3, 5]))
        #surfz2 = approx2.compute_approximation(nuv=np.array([2, 4]))
        surfzf = surfz.make_full_surface()
        surfzf2 = surfz2.make_full_surface()


        myplot.plot_surface_3d(surfzf, poles=False)
        myplot.plot_surface_3d(surfzf2, poles=False)

        #return surface_extrude, surface_extrude2, myplot
        return surfzf, surfzf2, myplot



    #def plot_function(self):
    #    fig = plt.figure()
    #    ax = fig.gca(projection='3d')

        # function surface
    #    def function(x):
    #        return math.sin(x[0]) * math.cos(x[1])

        #poles = bs.make_function_grid(function, 4, 5)
        #u_basis = bs.SplineBasis.make_equidistant(2, 2)
        #v_basis = bs.SplineBasis.make_equidistant(2, 3)
        #surface_func = bs.Surface( (u_basis, v_basis), poles)
        #bs_plot.plot_surface_3d(surface_func, ax)
        #bs_plot.plot_surface_poles_3d(surface_func, ax)

        #plt.show()


    #def boudingbox(self):
    #    SurfSurf = IsecSurfSurf.bounding_boxes()
    #@pytest.mark.skip
    def test_isec(self):


        surf1, surf2, myplot = self.plot_extrude()
        isec = iss.IsecSurfSurf(surf1, surf2)
        #box1, tree1 = isec.bounding_boxes(surf1)
        #box2, tree2 = isec.bounding_boxes(surf2)
        #isec.get_intersection(surf1,surf2,tree1,tree2,box1,box2,isec.nt,isec.nit) # surf1,surf2,tree1,tree2
        point_list1, point_list2 = isec.get_intersection()

        m = point_list1.__len__() + point_list2.__len__()

        X = np.zeros([m])
        Y = np.zeros([m])
        Z = np.zeros([m])

        i = -1

        for point in point_list1:
            i += 1
            X[i] = point.xyz[0]
            Y[i] = point.xyz[1]
            Z[i] = point.xyz[2]

        for point in point_list2:
            i += 1
            X[i] = point.xyz[0]
            Y[i] = point.xyz[1]
            Z[i] = point.xyz[2]



        myplot.scatter_3d(X, Y, Z)


        myplot.show() # view

        #print(tree1.find_box(boxes2[0]))
        #print(surf1.poles[:,:,1])
        #print(surf1.u_basis.n_intervals)
        #print(surf1.u_basis.knots)


class SurfApprox:

    def __init__(self, plane_coefficients, x_length, y_length, x_n_samples, y_n_samples, x_n_control_points, y_n_control_points, additive_function):
        """
         :param plane_normal_vector: plane vector as numpy array 4x1
         :param x_size: x range as double
         :param y_size: y range as double
         :param x_n_samples: number of sample points in x range as double
         :param y_n_samples: number of sample points in y range as double
         :param x_n_control_points: x control points as double
         :param y_n_control_points: y control points as double
         :return:
         """
        self.plane_coefficients = plane_coefficients
        self.x_length = x_length
        self.y_length = y_length
        self.x_n_samples = x_n_samples
        self.y_n_samples = y_n_samples
        self.x_n_control_points = x_n_control_points
        self.y_n_control_points = y_n_control_points
        self.err = None
        self.surfz = None
        self.additive_function = additive_function

        def surfzf(X):
            return -(self.plane_coefficients[0] * X[0] + self.plane_coefficients[1] * X[1] + self.plane_coefficients[3]) / self.plane_coefficients[2] + self.additive_function(3 * X[0])

        self.surfzf = surfzf

        self.err, self.surfz = self.surf_app()

    def surf_app(self):

        fc = np.zeros([self.x_n_samples * self.y_n_samples, 3])
        for i in range(self.x_n_samples):
            for j in range(self.y_n_samples):
                fc[i + j * self.y_n_samples, 0:2] = [i / self.x_n_samples * self.x_length, j / self.y_n_samples * self.y_length]
                fc[i + j * self.y_n_samples, 2] = self.surfzf(fc[i + j * self.y_n_samples, 0:2])

        approx = bsa.SurfaceApprox(fc)
        surfz = approx.compute_approximation(nuv=np.array([self.x_n_control_points, self.y_n_control_points]))
        err = approx.error
        surfzf = surfz.make_full_surface()
        return err, surfzf


class TestIsecx:

    def test(self):
        plane_coefficients1 = np.array([-1, 1, -1, 7])
        plane_coefficients2 = np.array([2, -1, -1, 3])

        def cosx(x):
            return math.cos(3 * x)

        x_length = 5
        y_length = 7
        #x_n_samples = 100
        #y_n_samples = 100
        #x1_n_control_points = 11
        #y1_n_control_points = 26
        #x2_n_control_points = 20
        #y2_n_control_points = 16


        x_n_samples = 20
        y_n_samples = 20


        #x1_n_control_points = 11
        #y1_n_control_points = 10
        #x2_n_control_points = 13
        #y2_n_control_points = 11

        x1_n_control_points = 11
        y1_n_control_points = 10
        x2_n_control_points = 10
        y2_n_control_points = 11


        sapp1 = SurfApprox(plane_coefficients1, x_length, y_length, x_n_samples, y_n_samples, x1_n_control_points, y1_n_control_points, cosx)
        sapp2 = SurfApprox(plane_coefficients2, x_length, y_length, x_n_samples, y_n_samples, x2_n_control_points, y2_n_control_points, cosx)
        ist = IntersectTest(sapp1, sapp2)
        ist.test_isec()


class IntersectTest:

    def __init__(self, surfapprox, surfapprox2):
        self.surfapprox = surfapprox
        self.surfapprox2 = surfapprox2


    def plot(self):

        myplot = bp.Plotting((bp.PlottingPlotly()))
        myplot.plot_surface_3d(self.surfapprox.surfz, poles=False)
        myplot.plot_surface_3d(self.surfapprox2.surfz, poles=False)
        #myplot.scatter_3d(X, Y, Z)
        myplot.show() # view
        #return surfzf, surfzf2, myplot, err

    def test_isec(self):


        plane_coefficients1 = self.surfapprox.plane_coefficients
        plane_coefficients2 = self.surfapprox2.plane_coefficients
        additive_function = self.surfapprox.additive_function

        a = plane_coefficients1[0] - plane_coefficients2[0]
        b = plane_coefficients1[1] - plane_coefficients2[1]
        d = plane_coefficients1[3] - plane_coefficients2[3]


        def xp(y):
            return -(b * y + d) / a

        def yp(x):
            return -(a * x + d) / b


        def srf(x):
            xpp = xp(x[1])
            ypp = yp(x[0])
            x_avg = (xpp + x[0]) / 2
            y_avg = (ypp + x[1]) / 2
            X = np.array([x_avg, y_avg, 0])
            X[2] = self.surfapprox.surfzf(X[0:2])
            return X

        self.plot()

        isec = iss.IsecSurfSurf(self.surfapprox.surfz, self.surfapprox2.surfz)
        point_list1, point_list2 = isec.get_intersection()



        tol = self.surfapprox.err + self.surfapprox2.err
        points = point_list1 + point_list2

        #print(tol)
        for point in points:
            #print(np.linalg.norm(point.xyz - srf(point.xyz)))
            assert(np.linalg.norm(point.xyz - srf(point.xyz)) < tol), "intersection point tolerance has been exceeded."





