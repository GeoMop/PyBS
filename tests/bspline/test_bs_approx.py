import numpy as np
from bgem.bspline import bspline as bs, bspline_plot as bs_plot, bspline_approx as bs_approx
import math
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import time
import logging


def function_sin_cos(x):
    return math.sin(x[0] * 4) * math.cos(x[1] * 4)


def gen_uv_grid(nu, nv):
    # surface on unit square
    U = np.linspace(0.0, 1.0, nu)
    V = np.linspace(0.0, 1.0, nv)
    V_grid, U_grid = np.meshgrid(V, U)

    return np.stack([U_grid.ravel(), V_grid.ravel()], axis=1)

def eval_func_on_grid(func, xy_mat, xy_shift,  shape=(50, 50)):
    nu, nv = shape
    UV = gen_uv_grid(nu, nv)
    XY = xy_mat.dot(UV.T).T + xy_shift
    #z_func_eval = np.array([z_mat[0] * func([u, v]) + z_mat[1] for u, v in UV])
    z_func_eval = np.array([func([u, v]) for u, v in UV], dtype=float)
    return  np.concatenate((XY, z_func_eval[:, None]), axis=1).reshape(nu, nv, 3)


def eval_z_surface_on_grid(surface, xy_mat, xy_shift, shape=(50, 50)):
    nu, nv = shape
    UV = gen_uv_grid(nu, nv)
    XY = xy_mat.dot(UV.T).T + xy_shift
    Z = surface.z_eval_xy_array(XY)
    return np.concatenate((XY, Z[:, None]), axis=1).reshape(nu, nv, 3)


def eval_surface_on_grid(surface, shape=(50, 50)):
    nu, nv = shape
    UV = gen_uv_grid(nu, nv)
    return surface.eval_array(UV).reshape(nu, nv, 3)

def plot_cmp(a_grid, b_grid):
    plt = bs_plot.Plotting()
    plt.scatter_3d(a_grid[:, :, 0], a_grid[:, :, 1], a_grid[:, :, 2])
    plt.scatter_3d(b_grid[:, :, 0], b_grid[:, :, 1], b_grid[:, :, 2])
    plt.show()

    diff = b_grid - a_grid
    plt.scatter_3d(a_grid[:, :, 0], a_grid[:, :, 1], diff[:, :, 0])
    plt.scatter_3d(a_grid[:, :, 0], a_grid[:, :, 1], diff[:, :, 1])
    plt.scatter_3d(a_grid[:, :, 0], a_grid[:, :, 1], diff[:, :, 2])
    plt.show()

def grid_cmp( a, b, tol):
    a_z = a[:, :, 2].ravel()
    b_z = b[:, :, 2].ravel()
    eps = 0.0
    n_err=0
    for i, (za, zb) in enumerate(zip(a_z, b_z)):
        diff = np.abs( za - zb)
        eps = max(eps, diff)
        if diff > tol:
            n_err +=1
            if n_err < 10:
                print(" {} =|a({}) - b({})| > tol({}), idx: {}".format(diff, za, zb, tol, i) )
            elif n_err == 10:
                print("... skipping")
    print("Max norm: ", eps, "Tol: ", tol)
    if eps > tol:
        plot_cmp(a, b)
        assert False, "Different surfaces."

class TestSurfaceApprox:
    # todo: numerical test


    def test_approx_func(self):
        logging.basicConfig(level=logging.DEBUG)

        xy_mat = np.array( [ [1.0, -1.0, 10 ], [1.0, 1.0, 20 ]])    # rotate left pi/4 and blow up 1.44
        z_mat = np.array( [2.0, -10] )


        # print("Compare: Func - GridSurface.transform")
        # points = bs.make_function_grid(function_sin_cos, 200, 200)
        # gs = bs.GridSurface(points.reshape(-1, 3))
        # gs.transform(xy_mat, z_mat)
        # xyz_grid = eval_z_surface_on_grid(gs, xy_mat[:2,:2], xy_mat[:, 2])
        # grid_cmp(xyz_func, xyz_grid, 0.02)
        #
        # print("Compare: Func - GridSurface.transform.z_surf")
        # xyz_grid = eval_z_surface_on_grid(gs.z_surface, xy_mat[:2, :2], xy_mat[:, 2])
        # xyz_grid[:,:,2] *= z_mat[0]
        # xyz_grid[:, :, 2] += z_mat[1]
        # grid_cmp(xyz_func, xyz_grid, 0.02)

        print("\nCompare: Func - GridSurface.approx")
        points = bs.make_function_grid(function_sin_cos, 50, 50)
        gs = bs.GridSurface(points.reshape(-1, 3))
        xy_center = gs.center()[0:2]
        z_center = gs.center()[2]
        gs.transform(xy_mat, z_mat)
        approx = bs_approx.SurfaceApprox.approx_from_grid_surface(gs)
        surface = approx.compute_approximation()

        xy_shift = xy_mat[:, 2] - np.dot(xy_mat[:2, :2], xy_center) + xy_center
        xyz_grid = eval_z_surface_on_grid(surface, xy_mat[:2, :2], xy_shift)

        xyz_func = eval_func_on_grid(function_sin_cos, xy_mat[:2, :2], xy_mat[:, 2])
        xyz_func[:, :, 2] -= z_center
        xyz_func[:, :, 2] *= z_mat[0]
        xyz_func[:, :, 2] += z_mat[1] + z_center
        print("Approx error: ", approx.error)
        grid_cmp(xyz_func, xyz_grid, 0.02)

        print("\nCompare: Func - points.approx")
        np.random.seed(seed=123)
        uv = np.random.rand(1000,2)
        xy = xy_mat[:2,:2].dot(uv.T).T + xy_mat[:, 2]
        z = np.array( [function_sin_cos([u, v])  for u, v in uv] )
        xyz = np.concatenate((xy, z[:, None]), axis=1)
        approx = bs_approx.SurfaceApprox(xyz)
        quad = approx.compute_default_quad()

        nuv = approx.nuv
        ref_quad = np.array([  [-1 , 1], [0,0], [1, 1],  [0, 2] ])
        ref_quad += np.array([10, 20])
        assert np.allclose(ref_quad, quad, atol=1e-2)

        nuv = approx.compute_default_nuv()
        assert np.allclose( np.array([8, 8]), nuv)

        surface = approx.compute_approximation()
        z_center = surface.center()[2]
        surface.transform(xy_mat = None, z_mat=z_mat)
        nu, nv = 50, 50
        uv_probe = gen_uv_grid(nu, nv)
        uv_probe = (0.9*uv_probe + 0.05)
        xy_probe = xy_mat[:2,:2].dot(uv_probe.T).T + xy_mat[:, 2]
        z_func = np.array( [function_sin_cos([u, v])  for u, v in uv_probe] )
        z_func -= z_center
        z_func *= z_mat[0]
        z_func += z_mat[1] + z_center
        xyz_func = np.concatenate((xy_probe, z_func[:, None]), axis=1).reshape(nu, nv, 3)
        xyz_approx = surface.eval_xy_array(xy_probe).reshape(nu, nv, 3)
        print("Approx error: ", approx.error)
        grid_cmp(xyz_func, xyz_approx, 0.02)

        # approx = bs_approx.SurfaceApprox(xyz)
        # surface = approx.compute_approximation()
        #
        #
        # plt = bs_plot.Plotting()
        # plt.plot_surface_3d(gs)
        # plt.plot_surface_3d(surface)
        # plt.scatter_3d(xyz[:,0], xyz[:,1], xyz[:,2])
        # plt.show()

        # xyz_grid = eval_z_surface_on_grid(z_surf, xy_mat[:2,:2], xy_mat[:, 2])
        # grid_cmp(xyz_func, xyz_grid, 0.02)
        #
        # print("Compare: Func - GridSurface.transform.Z_surf_approx.full_surface")
        # xyz_grid = eval_surface_on_grid(z_surf.make_full_surface())
        # grid_cmp(xyz_func, xyz_grid, 0.01)

    def test_approx_real_surface(self):
        # Test approximation of real surface grid using a random subset
        #  hard test of regularization.
        logging.basicConfig(level=logging.DEBUG)

        xy_mat = np.array( [ [1.0, 0.0, 0 ], [0.0, 1.0, 0 ]])    # rotate left pi/4 and blow up 1.44
        #z_mat = np.array( [1.0, 0] )
        xyz_func = eval_func_on_grid(function_sin_cos, xy_mat[:2,:2], xy_mat[:, 2])

        print("Compare: Func - Randomized.approx")
        points = bs.make_function_grid(function_sin_cos, 50, 50)
        points = points.reshape( (-1, 3) )
        n_sample_points= 400 # this is near the limit number of points to keep desired precision
        random_subset = np.random.random_integers(0, len(points)-1, n_sample_points)
        points_random = points[random_subset, :]

        approx = bs_approx.SurfaceApprox(points_random)
        approx.set_quad(None)   # set unit square
        surface = approx.compute_approximation()
        xyz_grid = eval_z_surface_on_grid(surface, xy_mat[:2, :2], xy_mat[:, 2])
        print("Approx error: ", approx.error)
        grid_cmp(xyz_func, xyz_grid, 0.02)


    # def test_transformed_quad(self):
    #     xy_mat = np.array( [ [1.0, -1.0, 0 ], [1.0, 1.0, 0 ]])    # rotate left pi/4 and blow up 1.44
    #     np.random.seed(seed=123)
    #     uv = np.random.rand(1000,2)
    #     xy = xy_mat[:2,:2].dot(uv.T).T + xy_mat[:, 2]
    #     z = np.array( [function_sin_cos([u, v])  for u, v in uv] )
    #     xyz = np.concatenate((xy, z[:, None]), axis=1)
    #     approx = bs_approx.SurfaceApprox(xyz)
    #     approx.quad = np.array([  [-1 , 1], [0,0], [1, 1],  [0, 2] ])
    #
    #     xy_mat = np.array([[2, 1, -1],[1, 2, -2]])
    #     assert np.allclose( np.array([  [-2 , -1], [-1,-2], [2, 1],  [1, 2] ]), approx.transformed_quad(xy_mat))


    def plot_approx_transformed_grid(self):
        pass

    def plot_plane(self):
        surf = bs_approx.plane_surface([ [0.0, 0, 0], [1.0, 0, 0], [0.0, 0, 1] ], overhang=0.1)
        self.plot_surf(surf)

    # def test_approx_speed(self):
    #     logging.basicConfig(level=logging.DEBUG)
    #
    #     print("Performance test for 100k points.")
    #     np.random.seed(seed=123)
    #     uv = np.random.rand(100000,2)
    #     xy = uv
    #     z = np.array( [function_sin_cos([u, v])  for u, v in uv] )
    #     xyz = np.concatenate((xy, z[:, None]), axis=1)
    #     start_time = time.time()
    #     approx = bs_approx.SurfaceApprox(xyz)
    #     surface = approx.compute_approximation()
    #     end_time = time.time()
    #     print("\nApprox 100k points by 100x100 grid in: {} sec".format(end_time - start_time))
    #     assert end_time - start_time < 6
    #     # target is approximation of 1M points in one minute
    #     # B matrix 3.6 sec, A matrix 0.7 sec, SVD + Z solve 0.7 sec

class TestBoundingBox:
    def test_hull_and_box(self):
        points = np.random.randn(1000000,2)

        print()
        start = time.perf_counter()
        for i in range(1):
            hull = bs_approx.convex_hull_2d(points)
        end = time.perf_counter()
        print("\nConvex hull of 1M points: {} s".format(end - start))

        start = time.perf_counter()
        for i in range(10):
            quad = bs_approx.min_bounding_rect(hull)
        end = time.perf_counter()
        print("Min area bounding box: {} s".format(end - start))
        return

        plt = bs_plot.Plotting()
        plt.scatter_2d(points[:,0], points[:,1])
        plt.plot_2d(hull[:, 0], hull[:, 1])
        box_lines = np.concatenate((quad, quad[0:1,:]), axis=0)
        plt.plot_2d(box_lines[:, 0], box_lines[:, 1])
        plt.show()

class TestCurveApprox:

    def plot_approx_2d(self):
        x_vec = np.linspace(1.1, 3.0, 100)
        y_vec = np.array([ np.sin(10*x) for x in x_vec ])
        points = np.stack( (x_vec, y_vec), axis=1)
        curve = bs_approx.curve_from_grid(points)

        bs_plot.plot_curve_2d(curve, 1000)
        bs_plot.plot_curve_poles_2d(curve)

        plt.show()

        #plt.plot(x_vec, y_vec, color='green')
        #plt.show()

    def test_approx_2d(self):
        # self.plot_approx_2d()
        pass