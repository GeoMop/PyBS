from bgem.bspline import bspline as bs, bspline_plot as bp
import numpy as np
import math


class TestPlottingMatplot:
    # plotting = bp.Plotting(bp.PlottingMatplot())

    def plotting_2d(self, plotting):
        # simple 2d graphs
        n_points = 201
        eq_basis = bs.SplineBasis.make_equidistant(2, 4)
        x_coord = np.linspace(eq_basis.domain[0], eq_basis.domain[1], n_points)

        for i_base in range(eq_basis.size):
            y_coord = [ eq_basis.eval(i_base, x) for x in x_coord ]
            plotting.plot_2d(x_coord, y_coord)
        plotting.show()

        # 2d curves
        degree = 2
        poles = [ [0., 0.], [1.0, 0.5], [2., -2.], [3., 1.] ]
        basis = bs.SplineBasis.make_equidistant(degree, 2)
        curve = bs.Curve(basis, poles)
        plotting.plot_curve_2d(curve, poles=True)

        poles = [[0., 0.], [-1.0, 0.5], [-2., -2.], [3., 1.]]
        basis = bs.SplineBasis.make_equidistant(degree, 2)
        curve = bs.Curve(basis, poles)
        plotting.plot_curve_2d(curve, poles=True)

        plotting.show()

    def test_plot_2d(self):
        self.plotting_2d(bp.Plotting(bp.PlottingMatplot()))
        self.plotting_2d(bp.Plotting((bp.PlottingPlotly())))

    def plotting_3d(self, plotting):
        # plotting 3d surfaces
        def function(x):
            return math.sin(x[0]*4) * math.cos(x[1]*4)

        poles = bs.make_function_grid(function, 4, 5)
        u_basis = bs.SplineBasis.make_equidistant(2, 2)
        v_basis = bs.SplineBasis.make_equidistant(2, 3)
        surface_func = bs.Surface( (u_basis, v_basis), poles[:,:, [2] ])

        #quad = np.array( [ [0, 0], [0, 0.5], [1, 0.1],  [1.1, 1.1] ]  )
        quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        z_surf = bs.Z_Surface(quad, surface_func)
        full_surf = z_surf.make_full_surface()
        z_surf.transform(np.array([[1., 0, 0], [0, 1, 0]]), np.array([2.0, 0]) )
        plotting.plot_surface_3d(z_surf)
        plotting.plot_surface_3d(full_surf)

        plotting.show()

    def test_plot_3d(self):
        self.plotting_3d(bp.Plotting(bp.PlottingMatplot()))
        self.plotting_3d(bp.Plotting(bp.PlottingPlotly()))