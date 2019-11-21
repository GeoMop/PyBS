"""
Functions to plot Bspline curves and surfaces.
"""

plot_lib = "plotly"

import plotly.offline as pl
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np




class PlottingPlotly:
    def __init__(self):
        self.i_figure = -1
        self._reinit()

    def _reinit(self):
        self.i_figure += 1
        self.data_3d = []
        self.data_2d = []

    def add_curve_2d(self, X, Y, **kwargs):
        self.data_2d.append(  go.Scatter(x=X, y=Y, mode = 'lines') )

    def add_points_2d(self, X, Y, **kwargs):
        marker = dict(
            size=10,
            color='red',
        )
        self.data_2d.append(  go.Scatter(x=X, y=Y,
                         mode = 'markers',
                         marker=marker) )


    def add_surface_3d(self, X, Y, Z, **kwargs):
        hue = (120.0*(len(self.data_3d)))%360
        colorscale = [[0.0, 'hsv({}, 50%, 10%)'.format(hue)], [1.0, 'hsv({}, 50%, 90%)'.format(hue)]]
        self.data_3d.append( go.Surface(x=X, y=Y, z=Z, colorscale=colorscale))


    def add_points_3d(self, X, Y, Z, **kwargs):
        marker = dict(
            size=5,
            color='red',
            # line=dict(
            #     color='rgba(217, 217, 217, 0.14)',
            #     width=0.5
            # ),
            opacity=0.6
        )
        self.data_3d.append( go.Scatter3d(
            x=X, y=Y, z=Z,
            mode='markers',
            marker=marker
        ))


    def show(self):
        """
        Show added plots and clear the list for other plotting.
        :return:
        """
        if self.data_3d:
            fig_3d = go.Figure(data=self.data_3d)
            pl.plot(fig_3d, filename='bc_plot_3d_%d.html'%(self.i_figure))
        if self.data_2d:
            fig_2d = go.Figure(data=self.data_2d)
            pl.plot(fig_2d, filename='bc_plot_2d_%d.html'%(self.i_figure))
        self._reinit()


class PlottingMatplot:
    def __init__(self):
        self.fig_2d = plt.figure(1)
        self.fig_3d = plt.figure(2)
        self.ax_3d = self.fig_3d.gca(projection='3d')

    def add_curve_2d(self, X, Y, **kwargs):
        plt.figure(1)
        plt.plot(X, Y, **kwargs)

    def add_points_2d(self, X, Y, **kwargs):
        plt.figure(1)
        plt.plot(X, Y, 'bo', color='red', **kwargs)

    def add_surface_3d(self, X, Y, Z, **kwargs):
        plt.figure(2)
        self.ax_3d.plot_surface(X, Y, Z, **kwargs)

    def add_points_3d(self, X, Y, Z, **kwargs):
        plt.figure(2)
        return self.ax_3d.scatter(X, Y, Z, color='red', **kwargs)

    def show(self):
        """
        Show added plots and clear the list for other plotting.
        :return:
        """
        plt.show()


class Plotting:
    """
    Debug plotting class. Several 2d and 3d plots can be added and finally displayed on common figure
    calling self.show(). Matplotlib or plotly library is used as backend.
    """
    def __init__(self, backend = PlottingPlotly()):
        self.backend = backend

    def plot_2d(self, X, Y):
        """
        Add line scatter plot. Every plot use automatically different color.
        :param X: x-coords of points
        :param Y: y-coords of points
        """
        self.backend.add_curve_2d(X,Y)

    def scatter_2d(self, X, Y):
        """
        Add point scatter plot. Every plot use automatically different color.
        :param X: x-coords of points
        :param Y: y-coords of points
        """
        self.backend.add_points_2d(X,Y)

    def plot_surface(self, X, Y, Z):
        """
        Add line scatter plot. Every plot use automatically different color.
        :param X: x-coords of points
        :param Y: y-coords of points
        """
        self.backend.add_surface_3d(X, Y, Z)

    def plot_curve_2d(self, curve, n_points=100, poles=False):
        """
        Add plot of a 2d Bspline curve.
        :param curve: Curve t -> x,y
        :param n_points: Number of evaluated points.
        :param: kwargs: Additional parameters passed to the mtplotlib plot command.
        """

        basis = curve.basis
        t_coord = np.linspace(basis.domain[0], basis.domain[1], n_points)

        coords = [curve.eval(t) for t in t_coord]
        x_coord, y_coord = zip(*coords)

        self.backend.add_curve_2d(x_coord, y_coord)
        if poles:
            self.plot_curve_poles_2d(curve)


    def plot_curve_poles_2d(self, curve):
        """
        Plot poles of the B-spline curve.
        :param curve: Curve t -> x,y
        :return: Plot object.
        """
        x_poles, y_poles = curve.poles.T[0:2, :]    # remove weights
        return self.backend.add_points_2d(x_poles, y_poles)

    def scatter_3d(self, X, Y, Z):
        """
        Add point scatter plot. Every plot use automatically different color.
        :param X: x-coords of points
        :param Y: y-coords of points
        """
        self.backend.add_points_3d(X, Y, Z)


    def plot_surface_3d(self, surface, n_points=(100, 100), poles=False):
        """
        Plot a surface in 3d.
        Usage:
            plotting=Plotting()
            plotting.plot_surface_3d(surf_1)
            plotting.plot_surface_3d(surf_2)
            plotting.show()

        :param surface: Parametric surface in 3d.
        :param n_points: (nu, nv), nu*nv - number of evaluation point
        """

        u_basis, v_basis = surface.u_basis, surface.v_basis

        u_coord = np.linspace(u_basis.domain[0], u_basis.domain[1], n_points[0])
        v_coord = np.linspace(v_basis.domain[0], v_basis.domain[1], n_points[1])

        U, V = np.meshgrid(u_coord, v_coord)
        points = np.stack( [U.ravel(), V.ravel()], axis = 1 )

        xyz = surface.eval_array(points)
        X, Y, Z = xyz.T

        X = X.reshape(U.shape)
        Y = Y.reshape(U.shape)
        Z = Z.reshape(U.shape)

        # Plot the surface.
        self.backend.add_surface_3d(X, Y,  Z)

        if poles:
            self.plot_surface_poles_3d(surface)




    def plot_grid_surface_3d(self, surface,  n_points=(100, 100)):
        """
        Plot a surface in 3d, on UV plane.

        :param surface: Parametric surface in 3d.
        :param n_points: (nu, nv), nu*nv - number of evaluation point
        """

        u_coord = np.linspace(0, 1.0, n_points[0])
        v_coord = np.linspace(0, 1.0, n_points[1])

        U, V = np.meshgrid(u_coord, v_coord)
        points = np.stack( [U.ravel(), V.ravel()], axis = 1 )

        xyz = surface.eval_array(points)
        X, Y, Z = xyz.T

        Z = Z.reshape(U.shape)

        # Plot the surface.
        self.backend.add_surface_3d(U, V,  Z)


    def plot_surface_poles_3d(self, surface, **kwargs):
        """
        Plot poles of the B-spline curve.
        :param curve: Curve t -> x,y
        :param: kwargs: Additional parameters passed to the mtplotlib plot command.
        :return: Plot object.
        """
        x_poles, y_poles, z_poles = surface.poles[:, :, 0:3].reshape(-1, 3).T          # remove weights and flatten nu, nv
        return self.backend.add_points_3d(x_poles, y_poles, z_poles, **kwargs)


    def show(self):
        """
        Display added plots. Empty the queue.
        :return:
        """
        self.backend.show()
