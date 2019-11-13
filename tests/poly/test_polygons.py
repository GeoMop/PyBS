# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import collections  as mc
# from matplotlib import patches as mp
import pytest

from polygons.polygons import *
from polygons.decomp import PolygonChange
from polygons.plot_polygons import plot_polygon_decomposition
#
# def plot_polygon(self, polygon):
#     if polygon is None or polygon.displayed or polygon.outer_wire.is_root():
#         return []
#
#     # recursion
#     assert polygon.outer_wire.parent.polygon != polygon
#     patches = self.plot_polygon( polygon.outer_wire.parent.polygon )
#     pts = [ pt.xy for pt in polygon.vertices() ]
#
#     patches.append(mp.Polygon(pts))
#     return patches
#
# def plot_polygons(self, decomp):
#     fig, ax = plt.subplots()
#
#     # polygons
#     for poly in decomp.polygons.values():
#         poly.displayed = False
#
#     patches = []
#     for poly in decomp.polygons.values():
#         patches.extend( self.plot_polygon(poly) )
#     p = mc.PatchCollection(patches, color='blue', alpha=0.2)
#
#     ax.add_collection(p)
#
#
#     for s in decomp.segments.values():
#         ax.plot((s.vtxs[0].xy[0], s.vtxs[1].xy[0]), (s.vtxs[0].xy[1], s.vtxs[1].xy[1]), color='green')
#
#     x_pts = []
#     y_pts = []
#     for pt in decomp.points.values():
#         x_pts.append(pt.xy[0])
#         y_pts.append(pt.xy[1])
#     ax.plot(x_pts, y_pts, 'bo', color = 'red')
#
#     plt.show()


def test_snap_point():
    pd = PolygonDecomposition()
    decomp = pd.decomp
    
    # vlines
    pd.add_line([0, 0], [0, 4])
    pd.add_line([1, 1], [1, 3])
    pd.add_line([2, 1], [2, 2])
    pd.add_line([3, 0], [3, 2])
    pd.add_line([4, 1], [4, 4])
    sg5, = pd.add_line([5, 1], [5, 4])
    sg6, = pd.add_line([6, 0], [6, 4])
    # h lines
    pd.add_line([0, 0], [6, 0])
    pd.add_line([4, 1], [5, 1])
    pd.add_line([4, 4], [5, 4])
    pd.add_line([0, 4], [6, 4])
    # diagonal
    pd.add_line([0, 2], [2, 4])
    #plot_polygon_decomposition(decomp)
    #decomp.check_consistency()
    pd.add_line([0, 3], [1, 4])
    pd.add_line([0.5, 0], [0, 0.5])
    decomp.check_consistency()



    def check_snap(dim, obj, snap):
        dim_, obj_, param_ = snap
        assert dim == dim_
        assert obj == obj_

    check_snap(2, decomp.outer_polygon, pd._snap_point([7, 2]))
    check_snap(1, sg6, pd._snap_point([6.009, 2]))
    check_snap(1, sg6, pd._snap_point([5.99, 2]))
    check_snap(0, sg6.vtxs[out_vtx], pd._snap_point([6.009, 0]))
    check_snap(2, decomp.polygons[2], pd._snap_point([5.5, 2]))
    check_snap(1, sg5, pd._snap_point([5, 2]))
    check_snap(2, decomp.polygons[1], pd._snap_point([4.5, 2]))
    check_snap(2, decomp.polygons[2], pd._snap_point([3.5, 2]))
    check_snap(2, decomp.polygons[2], pd._snap_point([3.5, 2]))
    check_snap(2, decomp.polygons[2], pd._snap_point([2.5, 1.5]))
    check_snap(2, decomp.polygons[2], pd._snap_point([2.5, 2]))
    check_snap(2, decomp.polygons[2], pd._snap_point([1.5, 1.5]))
    check_snap(2, decomp.polygons[2], pd._snap_point([1.5, 1]))
    check_snap(2, decomp.polygons[2], pd._snap_point([1.5, 2]))
    check_snap(2, decomp.polygons[2], pd._snap_point([1.5, 3]))
    check_snap(2, decomp.polygons[3], pd._snap_point([0.5, 3]))
    check_snap(2, decomp.polygons[2], pd._snap_point([1.5, 0.5]))


def test_check_displacement():
    decomp = PolygonDecomposition()
    sg, = decomp.add_line((0, 0), (0, 2))
    pt0 = sg.vtxs[0]
    decomp.add_line((0, 0), (2, 0))
    decomp.add_line((0, 2), (2, 0))
    pt = decomp.add_point((.5,.5))
    decomp.add_line((0, 0), (.5,.5))
    decomp.add_line((2, 0), (.5,.5))
    decomp.add_line( (.5,.5), (0, 2))
    #print(decomp)
    #plot_polygon_decomposition(decomp)

    res = decomp.check_displacment([pt0, pt], (1.0, 1.0))
    assert not res  #la.norm( step - np.array([0.45, 0.45]) ) < 1e-6
    assert decomp.get_last_polygon_changes() == (PolygonChange.shape, [0,1,2,3], None)

    res = decomp.check_displacment([pt0, pt], (0.4, 0.4))
    assert res

    decomp.move_points([pt], (0.4, 0.4))
    assert la.norm( pt.xy - np.array([0.9, 0.9]) ) < 1e-6

    res = decomp.check_displacment([pt], (1.0, 1.0))
    assert not res
    assert decomp.get_last_polygon_changes() == (PolygonChange.shape, [1,2,3], None)

