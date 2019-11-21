from bgem.polygons.polygons import PolygonDecomposition
import bgem.polygons.merge as merge
import numpy as np
import pytest
# def test_deep_copy(self):
#     print("===== test deep_copy")
#     da = PolygonDecomposition()
#     da.add_line((0, 0), (1,0))
#     sb, = da.add_line((0, 0), (0, 1))
#     da.add_line((1, 1), (1, 0))
#     da.add_line((1, 1), (0, 1))
#     da.add_line((0, 0), (1, 1))
#     da.add_line((1, 0), (0, 1))
#     da.delete_segment(sb)
#     print("da:\n", da)
#
#     db, maps = merge.deep_copy(da)
#     print("db:\n", db)
#     print(maps)
#
#     for poly_b in db.polygons.values():
#         poly_a = da.polygons[maps[2][poly_b.id]]
#         for sa, sb in zip(poly_a.outer_wire.segments(), poly_b.outer_wire.segments()):
#             seg_a, _ = sa
#             seg_b, _ = sb
#             assert seg_a.id == maps[1][seg_b.id]


def test_simple_intersections():
    da = PolygonDecomposition()
    da.add_line((0, 0), (1,0))
    da.add_line((0, 0), (0, 1))
    da.add_line((1, 1), (1, 0))
    da.add_line((1, 1), (0, 1))
    da.add_line((0, 0), (1, 1))
    #print("da:\n", da)

    db = PolygonDecomposition()
    db.add_line((0, 0), (1,0))
    db.add_line((0, 0), (0, 1))
    db.add_line((1, 1), (1, 0))
    db.add_line((1, 1), (0, 1))
    db.add_line((1, 0), (0, 1))
    #print("db:\n", db)

    (dc, maps_a, maps_b) = merge.intersect_single(da, db)
    #print("dc\n", dc)
    #plot_polygon_decomposition(dc)
    assert maps_a[0] == {}
    assert maps_b[0] == { 0: 0, 1: 1, 2: 2, 3: 3}
    assert maps_a[1] == { 5:4}
    assert maps_b[1] == { 0: 0, 1: 1, 2: 2, 3: 3, 6:4, 7:4}
    assert maps_a[2] == { 3: 1, 4: 2}
    assert maps_b[2] == { 0: 0, 1: 1, 2: 1, 3: 2, 4: 2}





#@pytest.mark.skip
def test_frac_intersections():
    # import sys
    # import trace
    #
    # # create a Trace object, telling it what to ignore, and whether to
    # # do tracing or line-counting or both.
    # tracer = trace.Trace(
    #     ignoredirs=[sys.prefix, sys.exec_prefix],
    #     trace=0,
    #     count=1)


    da = PolygonDecomposition()
    box = np.array([[0.0, 0.0],
                    [2.0, 3.0]])
    p00, p11 = box
    p01 = np.array([p00[0], p11[1]])
    p10 = np.array([p11[0], p00[1]])
    da.add_line(p00, p01)
    da.add_line(p01, p11)
    da.add_line(p11, p10)
    da.add_line(p10, p00)
    decomps = [da]

    np.random.seed(1)
    n_frac = 50
    p0 = np.random.rand(n_frac, 2) * (box[1] - box[0]) + box[0]
    p1 = np.random.rand(n_frac, 2) * (box[1] - box[0]) + box[0]

    for pa, pb in zip(p0, p1):
        dd = PolygonDecomposition()
        dd.add_line(pa, pb)
        decomps.append(dd)

    def tracer_func():
        return merge.intersect_decompositions(decomps)

    # import cProfile
    # cProfile.runctx('tracer_func()', globals(), locals(), 'prof_stats')
    #
    # import pstats
    # p = pstats.Stats('prof_stats')
    # p.sort_stats('cumulative').print_stats()

    decomp, maps = tracer_func()

    #######
    # Test merge with empty decomp.
    # Test fix for split points under tolerance.
    copy_decomp, maps = merge.intersect_decompositions([ decomp ])