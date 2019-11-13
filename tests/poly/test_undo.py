from polygons.polygons import *


enable_undo()


class DState:
    def __init__(self, decomp):
        self.points = {k: (v.xy[0], v.xy[1]) for k, v in decomp.points.items()}
        self.segments = {k: (v.vtxs[0].id, v.vtxs[1].id) for k, v in decomp.segments.items()}
        self.polygons_keys = sorted(decomp.polygons.keys())

    def __eq__(self, other):
        if isinstance(other, DState):
            return self.points == other.points and \
                   self.segments == other.segments and \
                   self.polygons_keys == other.polygons_keys
        else:
            return NotImplemented


def test_add_point():
    decomp = PolygonDecomposition()
    s0 = DState(decomp)
    decomp._add_point([1, 1], decomp.outer_polygon)
    s1 = DState(decomp)

    undo.stack().undo()
    assert DState(decomp) == s0

    undo.stack().redo()
    assert DState(decomp) == s1


def test_add_segmnet():
    decomp = PolygonDecomposition()
    s0 = DState(decomp)
    pt1 = decomp._add_point([1, 1], decomp.outer_polygon)
    s1 = DState(decomp)
    pt2 = decomp._add_point([2, 2], decomp.outer_polygon)
    s2 = DState(decomp)
    decomp._add_segment(pt1, pt2)
    s3 = DState(decomp)

    undo.stack().undo()
    assert DState(decomp) == s2
    undo.stack().undo()
    assert DState(decomp) == s1
    undo.stack().undo()
    assert DState(decomp) == s0

    undo.stack().redo()
    assert DState(decomp) == s1
    undo.stack().redo()
    assert DState(decomp) == s2
    undo.stack().redo()
    assert DState(decomp) == s3


def test_rm_point():
    decomp = PolygonDecomposition()
    pt1 = decomp._add_point([1, 1], decomp.outer_polygon)
    pt2 = decomp._add_point([2, 2], decomp.outer_polygon)
    s0 = DState(decomp)
    decomp._rm_point(pt1)
    s1 = DState(decomp)

    undo.stack().undo()
    assert DState(decomp) == s0

    undo.stack().redo()
    assert DState(decomp) == s1


def test_rm_segment():
    decomp = PolygonDecomposition()
    pt1 = decomp._add_point([1, 1], decomp.outer_polygon)
    pt2 = decomp._add_point([2, 2], decomp.outer_polygon)
    seg = decomp._add_segment(pt1, pt2)
    s0 = DState(decomp)
    decomp._rm_segment(seg)
    s1 = DState(decomp)

    undo.stack().undo()
    assert DState(decomp) == s0

    undo.stack().redo()
    assert DState(decomp) == s1

def test_split_segment():
    decomp = PolygonDecomposition()
    pt1 = decomp._add_point([1, 1], decomp.outer_polygon)
    mid_pt = decomp._add_point([2, 2], decomp.outer_polygon)
    pt3 = decomp._add_point([3, 3], decomp.outer_polygon)
    seg = decomp._add_segment(pt1, pt3)
    s0 = DState(decomp)
    decomp._split_segment(seg, mid_pt)
    s1 = DState(decomp)

    undo.stack().undo()
    assert DState(decomp) == s0

    undo.stack().redo()
    assert DState(decomp) == s1


def test_join_segment():
    decomp = PolygonDecomposition()
    pt1 = decomp._add_point([1, 1], decomp.outer_polygon)
    mid_pt = decomp._add_point([2, 2], decomp.outer_polygon)
    pt3 = decomp._add_point([3, 3], decomp.outer_polygon)
    seg1 = decomp._add_segment(pt1, mid_pt)
    seg2 = decomp._add_segment(mid_pt, pt3)
    s0 = DState(decomp)
    decomp._join_segments(mid_pt, seg1, seg2)
    s1 = DState(decomp)

    undo.stack().undo()
    assert DState(decomp) == s0

    undo.stack().redo()
    assert DState(decomp) == s1


def test_set_attr():
    decomp = PolygonDecomposition()
    pt = decomp._add_point([1, 1], decomp.outer_polygon)
    old_attr = pt.attr
    new_attr = "attr"
    decomp.set_attr(pt, new_attr)

    undo.stack().undo()
    assert pt.attr == old_attr
    undo.stack().undo()

    undo.stack().redo()
    undo.stack().redo()
    pt = decomp.points[pt.id]
    assert pt.attr == new_attr


def test_split_triangle():
    decomp = PolygonDecomposition()
    pt1 = decomp._add_point([0, 0], decomp.outer_polygon)
    pt2 = decomp._add_point([2, 0], decomp.outer_polygon)
    pt3 = decomp._add_point([1, 1], decomp.outer_polygon)
    mid_pt = decomp._add_point([1, 0], decomp.outer_polygon)
    seg = decomp._add_segment(pt1, pt2)
    decomp._add_segment(pt2, pt3)
    decomp._add_segment(pt3, pt1)
    s0 = DState(decomp)
    decomp._split_segment(seg, mid_pt)
    s1 = DState(decomp)

    undo.stack().undo()
    assert DState(decomp) == s0

    undo.stack().redo()
    assert DState(decomp) == s1
