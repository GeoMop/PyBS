import pytest
import numpy as np
import numpy.linalg as la

import sys
import os
print("cwg : ", os.getcwd())
print("test path: ", sys.path)


from bgem.polygons.aabb_lookup import *


def test_make_aabb():
    points = [ [0,0], [4,1], [0,-4],[2, 5]]
    box = make_aabb(points)
    assert np.all(box == np.array([0,-4,4,5]))
    box = make_aabb(points, margin=0.1)
    assert np.all(box == np.array([-0.1, -4.1, 4.1, 5.1]))

def test_intersect_candidates():
    al = AABB_Lookup()
    box = make_aabb([[-1,-1],[1,1]], margin = 0.1)

    def add_box(*pts):
        al.add_object(add_box.ibox, make_aabb(pts) )
        add_box.ibox += 1

    add_box.ibox = 0

    add_box([1, 1], [2, 3])
    add_box([0, 0], [1, 1])
    add_box([1, -1], [2, -1])
    add_box([-1,-1], [1,1])

    add_box([1, 5], [2, 10])
    add_box([1, -5], [2, -10])
    candidates = al.intersect_candidates(box)
    assert candidates.tolist() == [0,1,2,3]


def min_distance(point, box_list):
    min_dist = (np.inf, None)
    for i, box in enumerate(box_list):
        dist = min( la.norm(box[0:2] - point), la.norm(box[2:4] - point) )
        if dist < min_dist[0]:
            min_dist = (dist, i)
    return min_dist

@pytest.mark.parametrize("seed", list(range(40)))
def test_closest_candidates(seed):
    al = AABB_Lookup(init_size=10)

    def add_box(*pts):
        al.add_object(add_box.ibox, make_aabb(pts) )
        add_box.ibox += 1

    add_box.ibox = 0
    np.random.seed(seed)
    size = 1000
    boxes1 = 3*np.random.rand(size, 2)
    boxes2 = boxes1 + 1.5*(np.random.rand(size, 2) -0.5)
    boxes = np.concatenate((boxes1, boxes2), axis=1)
    for i, box in enumerate(boxes):
        add_box(*box.reshape(2,2))

    point = np.array([1., 2.])
    ref_min_dist = min_distance(point, boxes)

    candidates = al.closest_candidates(point)

    print("{} < {}".format(len(candidates), boxes.shape[0]))
    assert len(candidates) < boxes.shape[0]
    c_boxes = boxes[candidates,:]
    min_dist = min_distance(point, c_boxes)
    min_dist = (min_dist[0], candidates[min_dist[1]])
    assert ref_min_dist == min_dist

