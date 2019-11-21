import unittest
from bgem.bspline import brep_writer as bw


class TestConstructors(unittest.TestCase):
    def test_Location(self):
        print( "test locations")
        la = bw.Location([[1, 0, 0, 4], [0, 1, 0, 8], [0, 0, 1, 12]])
        lb = bw.Location([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        lc = bw.ComposedLocation([ (la, 2), (lb, 1) ])
        ld = bw.ComposedLocation([ (lb, 2), (lc, 3) ])
        with self.assertRaises(bw.ParamError):
            bw.Location([1,2,3])
        with self.assertRaises(bw.ParamError):
            bw.Location([[1], [2], [3]])
        with self.assertRaises(bw.ParamError):
            a = 1
            b = 'a'
            lb = bw.Location([[a, b, a, b], [a, b, a, b], [a, b, a, b]])

    def test_Shape(self):

        # check sub types
        with self.assertRaises(bw.ParamError):
            bw.Wire(['a', 'b'])
        v=bw.Vertex([1,2,3])
        with self.assertRaises(bw.ParamError):
            bw.Wire([v, v])


    def test_Vertex(self):
        with self.assertRaises(bw.ParamError):
            bw.Vertex(['a','b','c'])


class TestPlanarGeomeries(unittest.TestCase):

    def _cube(self):
        # 0, 0; top, bottom
        v1=bw.Vertex((0.0, 0.0, 1.0))
        v2=bw.Vertex((0.0, 0.0, 0.0))

        v3=bw.Vertex((0.0, 1.0, 1.0))
        v4=bw.Vertex((0.0, 1.0, 0.0))

        v5=bw.Vertex((1.0, 0.0, 1.0))
        v6=bw.Vertex((1.0, 0.0, 0.0))

        # vertical edges
        e1=bw.Edge([v1,v2])
        e2=bw.Edge([v3,v4])
        e3=bw.Edge([v5,v6])

        # face v12 - v34
        # top
        e4=bw.Edge([v1,v3])
        # bottom
        e5=bw.Edge([v2,v4])
        f1 = bw.Face([e1.m(), e4, e2, e5.m()])

        # face v34 - v56
        # top
        e6=bw.Edge([v3, v5])
        # bottom
        e7=bw.Edge([v4, v6])
        f2 = bw.Face([e2.m(), e6, e3, e7.m()])

        # face v56 - v12
        # top
        e8=bw.Edge([v5, v1])
        # bottom
        e9=bw.Edge([v6, v2])
        f3 = bw.Face([e3.m(), e8, e1, e9.m()])

        # top cup
        f4 = bw.Face([e4, e6, e8])
        # bot cup
        w5=bw.Wire([e5, e7, e9])
        f5 = bw.Face([w5.m()])

        shell = bw.Shell([ f1, f2, f3, f4, f5.m() ])
        return shell

    def _permuted_cube(self):
        # 0, 0; top, bottom
        v1=bw.Vertex((0.0, 0.0, 1.0))
        v2=bw.Vertex((0.0, 0.0, 0.0))

        v3=bw.Vertex((0.0, 1.0, 1.0))
        v4=bw.Vertex((0.0, 1.0, 0.0))

        v5=bw.Vertex((1.0, 0.0, 1.0))
        v6=bw.Vertex((1.0, 0.0, 0.0))

        # face v12 - v34
        # top
        e4=bw.Edge([v1,v3])
        # top
        e6=bw.Edge([v3, v5])
        # top
        e8=bw.Edge([v5, v1])

        # top cup
        f4 = bw.Face([e4, e6, e8])

        # bottom
        e5=bw.Edge([v2,v4])
        # face v34 - v56
        # bottom
        e7=bw.Edge([v4, v6])
        # face v56 - v12
        # bottom
        e9=bw.Edge([v6, v2])

        # bot cup
        w5=bw.Wire([e5, e7, e9])
        f5 = bw.Face([w5.m()])


        # vertical edges
        e1=bw.Edge([v1,v2])
        e2=bw.Edge([v3,v4])
        e3=bw.Edge([v5,v6])

        f1 = bw.Face([e1.m(), e4, e2, e5.m()])

        f2 = bw.Face([e2.m(), e6, e3, e7.m()])

        f3 = bw.Face([e3.m(), e8, e1, e9.m()])

        shell = bw.Shell([ f4, f5.m(), f1, f2, f3 ])
        return shell

    def test_cube(self):

        shell = self._cube()
        #shell = self._permuted_cube()

        s1=bw.Solid([ shell ])

        c1=bw.Compound([s1])

        loc1=bw.Location([[0,0,1,0],[1,0,0,0],[0,1,0,0]])
        loc2=bw.Location([[0,0,1,0],[1,0,0,0],[0,1,0,0]]) #dej tam tu druhou z prikladu
        cloc=bw.ComposedLocation([(loc1,1),(loc2,1)])

        with open("_out_test_prism.brep", "w") as f:
            bw.write_model(f, c1, cloc)
            #bw.write_model(sys.stdout, c1, cloc)
        print(c1)


if __name__ == '__main__':
    unittest.main()