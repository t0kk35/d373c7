"""
Definition of a set of Numpy Helper classes.
(c) 2020 d373c7
"""
import unittest
import numpy as np
import d373c7.engines as en


class TestCreation(unittest.TestCase):
    def test_creation_base(self):
        x = np.arange(10)
        y = np.arange(10)
        c = [x, y]
        n = en.NumpyList(c)
        self.assertIsInstance(n, en.NumpyList)
        self.assertEqual(len(n), len(x), f'Length not correct {len(n)}/{len(x)}')
        self.assertEqual(len(n.shapes[0]), 1, f'Shape should only have 1 dim {len(n.shapes[0])}')
        self.assertEqual(n.shapes[0][0], len(x), f'Shape of dim 0 incorrect {n.shapes[0][0]}')
        self.assertEqual(len(n.shapes[1]), 1, f'Shape should only have 1 dim {len(n.shapes[1])}')
        self.assertEqual(n.shapes[1][0], len(y), f'Shape of dim 0 incorrect {n.shapes[1][0]}')
        self.assertEqual(n.number_of_lists, len(c), f'Number of lists incorrect {n.number_of_lists}')
        self.assertEqual(n.dtype_names[0], x.dtype.name, f'dtype not expected {n.dtype_names[0]}')
        self.assertEqual(n.dtype_names[1], y.dtype.name, f'dtype not expected {n.dtype_names[1]}')
        self.assertListEqual(n.lists, c, f'Not the expected return from numpy_list {n.lists}')

    def test_creation_wrong_size(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(2, 2)
        with self.assertRaises(en.NumpyListException):
            en.NumpyList([x, y])

    def test_lists(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = [x, y]
        n = en.NumpyList(c)
        self.assertEqual(len(n.lists), len(c), f'Number of lists does not add up {len(n.lists)}')
        self.assertEqual((n.lists[0] == x).all(), True, f'Lists not equal')
        self.assertEqual((n.lists[1] == y).all(), True, f'Lists not equal')

    def test_slice_good(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = [x, y]
        n = en.NumpyList(c)
        x0, y0 = n[0].lists
        self.assertEqual(np.array(x0 == x[0]).all(), True, f'First entries do not match {x0}, {x[0]}')
        self.assertEqual(np.array(y0 == y[0]).all(), True, f'First entries do not match {y0}, {y[0]}')
        x1, y1 = n[1].lists
        self.assertEqual(np.array(x1 == x[1]).all(), True, f'Second entries do not match {x1}, {x[1]}')
        self.assertEqual(np.array(y1 == y[1]).all(), True, f'Second entries do not match {y1}, {y[1]}')
        xf, yf = n[0:5].lists
        self.assertEqual(np.array(xf == x).all(), True, f'All entries do not match {xf}, {x}')
        self.assertEqual(np.array(yf == y).all(), True, f'All entries do not match {yf}, {y}')
        xm, ym = n[1:4].lists
        self.assertEqual(np.array(xm == x[1:4]).all(), True, f'Mid entries do not match {xf}, {x[1:4]}')
        self.assertEqual(np.array(ym == y[1:4]).all(), True, f'Mid entries do not match {yf}, {y[1:4]}')
        xl, yl = n[4].lists
        self.assertEqual(np.array(xl == x[-1]).all(), True, f'Last entries do not match {xl}, {x[-1]}')
        self.assertEqual(np.array(yl == y[-1]).all(), True, f'Last entries do not match {yl}, {y[-1]}')
        xl, yl = n[-1].lists
        self.assertEqual(np.array(xl == x[-1]).all(), True, f'Last entries do not match {xl}, {x[-1]}')
        self.assertEqual(np.array(yl == y[-1]).all(), True, f'Last entries do not match {yl}, {y[-1]}')

    def test_slice_bad(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = [x, y]
        n = en.NumpyList(c)
        with self.assertRaises(en.NumpyListException):
            _ = n[5]
        with self.assertRaises(en.NumpyListException):
            _ = n[-6]
        with self.assertRaises(en.NumpyListException):
            _ = n[0:6]
        with self.assertRaises(en.NumpyListException):
            _ = n[-1:5]

    def test_pop(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = [x, y]
        n = en.NumpyList(c)
        a = n.pop(1)
        self.assertEqual(n.number_of_lists, len([x, y])-1, f'Length not reduced after pop {n.number_of_lists}')
        self.assertEqual((y == a).all(), True, f'Popped list not equal to original')
        with self.assertRaises(en.NumpyListException):
            n.pop(1)

    def test_remove(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c = [x, y]
        n = en.NumpyList(c)
        n.remove(1)
        self.assertEqual(n.number_of_lists, len([x, y])-1, f'Length not reduced after pop {n.number_of_lists}')
        with self.assertRaises(en.NumpyListException):
            n.remove(1)

    def test_unique(self):
        x = np.random.rand(5, 2)
        y = np.random.randint(32, size=(5, 2))
        c = [x, y]
        n = en.NumpyList(c)
        vl_1, cn_1 = np.unique(y, return_counts=True)
        vl_1, cn_1 = list(vl_1), list(cn_1)
        vl_2, cn_2 = n.unique(1)
        self.assertListEqual(vl_1, vl_2, f'Unique values not correct. Got {vl_2}. Expected {vl_1}')
        self.assertListEqual(cn_1, cn_2, f'Unique counts not correct. Got {cn_2}. Expected {cn_1}')
        with self.assertRaises(en.NumpyListException):
            _, _ = n.unique(0)

    def test_shuffle(self):
        x = np.arange(5)
        y = np.arange(5)
        c = [x, y]
        n = en.NumpyList(c)
        xs, ys = n.shuffle().lists
        self.assertEqual(np.array(xs == ys).all(), True, f'Rows not equal after 1st shuffle')
        n = en.NumpyList(c)
        xs, ys = n.shuffle().lists
        self.assertEqual(np.array(xs == ys).all(), True, f'Rows not equal after 2nd shuffle')

    def test_sample(self):
        x = np.arange(5)
        y = np.arange(5)
        c = [x, y]
        n = en.NumpyList(c)
        xs, ys = n.sample(3).lists
        self.assertEqual(np.array(xs == ys).all(), True, f'Rows not equal after 1st sample')
        xs, ys = n.sample(2).lists
        self.assertEqual(np.array(xs == ys).all(), True, f'Rows not equal after 2nd sample')
        with self.assertRaises(en.NumpyListException):
            _ = n.sample(len(x)+1)

    def test_concatenate_good(self):
        a = np.random.rand(5, 2)
        b = np.random.rand(5, 2)
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        c1 = [x, y]
        n1 = en.NumpyList(c1)
        c2 = [a, b]
        n2 = en.NumpyList(c2)
        l1, l2 = n1.concat(n2).lists
        self.assertEqual((l1 == np.concatenate([x, a])).all(), True, f'Concatenate of first list failed')
        self.assertEqual((l2 == np.concatenate([y, b])).all(), True, f'Concatenate of first list failed')

    def test_concatenate_bad(self):
        a = np.random.rand(5, 2)
        b = np.random.rand(5, 2)
        x = np.random.rand(5, 3)
        y = np.random.rand(5, 3)
        c1 = [x, y]
        n1 = en.NumpyList(c1)
        c2 = [a, b]
        n2 = en.NumpyList(c2)
        with self.assertRaises(en.NumpyListException):
            _ = n1.concat(n2).lists

        a = np.random.rand(5, 2)
        b = np.random.rand(5, 2)
        x = np.random.rand(5, 2, 1)
        y = np.random.rand(5, 2, 1)
        c1 = [x, y]
        n1 = en.NumpyList(c1)
        c2 = [a, b]
        n2 = en.NumpyList(c2)
        with self.assertRaises(en.NumpyListException):
            _ = n1.concat(n2).lists


def main():
    unittest.main()


if __name__ == '__main__':
    main()
