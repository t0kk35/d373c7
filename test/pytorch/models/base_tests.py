"""
Unit Tests for PyTorch Model Base Functionality
(c) 2020 d373c7
"""
import unittest
from d373c7.pytorch.models.common import ModelDefaults, PyTorchModelException


class ModelDefaultTests(unittest.TestCase):
    def test_create(self):
        d = ModelDefaults()
        d.set('i', 0)
        d.set('b', False)
        d.set('f', 0.0)
        d.set('s', 'string')
        self.assertEqual(d.get_int('i'), 0, f'Get int should have returned 0. Got {d.get_int("i")}')
        self.assertEqual(d.get_bool('b'), False, f'Get bool should have returned False. Got {d.get_bool("b")}')
        self.assertEqual(d.get_float('f'), 0.0, f'Get float should have returned 0.0. Got {d.get_float("f")}')
        self.assertEqual(d.get_str('s'), 'string', f'Get str should have returned "string". Got {d.get_str("s")}')

    def test_fail_int(self):
        d = ModelDefaults()
        d.set('i', 0)
        d.set('ni', 0.0)
        with self.assertRaises(PyTorchModelException):
            _ = d.get_int('x')
            _ = d.get_int('ni')
            _ = d.get_float('i')
            _ = d.get_str('i')
            _ = d.get_bool('i')

    def test_fail_bool(self):
        d = ModelDefaults()
        d.set('b', False)
        d.set('nb', 0.0)
        with self.assertRaises(PyTorchModelException):
            _ = d.get_bool('x')
            _ = d.get_bool('nb')
            _ = d.get_int('b')
            _ = d.get_float('b')
            _ = d.get_str('b')

    def test_fail_float(self):
        d = ModelDefaults()
        d.set('f', 0.0)
        d.set('nf', 'str')
        with self.assertRaises(PyTorchModelException):
            _ = d.get_float('x')
            _ = d.get_float('nf')
            _ = d.get_int('f')
            _ = d.get_bool('f')
            _ = d.get_str('f')

    def test_fail_str(self):
        d = ModelDefaults()
        d.set('s', 'string')
        d.set('ns', 0.0)
        with self.assertRaises(PyTorchModelException):
            _ = d.get_str('x')
            _ = d.get_str('ns')
            _ = d.get_int('s')
            _ = d.get_bool('s')
            _ = d.get_float('s')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
