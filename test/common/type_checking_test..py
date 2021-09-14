"""
Testing of the type_checking module
(c) 2021 d373c7
"""
import typing
import unittest
import d373c7.common as cm
from dataclasses import dataclass


class TestDataClass(unittest.TestCase):
    def test_creation_simple_good(self):
        @cm.enforce_types
        @dataclass
        class Dummy:
            an_int: int
            a_float: float
            a_string: str
            an_optional_int: int = 1
        _ = Dummy(0, 1.0, 'x')
        _ = Dummy(0, 1.0, 'x', 0)

    def test_creation_simple_bad(self):
        @cm.enforce_types
        @dataclass
        class Dummy:
            an_int: int
            a_float: float
            a_string: str
            an_optional_int: int = 1
        with self.assertRaises(TypeError):
            # String instead of optional int
            _ = Dummy(0, 1.0, 'x', 'x')
            # int instead of float
            _ = Dummy(0, 1, 'x', 0)
            # too few arguments
            _ = Dummy(0, 1)

    def test_creation_list_good(self):
        @cm.enforce_types
        @dataclass
        class Dummy:
            an_int: typing.List[int]
        _ = Dummy([1, 0])

    def test_creation_list_bad(self):
        @cm.enforce_strict_types
        @dataclass
        class Dummy:
            an_int: typing.List[int]
        with self.assertRaises(TypeError):
            _ = Dummy([1.0, 1.0])


def main():
    unittest.main()


if __name__ == '__main__':
    main()
