"""
Unit Tests for Profile Package
(c) 2022 d373c7
"""
import unittest
import numpy as np
import d373c7.engines as en
import d373c7.features as ft

FILES_DIR = './files/'


class TestNative(unittest.TestCase):
    def test_go(self):
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        td = ft.TensorDefinition('Source', [fa, fc])
        with en.EnginePandasNumpy(num_threads=threads) as e:




def main():
    unittest.main()


if __name__ == '__main__':
    main()

