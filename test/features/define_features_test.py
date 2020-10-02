"""
Unit Tests for Feature Creation
(c) 2020 d373c7
"""
import unittest
import d373c7.features as ft


class TestFeatureSource(unittest.TestCase):
    def test_creation_base(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_STRING
        f = ft.FeatureSource(name, f_type)
        self.assertIsInstance(f, ft.FeatureSource, f'Unexpected Type {type(f)}')
        self.assertEqual(f.name, name, f'Feature Name should be {name}')
        self.assertEqual(f.type, f_type, f'Feature Type should be {f_type}')
        self.assertIsNone(f.default, 'Should not have a default')
        self.assertIsNone(f.format_code, 'Should not have format code')
        self.assertEqual(len(f.embedded_features), 0, 'Should not have embedded features')

    def test_creation_w_format_code(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_STRING
        code = 'anything'
        f = ft.FeatureSource(name, f_type, code)
        self.assertIsInstance(f, ft.FeatureSource, f'Unexpected Type {type(f)}')
        self.assertEqual(f.name, name, f'Feature Name should be {name}')
        self.assertEqual(f.type, f_type, f'Feature Type should be {f_type}')
        self.assertIsNone(f.default, 'Should not have a default')
        self.assertEqual(f.format_code, code, f'Format code should have been {code}')
        self.assertEqual(len(f.embedded_features), 0, 'Should not have embedded features')

    def test_creation_w_default(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_STRING
        default = 'NA'
        f = ft.FeatureSource(name, f_type, default=default)
        self.assertIsInstance(f, ft.FeatureSource, f'Unexpected Type {type(f)}')
        self.assertEqual(f.name, name, f'Feature Name should be {name}')
        self.assertEqual(f.type, f_type, f'Feature Type should be {f_type}')
        self.assertEqual(f.default, default, f'Default should be {default}')
        self.assertIsNone(f.format_code, 'Should not have format code')
        self.assertEqual(len(f.embedded_features), 0, 'Should not have embedded features')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
