"""
Unit Tests for TensorDefinition Creation
(c) 2020 d373c7
"""
import unittest
import d373c7.features as ft


class TestTensorCreate(unittest.TestCase):
    def test_creation(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureSource('test-feature-2', ft.FEATURE_TYPE_STRING)
        t = ft.TensorDefinition(name_t, [f1, f2])
        self.assertIsInstance(t, ft.TensorDefinition, f'TensorDefinition creation failed')
        self.assertEqual(t.name, name_t, f'Tensor Definition name not correct. Got {name_t}')
        self.assertListEqual([f1, f2], t.features, f'Tensor def feature list incorrect {t.features}')

    def test_remove(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureSource('test-feature-2', ft.FEATURE_TYPE_STRING)
        t = ft.TensorDefinition(name_t, [f1, f2])
        t.remove(f2)
        self.assertNotIn(f2, t.features, f'Tensor Definition Feature Removal failed')

    def test_len(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureSource('test-feature-2', ft.FEATURE_TYPE_STRING)
        t = ft.TensorDefinition(name_t, [f1, f2])
        self.assertEqual(len(t), len([f1, f2]), f'Tensor definition lenght not working. Got {len(t)}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
