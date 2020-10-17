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
        self.assertEqual(t.inference_ready, False, f'Tensor should not be ready for inference at this point')
        with self.assertRaises(ft.TensorDefinitionException):
            _ = t.rank

    def test_duplicate_bad(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.TensorDefinitionException):
            _ = ft.TensorDefinition(name_t, [f1, f1])

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
        self.assertEqual(len(t), len([f1, f2]), f'Tensor definition length not working. Got {len(t)}')

    def test_set_label(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureSource('test-feature-2', ft.FEATURE_TYPE_STRING)
        f3 = ft.FeatureSource('test-feature-3', ft.FEATURE_TYPE_STRING)
        t = ft.TensorDefinition(name_t, [f1, f2])
        t.set_label(f1)
        self.assertListEqual(t.label_features(), [f1])
        t2 = ft.TensorDefinition(name_t, [f1, f2])
        t2.set_labels([f1, f2])
        self.assertListEqual(t2.label_features(), [f1, f2])
        with self.assertRaises(ft.TensorDefinitionException):
            t2.set_label(f3)

    def test_filtering(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureIndex('test-feature-2', ft.FEATURE_TYPE_INT_8, f1)
        f3 = ft.FeatureOneHot('test-feature-3', f1)
        f4 = ft.FeatureSource('test-feature-4', ft.FEATURE_TYPE_FLOAT)
        f5 = ft.FeatureNormalizeScale('test-feature-5', ft.FEATURE_TYPE_FLOAT, f4)
        f6 = ft.FeatureNormalizeStandard('test-feature-6', ft.FEATURE_TYPE_FLOAT, f4)
        f7 = ft.FeatureSource('test-feature-7', ft.FEATURE_TYPE_FLOAT)
        t = ft.TensorDefinition(name_t, [f1, f2, f3, f4, f5, f6, f7])
        t.set_label(f7)
        self.assertListEqual(t.categorical_features(), [f2])
        self.assertListEqual(t.binary_features(), [f3])
        self.assertListEqual(t.continuous_features(), [f5, f6])
        self.assertListEqual(t.label_features(), [f7])

    def test_highest_precision(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureSource('test-feature-4', ft.FEATURE_TYPE_FLOAT)
        f3 = ft.FeatureIndex('test-feature-2', ft.FEATURE_TYPE_INT_8, f1)
        t = ft.TensorDefinition(name_t, [f1, f2, f3])
        self.assertEqual(t.highest_precision_feature, f2, f'Wrong HP feature {t.highest_precision_feature}')
        t.remove(f2)
        t.remove(f3)
        with self.assertRaises(ft.TensorDefinitionException):
            _ = t.highest_precision_feature


def main():
    unittest.main()


if __name__ == '__main__':
    main()
