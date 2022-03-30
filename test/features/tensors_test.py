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
        self.assertEqual(t.inference_ready, True, f'Tensor should ready for inference, feature have no inf attributes')
        with self.assertRaises(ft.TensorDefinitionException):
            _ = t.rank

    def test_duplicate_bad(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.TensorDefinitionException):
            _ = ft.TensorDefinition(name_t, [f1, f1])

    def test_duplicate_name_bad(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_FLOAT)
        with self.assertRaises(ft.TensorDefinitionException):
            _ = ft.TensorDefinition(name_t, [f1, f2])

    def test_overlap_base_feature(self):
        # Should fail because the base feature is shared
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureIndex('test-feature-2', ft.FEATURE_TYPE_INT_8, f1)
        f3 = ft.FeatureOneHot('test-feature-3', ft.FEATURE_TYPE_INT_8, f1)
        with self.assertRaises(ft.TensorDefinitionException):
            _ = ft.TensorDefinition(name_t, [f1, f2, f3])

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

    def test_filtering(self):
        name_t = 'test-tensor'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureIndex('test-feature-2', ft.FEATURE_TYPE_INT_8, f1)
        f3 = ft.FeatureSource('test-feature-3', ft.FEATURE_TYPE_STRING)
        f4 = ft.FeatureOneHot('test-feature-4', ft.FEATURE_TYPE_INT_8, f3)
        f5 = ft.FeatureSource('test-feature-5', ft.FEATURE_TYPE_FLOAT)
        f6 = ft.FeatureNormalizeScale('test-feature-6', ft.FEATURE_TYPE_FLOAT, f5)
        f7 = ft.FeatureNormalizeStandard('test-feature-7', ft.FEATURE_TYPE_FLOAT, f5)
        f8 = ft.FeatureLabelBinary('test-feature-8', ft.FEATURE_TYPE_INT_8, f2)
        t = ft.TensorDefinition(name_t, [f1, f2, f3, f4, f5, f6, f7, f8])
        self.assertEqual(len(t.learning_categories), 4, f'Should be 4 categories. Got {len(t.learning_categories)}')
        self.assertListEqual(t.categorical_features(), [f2])
        self.assertListEqual(t.binary_features(), [f4])
        self.assertListEqual(t.continuous_features(), [f5, f6, f7])
        self.assertListEqual(t.label_features(), [f8])
        # Should fail because the Tensor Definition is ready for inference.
        with self.assertRaises(ft.TensorDefinitionException):
            t.categorical_features(True)
            t.binary_features(True)
            t.continuous_features(True)
            t.label_features(True)
            t.filter_features(ft.LEARNING_CATEGORY_CATEGORICAL, True)

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


class TestTensorMultiCreate(unittest.TestCase):
    def test_creation(self):
        name_t1 = 'test-tensor-1'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureSource('test-feature-2', ft.FEATURE_TYPE_STRING)
        t1 = ft.TensorDefinition(name_t1, [f1, f2])
        name_t2 = 'test-tensor-2'
        f3 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_FLOAT)
        f4 = ft.FeatureSource('test-feature-2', ft.FEATURE_TYPE_STRING)
        f5 = ft.FeatureLabelBinary('test-feature-3', ft.FEATURE_TYPE_INT_8, f3)
        t2 = ft.TensorDefinition(name_t2, [f3, f4, f5])
        t3 = ft.TensorDefinitionMulti([t1, t2])
        self.assertIsInstance(t3, ft.TensorDefinitionMulti, f'Creation failed. Not correct type {type(t3)}')
        t4, t5 = t3.tensor_definitions
        self.assertEqual(t1, t4, f'First Tensor Def don not match {t1.name} {t4.name}')
        self.assertEqual(t2, t5, f'Second Tensor Def don not match {t1.name} {t5.name}')
        self.assertEqual(t3.label_tensor_definition, t2, f'That is not the tensor def with the label')

    def test_creation_bad(self):
        name_t1 = 'test-tensor-1'
        f1 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_FLOAT)
        f2 = ft.FeatureLabelBinary('test-feature-3', ft.FEATURE_TYPE_INT_8, f1)
        t1 = ft.TensorDefinition(name_t1, [f1, f2])
        name_t2 = 'test-tensor-2'
        f3 = ft.FeatureSource('test-feature-1', ft.FEATURE_TYPE_FLOAT)
        f4 = ft.FeatureSource('test-feature-2', ft.FEATURE_TYPE_STRING)
        f5 = ft.FeatureLabelBinary('test-feature-3', ft.FEATURE_TYPE_INT_8, f3)
        t2 = ft.TensorDefinition(name_t2, [f3, f4, f5])
        # 2 TensorDefinitions with labels
        with self.assertRaises(ft.TensorDefinitionException):
            _ = ft.TensorDefinitionMulti([t1, t2])


def main():
    unittest.main()


if __name__ == '__main__':
    main()
