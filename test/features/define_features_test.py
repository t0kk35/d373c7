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
        self.assertEqual(f.learning_category, ft.LEARNING_CATEGORY_NONE, f'String should have learning type NONE')

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

    def test_equality(self):
        name_1 = 'test_1'
        name_2 = 'test_2'
        f_type_1 = ft.FEATURE_TYPE_STRING
        f_type_2 = ft.FEATURE_TYPE_FLOAT
        f1 = ft.FeatureSource(name_1, f_type_1)
        f2 = ft.FeatureSource(name_1, f_type_1)
        f3 = ft.FeatureSource(name_2, f_type_1)
        f4 = ft.FeatureSource(name_1, f_type_2)
        self.assertEqual(f1, f2, f'Should have been equal')
        self.assertNotEqual(f1, f3, f'Should have been not equal')
        self.assertNotEqual(f1, f4, f'Should not have been equal. Different Type')


class TestFeatureVirtual(unittest.TestCase):
    def test_creation_name_type(self):
        name = 'Virtual'
        f_type = ft.FEATURE_TYPE_STRING
        vf = ft.FeatureVirtual(name=name, f_type=f_type)
        self.assertIsInstance(vf, ft.FeatureVirtual, f'Not expected type {type(vf)}')
        self.assertEqual(vf.name, name, f'Name should have been {name}')
        self.assertEqual(vf.type, f_type, f'Type Should have been {f_type}')
        self.assertEqual(len(vf.embedded_features), 0, f'Virtual feature should not have embedded features')

    def test_creation_copy(self):
        name = 'copied'
        f_type = ft.FEATURE_TYPE_STRING
        sf = ft.FeatureSource(name, f_type)
        vf = ft.FeatureVirtual(sf)
        self.assertEqual(vf.name, sf.name, f'Virtual feature should have some name when copied')
        self.assertEqual(vf.type, sf.type, f'Virtual feature should have same type when copied')

    def test_creation_name_only(self):
        name = 'copied'
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureVirtual(name=name)

    def test_creation_copy_type_name(self):
        name = 'copied'
        f_type = ft.FEATURE_TYPE_STRING
        sf = ft.FeatureSource(name, f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureVirtual(sf, name=name, f_type=f_type)

    def test_creation_copy_type(self):
        name = 'copied'
        f_type = ft.FEATURE_TYPE_STRING
        sf = ft.FeatureSource(name, f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureVirtual(sf, f_type=f_type)

    def test_equality(self):
        name_1 = 'test_1'
        name_2 = 'test_2'
        f_type_1 = ft.FEATURE_TYPE_STRING
        f_type_2 = ft.FEATURE_TYPE_FLOAT
        v1 = ft.FeatureVirtual(ft.FeatureSource(name_1, f_type_1))
        v2 = ft.FeatureVirtual(ft.FeatureSource(name_1, f_type_1))
        v3 = ft.FeatureVirtual(ft.FeatureSource(name_2, f_type_1))
        v4 = ft.FeatureVirtual(ft.FeatureSource(name_1, f_type_2))
        self.assertEqual(v1, v2, f'Should have been equal')
        self.assertNotEqual(v1, v3, f'Should have been not equal')
        self.assertNotEqual(v1, v4, f'Should not have been equal. Different Type')


class TestFeatureOneHot(unittest.TestCase):
    def test_creation_base(self):
        name = 'OneHot'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        oh = ft.FeatureOneHot(name, sf)
        self.assertIsInstance(oh, ft.FeatureOneHot, f'Not expected type {type(oh)}')
        self.assertEqual(oh.name, name, f'Feature Name should be {name}')
        self.assertEqual(oh.base_feature, sf, f'Base Feature not set correctly')
        self.assertEqual(len(oh.embedded_features), 1, f'Should only have 1 emb feature {len(oh.embedded_features)}')
        self.assertIn(sf, oh.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(oh.inference_ready, False, 'Should be not inference ready upon creation')
        self.assertIsNone(oh.expand_names, f'Expand Names should be None {oh.expand_names}')
        self.assertEqual(len(oh.expand()), 0, f'Expand should yields empty list')
        self.assertEqual(oh.type, ft.FEATURE_TYPE_INT_8, 'Must always be int-8 type. Smallest possible')
        self.assertEqual(oh.learning_category, ft.LEARNING_CATEGORY_BINARY, f'Must have learning category Binary')

    def test_creation_non_string(self):
        name = 'OneHot'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureOneHot(name, sf)


class TestNormalizeScaleFeature(unittest.TestCase):
    def test_creation_base(self):
        name = 'scale'
        f_type = ft.FEATURE_TYPE_FLOAT
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        scf = ft.FeatureNormalizeScale(name, f_type, sf)
        self.assertIsInstance(scf, ft.FeatureNormalizeScale, f'Incorrect Type {type(scf)}')
        self.assertEqual(scf.name, name, f'Scale feature name incorrect {name}')
        self.assertEqual(scf.type, f_type, f'Scale feature type should have been {f_type}')
        self.assertEqual(scf.inference_ready, False, f'Scale feature should NOT be inference ready')
        self.assertIsNone(scf.minimum, f'Scale minimum should be None')
        self.assertIsNone(scf.maximum, f'Scale maximum should be None')
        self.assertEqual(scf.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Wrong Learning category')

    def test_creation_non_float(self):
        name = 'scale'
        f_type_str = ft.FEATURE_TYPE_STRING
        f_type_flt = ft.FEATURE_TYPE_FLOAT
        sf_flt = ft.FeatureSource('Source', f_type_flt)
        sf_str = ft.FeatureSource('Source', f_type_str)
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureNormalizeScale(name, f_type_str, sf_flt)
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureNormalizeScale(name, f_type_flt, sf_str)


class TestNormalizeStandardFeature(unittest.TestCase):
    def test_creation_base(self):
        name = 'standard'
        f_type = ft.FEATURE_TYPE_FLOAT
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        scf = ft.FeatureNormalizeStandard(name, f_type, sf)
        self.assertIsInstance(scf, ft.FeatureNormalizeStandard, f'Incorrect Type {type(scf)}')
        self.assertEqual(scf.name, name, f'Scale feature name incorrect {name}')
        self.assertEqual(scf.type, f_type, f'Scale feature type should have been {f_type}')
        self.assertEqual(scf.inference_ready, False, f'Scale feature should NOT be inference ready')
        self.assertIsNone(scf.mean, f'Scale mean should be None')
        self.assertIsNone(scf.stddev, f'Scale stddev should be None')
        self.assertEqual(scf.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Wrong Learning category')

    def test_creation_non_float(self):
        name = 'standard'
        f_type_str = ft.FEATURE_TYPE_STRING
        f_type_flt = ft.FEATURE_TYPE_FLOAT
        sf_flt = ft.FeatureSource('Source', f_type_flt)
        sf_str = ft.FeatureSource('Source', f_type_str)
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureNormalizeStandard(name, f_type_str, sf_flt)
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureNormalizeStandard(name, f_type_flt, sf_str)


class TestIndexFeature(unittest.TestCase):
    def creation_base(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_INT_16
        fs = ft.FeatureSource('source', f_type)
        fi = ft.FeatureIndex(name, ft.FEATURE_TYPE_INT_16, fs)
        self.assertIsInstance(fi, ft.FeatureIndex, f'Unexpected Type {type(fi)}')
        self.assertEqual(fi.name, name, f'Feature Name should be {name}')
        self.assertEqual(fi.type, f_type, f'Feature Type should be {f_type}')
        self.assertEqual(fi.base_feature, fs, f'Base Feature Should have been the source feature')
        self.assertEqual(len(fi.embedded_features), 1, f'Should only have 1 emb feature {len(fi.embedded_features)}')
        self.assertIn(fs, fi.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(fi.inference_ready, False, 'Should be not inference ready upon creation')
        self.assertEqual(fi.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Learning type should be CATEGORICAL')
        self.assertIsNone(fi.dictionary, f'Dictionary should be None')

    def creation_non_int_type(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_BOOL
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureIndex(name, ft.FEATURE_TYPE_FLOAT_32, fs)

    def creation_base_bool(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_BOOL
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureIndex(name, ft.FEATURE_TYPE_INT_16, fs)

    def creation_base_float(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureIndex(name, ft.FEATURE_TYPE_INT_16, fs)

    def test_equality(self):
        s_name_1 = 's_test_1'
        s_name_2 = 's_test_2'
        i_name_1 = 'i_test_1'
        i_name_2 = 'i_test_2'
        f_type_1 = ft.FEATURE_TYPE_INT_16
        f_type_2 = ft.FEATURE_TYPE_INT_8
        fs1 = ft.FeatureSource(s_name_1, f_type_1)
        fs2 = ft.FeatureSource(s_name_2, f_type_1)
        fi1 = ft.FeatureIndex(i_name_1, f_type_1, fs1)
        fi2 = ft.FeatureIndex(i_name_1, f_type_1, fs1)
        fi3 = ft.FeatureIndex(i_name_2, f_type_1, fs1)
        fi4 = ft.FeatureIndex(i_name_1, f_type_1, fs2)
        fi5 = ft.FeatureIndex(i_name_1, f_type_2, fs1)
        self.assertEqual(fi1, fi2, f'Should have been equal')
        self.assertNotEqual(fi1, fi3, f'Should have been not equal')
        self.assertNotEqual(fi1, fi4, f'Should not have been equal. Different Base Feature')
        self.assertNotEqual(fi1, fi5, f'Should not have been equal. Different Type')


class TestFeatureLabelBinary(unittest.TestCase):
    def test_create(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_INT_16
        fs = ft.FeatureSource('source', f_type)
        fl = ft.FeatureLabelBinary(name, fs)
        self.assertIsInstance(fl, ft.FeatureLabelBinary, f'Unexpected Type {type(fl)}')
        self.assertEqual(fl.name, name, f'Feature Name should be {name}')
        self.assertEqual(fl.type, ft.FEATURE_TYPE_INT_8, f'Feature Type should be int8 {fl.type}')
        self.assertEqual(fl.base_feature, fs, f'Base Feature Should have been the source feature')
        self.assertEqual(len(fl.embedded_features), 1, f'Should only have 1 emb feature {len(fl.embedded_features)}')
        self.assertIn(fs, fl.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(fl.learning_category, ft.LEARNING_CATEGORY_LABEL, f'Learning type should be LABEL')

    def creation_non_numerical_type(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_STRING
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureLabelBinary(name, fs)

    def test_equality(self):
        s_name_1 = 's_test_1'
        s_name_2 = 's_test_2'
        l_name_1 = 'l_test_1'
        l_name_2 = 'l_test_2'
        f_type_1 = ft.FEATURE_TYPE_INT_16
        f_type_2 = ft.FEATURE_TYPE_INT_8
        fs1 = ft.FeatureSource(s_name_1, f_type_1)
        fs2 = ft.FeatureSource(s_name_2, f_type_2)
        fl1 = ft.FeatureLabelBinary(l_name_1, fs1)
        fl2 = ft.FeatureLabelBinary(l_name_1, fs1)
        fl3 = ft.FeatureLabelBinary(l_name_2, fs1)
        fl4 = ft.FeatureLabelBinary(l_name_1, fs2)
        fl5 = ft.FeatureLabelBinary(l_name_1, fs1)
        self.assertEqual(fl1, fl2, f'Should have been equal')
        self.assertNotEqual(fl1, fl3, f'Should have been not equal')
        self.assertEqual(fl1, fl4, f'Should have been equal')
        self.assertEqual(fl1, fl5, f'Should have been equal')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
