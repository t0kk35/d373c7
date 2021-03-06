"""
Unit Tests for Feature Creation
(c) 2020 d373c7
"""
import unittest
import d373c7.features as ft
from typing import Any


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
    def test_creation_base(self):
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

    def test_creation_non_int_type(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_BOOL
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureIndex(name, ft.FEATURE_TYPE_FLOAT_32, fs)

    def test_creation_base_bool(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_BOOL
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureIndex(name, ft.FEATURE_TYPE_INT_16, fs)

    def test_creation_base_float(self):
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
    def test_create_base(self):
        name = 'label'
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

    def test_creation_non_numerical_type(self):
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


class TestFeatureBin(unittest.TestCase):
    def test_creation_base(self):
        name = 'Bin'
        nr_bin = 10
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        bn = ft.FeatureBin(name, f_type, sf, nr_bin)
        self.assertIsInstance(bn, ft.FeatureBin, f'Not expected type {type(bn)}')
        self.assertEqual(bn.name, name, f'Feature Name should be {name}')
        self.assertEqual(bn.base_feature, sf, f'Base Feature not set correctly')
        self.assertEqual(len(bn.embedded_features), 1, f'Should only have 1 emb feature {len(bn.embedded_features)}')
        self.assertIn(sf, bn.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(bn.inference_ready, False, 'Should be not inference ready upon creation')
        self.assertEqual(bn.type, f_type, 'Must always be int-8 type. Smallest possible')
        self.assertEqual(bn.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Must have learning cat Categorical')
        self.assertEqual(bn.number_of_bins, nr_bin, f'Number of bins is wrong. Got {bn.number_of_bins}')
        self.assertEqual(len(bn), nr_bin, f'Length of feature should have been nr_bins. Should have. Got {len(bn)}')
        self.assertIsNone(bn.bins, f'Bins should not only be set after inference. Got {bn.bins}')
        self.assertIsInstance(bn.scale_type, bn.ScaleTypeLinear, f'Default ScaleType should have been Linear')
        self.assertEqual(bn.range, list(range(1, nr_bin)), f'Unexpected Range. Got {bn.range}')

    def test_creation_bad_base(self):
        name = 'Bin'
        nr_bin = 10
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureBin(name, f_type, sf, nr_bin)

    def test_creation_bad_type(self):
        name = 'Bin'
        nr_bin = 10
        f_type = ft.FEATURE_TYPE_FLOAT
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureBin(name, f_type, sf, nr_bin)

    def test_equality(self):
        s_name_1 = 's_test_1'
        s_name_2 = 's_test_2'
        b_name_1 = 'b_test_1'
        b_name_2 = 'b_test_2'
        f_type_1 = ft.FEATURE_TYPE_FLOAT
        f_type_2 = ft.FEATURE_TYPE_INT_8
        f_type_3 = ft.FEATURE_TYPE_INT_16
        fs1 = ft.FeatureSource(s_name_1, f_type_1)
        fs2 = ft.FeatureSource(s_name_2, f_type_1)
        fb1 = ft.FeatureBin(b_name_1, f_type_2, fs1, 10)
        fb2 = ft.FeatureBin(b_name_1, f_type_2, fs1, 10)
        fb3 = ft.FeatureBin(b_name_2, f_type_2, fs1, 10)
        fb4 = ft.FeatureBin(b_name_1, f_type_2, fs2, 10)
        fb5 = ft.FeatureBin(b_name_1, f_type_3, fs1, 10)
        self.assertEqual(fb1, fb2, f'Should have been equal')
        self.assertNotEqual(fb1, fb3, f'Should have been not equal')
        self.assertNotEqual(fb1, fb4, f'Should not have been equal. Different Base Feature')
        self.assertNotEqual(fb1, fb5, f'Should not have been equal. Different Type')


def feature_expression(param: int):
    return param + 1


class TestFeatureExpression(unittest.TestCase):
    def test_creation_base(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        ef = ft.FeatureExpression(name, f_type, feature_expression, par)
        self.assertIsInstance(ef, ft.FeatureExpression, f'Not expected type {type(ef)}')
        self.assertEqual(ef.name, name, f'Feature Name should be {name}')
        self.assertEqual(ef.type, f_type, f'Feature Type incorrect. Got {ef.type}')
        self.assertEqual(len(ef.embedded_features), len(par), f'Should have had {len(par)} embedded features')
        self.assertEqual(ef.embedded_features[0], par[0], f'Embedded Features should have been the parameters')
        self.assertEqual(ef.expression, feature_expression, f'Should have gotten the expression. Got {ef.expression}')
        self.assertEqual(ef.param_features, par, f'Did not get the parameters {ef.param_features}')

    def test_creation_base_lambda(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        ef = ft.FeatureExpression(name, f_type, lambda x: x + 1, par)
        self.assertEqual(ef.is_lambda, True, f'Should been lambda')

    def test_learning_features(self):
        name = 'expr'
        f_type_1 = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        ef = ft.FeatureExpression(name, f_type_1, feature_expression, par)
        self.assertEqual(ef.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL)
        f_type_2 = ft.FEATURE_TYPE_FLOAT
        ef = ft.FeatureExpression(name, f_type_2, feature_expression, par)
        self.assertEqual(ef.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS)
        f_type_3 = ft.FEATURE_TYPE_BOOL
        ef = ft.FeatureExpression(name, f_type_3, feature_expression, par)
        self.assertEqual(ef.learning_category, ft.LEARNING_CATEGORY_BINARY)
        f_type_4 = ft.FEATURE_TYPE_STRING
        ef = ft.FeatureExpression(name, f_type_4, feature_expression, par)
        self.assertEqual(ef.learning_category, ft.LEARNING_CATEGORY_NONE)

    def test_creation_bad_not_expression(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        bad: Any = 'bad'
        # Not an expression
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureExpression(name, f_type, bad, par)

    def test_creation_bad_param(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par_1: Any = ''
        # Param not a list
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureExpression(name, f_type, feature_expression, par_1)
        par_2: Any = ['']
        # Not list with Feature objects
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureExpression(name, f_type, feature_expression, par_2)
        par_3 = [sf, sf]
        # Incorrect Number of Parameters
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureExpression(name, f_type, feature_expression, par_3)


class TestFeatureExpressionSeries(unittest.TestCase):
    def test_creation_base(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        ef = ft.FeatureExpressionSeries(name, f_type, feature_expression, par)
        self.assertIsInstance(ef, ft.FeatureExpressionSeries, f'Not expected type {type(ef)}')
        self.assertEqual(ef.name, name, f'Feature Name should be {name}')
        self.assertEqual(ef.type, f_type, f'Feature Type incorrect. Got {ef.type}')
        self.assertEqual(len(ef.embedded_features), len(par), f'Should have had {len(par)} embedded features')
        self.assertEqual(ef.embedded_features[0], par[0], f'Embedded Features should have been the parameters')
        self.assertEqual(ef.expression, feature_expression, f'Should have gotten the expression. Got {ef.expression}')
        self.assertEqual(ef.param_features, par, f'Did not get the parameters {ef.param_features}')

    def test_creation_bad_lambda(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureExpressionSeries(name, f_type, lambda x: x + 1, par)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
