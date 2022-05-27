"""
Unit Tests for PandasNumpy Engine
(c) 2020 d373c7
"""
import unittest
import numpy as np
from datetime import timedelta, datetime
from statistics import stdev
from typing import List
from collections import deque

import pandas as pd

import d373c7.features as ft
import d373c7.engines as en

FILES_DIR = './files/'


def fn_double(x: float) -> float:
    return x*2


def fn_not_one(x: float) -> bool:
    return x != 1.0


def fn_one(x: float) -> bool:
    return x == 1.0


class TestReading(unittest.TestCase):
    """Base Reading Tests
    """
    fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL)
    features = [
        (ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT), 'Float'),
        (ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING), 'String'),
        (fc, 'Categorical')
    ]

    def test_creation_base(self):
        threads = 1
        with en.EnginePandasNumpy(threads) as e:
            self.assertIsInstance(e, en.EnginePandasNumpy, 'PandasNumpy engine creation failed')
            self.assertEqual(e.num_threads, threads, f'Num_threads incorrect got {e.num_threads}')

    def test_read_base_single(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            for f, d in TestReading.features:
                td = ft.TensorDefinition(d, [f])
                df = e.from_csv(td, file, inference=False)
                self.assertEqual(len(df.columns), 1, f'Expected a one column panda for read test {d}')
                self.assertEqual(df.columns[0], f.name, f'Wrong panda column for read test {d}. Got {df.columns[0]}')
                self.assertEqual(td.inference_ready, True, 'TensorDefinition Should have been ready for inference')

    def test_read_base_all(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [f for f, d in TestReading.features]), file, inference=False)
            self.assertEqual(len(df.columns), len(TestReading.features),
                             f'Incorrect number of columns for read all test. got {len(df.columns)}')
            for (f, _), c in zip(TestReading.features, df.columns):
                self.assertEqual(f.name, c, f'Incorrect column name in read test all got {c}, expected {f.name}')

    def test_read_base_all_non_def_delimiter(self):
        file = FILES_DIR + 'engine_test_base_pipe.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [f for f, d in TestReading.features]),
                            file, inference=False, delimiter='|')
            self.assertEqual(len(df.columns), len(TestReading.features),
                             f'Incorrect number of columns for read all test. got {len(df.columns)}')
            for (f, _), c in zip(TestReading.features, df.columns):
                self.assertEqual(f.name, c, f'Incorrect column name in read test all got {c}, expected {f.name}')

    # TODO need test with multiple source features that are dates. There was an iterator problem?


class TestCategorical(unittest.TestCase):
    """Tests for categorical source feature
    """
    def test_is_categorical(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            self.assertEqual(df.dtypes[0], 'category', 'Source with f_type should be categorical')

    def test_categorical_default(self):
        default = 'DEF'
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default=default)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            self.assertEqual(fc.default, default, f'Default incorrect. Got {fc.default}')
            self.assertIn(default, list(df['MCC']), f'Default not found in Panda')


class TestOneHot(unittest.TestCase):
    """ Derived Features. One Hot feature tests. We'll use the from_df function for this
    """
    def test_read_base_non_inference(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fc])
            df = e.from_csv(td, file, inference=False)
            mcc_v = df['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            td2 = ft.TensorDefinition('Derived', [fo])
            df = e.from_df(td2, df, td, inference=False)
            self.assertEqual(len(df.columns), len(mcc_v), f'Col number must match values {len(df.columns), len(mcc_v)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')
            vt = set([vf for vf in fo.expand()])
            self.assertEqual(len(vt), len(mcc_v), f'Should have gotten {len(mcc_v)} expanded features')
            self.assertIsInstance(vt.pop(), ft.FeatureVirtual, f'Expanded features should be Virtual Features')
            vn = [vf.name for vf in fo.expand()]
            self.assertListEqual(vn, mcc_c, f'Names of the Virtual Features must match columns')
            self.assertEqual(td.inference_ready, True, f'Tensor should have been ready for inference now')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertListEqual(td2.binary_features(True), fo.expand(), f'Expanded Feature not correct')

    def test_root_missing(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fd = ft.FeatureSource('Card', ft.FEATURE_TYPE_CATEGORICAL)
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fd])
            df = e.from_csv(td, file, inference=False)
            with self.assertRaises(en.EnginePandaNumpyException):
                e.from_df(ft.TensorDefinition('Derived', [fo]), df, td, inference=False)

    def test_read_base_inference_removed_element(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td_c = ft.TensorDefinition('All', [fc])
            df_c = e.from_csv(td_c, file, inference=False)
            mcc_v = df_c['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            # Set derived Feature
            td = ft.TensorDefinition('Derived', [fo])
            e.from_df(td, df_c, td_c, inference=False)
            # Now Remove a line from the original csv panda and do an inference run.
            df_c = df_c.iloc[:-1]
            df = e.from_df(td, df_c, td_c, inference=True)
            self.assertEqual(len(df.columns), len(mcc_v), f'Col number must match values {len(df.columns), len(mcc_v)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')

    def test_read_base_inference_added_element(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td_c = ft.TensorDefinition('All', [fc])
            df_c = e.from_csv(td_c, file, inference=False)
            # Remove row and set variables. Also remove from categories!
            df_r = df_c.iloc[1:].copy()
            df_r['MCC'] = df_c['MCC'].cat.remove_categories(df_c['MCC'][0])
            mcc_v = df_r['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            # Run non-inference on removed panda
            td = ft.TensorDefinition('Derived', [fo])
            e.from_df(td, df_r, td_c, inference=False)
            # And inference on original
            df = e.from_df(td, df_c, td_c, inference=True)
            self.assertEqual(len(df.columns), len(mcc_v), f'Col number must match values {len(df.columns), len(mcc_v)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')


class TestNormalizeScale(unittest.TestCase):
    """ Derived Features. Normalize Scale feature tests. We'll use the from_df function for this
    """
    def test_read_base_non_inference(self):
        fc = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountScale'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeScale(s_name, s_type, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fc])
            df = e.from_csv(td, file, inference=False)
            mn = df['Amount'].min()
            mx = df['Amount'].max()
            td2 = ft.TensorDefinition('Derived', [fs])
            df = e.from_df(td2, df, td, inference=False)
            self.assertEqual(len(df.columns), 1, f'Only one columns should have been returned')
            self.assertEqual(df.columns[0], s_name, f'Column name is not correct {df.columns[0]}')
            self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
            self.assertEqual(fs.maximum, mx, f'Maximum not set correctly {fs.maximum}')
            self.assertEqual(fs.minimum, mn, f'Minimum not set correctly {fs.maximum}')
            self.assertListEqual(td2.continuous_features(True), [fs], f'Expanded Feature not correct')

    def test_read_w_logs_non_inference(self):
        # If logs are defined, then the amount should be logged first and then the normalizer applied.
        fc = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountScaleAndLog'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        file = FILES_DIR + 'engine_test_base_comma.csv'
        for log, log_fn in (('e', np.log), ('2', np.log2), ('10', np.log10)):
            with en.EnginePandasNumpy() as e:
                fs = ft.FeatureNormalizeScale(s_name, s_type, fc, log)
                td = ft.TensorDefinition('All', [fc])
                df = e.from_csv(td, file, inference=False)
                mn = log_fn(df['Amount']+fs.delta).min()
                mx = log_fn(df['Amount']+fs.delta).max()
                td2 = ft.TensorDefinition('Derived', [fs])
                df = e.from_df(td2, df, td, inference=False)
                self.assertEqual(len(df.columns), 1, f'Only one columns should have been returned')
                self.assertEqual(df.columns[0], s_name, f'Column name is not correct {df.columns[0]}')
                self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
                self.assertAlmostEqual(fs.maximum, mx, 5, f'Maximum not set correctly {fs.maximum}')
                self.assertAlmostEqual(fs.minimum, mn, 5, f'Minimum not set correctly {fs.maximum}')
                self.assertListEqual(td2.continuous_features(True), [fs], f'Expanded Feature not correct')

    def test_read_base_inference(self):
        fc = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountScale'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeScale(s_name, s_type, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td_df = ft.TensorDefinition('All', [fc])
            df = e.from_csv(td_df, file, inference=False)
            td = ft.TensorDefinition('Derived', [fs])
            df_1 = e.from_df(td, df, td_df, inference=False)
            # Now remove a line and run in inference mode
            df = df.iloc[:-1]
            df_1 = df_1.iloc[:-1]
            df_2 = e.from_df(td, df, td_df, inference=True)
            self.assertEqual(df_1[s_name].equals(df_2[s_name]), True, f'Inference problem. Series not the same')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')


class TestNormalizeStandard(unittest.TestCase):
    """ Derived Features. Normalize Standard feature tests. We'll use the from_df function for this
    """
    def test_read_base_non_inference(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fc = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountScaleAndLog'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeStandard(s_name, s_type, fc)
        with en.EnginePandasNumpy() as e:
            td_df = ft.TensorDefinition('All', [fc])
            df = e.from_csv(td_df, file, inference=False)
            mn = df['Amount'].mean()
            st = df['Amount'].std()
            td2 = ft.TensorDefinition('Derived', [fs])
            df = e.from_df(td2, df, td_df, inference=False)
            self.assertEqual(len(df.columns), 1, f'Only one columns should have been returned')
            self.assertEqual(df.columns[0], s_name, f'Column name is not correct {df.columns[0]}')
            self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
            self.assertEqual(fs.mean, mn, f'Mean not set correctly {fs.mean}')
            self.assertEqual(fs.stddev, st, f'Standard Dev not set correctly {fs.stddev}')
            self.assertListEqual(td2.continuous_features(True), [fs], f'Expanded Feature not correct')

    def test_read_w_logs_non_inference(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        # If logs are defined, then the amount should be logged first and then the normalizer applied.
        fc = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountScale'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        for log, log_fn in (('e', np.log), ('2', np.log2), ('10', np.log10)):
            fs = ft.FeatureNormalizeStandard(s_name, s_type, fc, log)
            with en.EnginePandasNumpy() as e:
                td_df = ft.TensorDefinition('All', [fc])
                df = e.from_csv(td_df, file, inference=False)
                mn = log_fn(df['Amount']+fs.delta).mean()
                st = log_fn(df['Amount']+fs.delta).std()
                td2 = ft.TensorDefinition('Derived', [fs])
                df = e.from_df(td2, df, td_df, inference=False)
                self.assertEqual(len(df.columns), 1, f'Only one columns should have been returned')
                self.assertEqual(df.columns[0], s_name, f'Column name is not correct {df.columns[0]}')
                self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
                self.assertEqual(fs.mean, mn, f'Mean not set correctly {fs.mean}')
                self.assertEqual(fs.stddev, st, f'Standard Dev not set correctly {fs.stddev}')
                self.assertListEqual(td2.continuous_features(True), [fs], f'Expanded Feature not correct')

    def test_read_base_inference(self):
        fc = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountScale'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeStandard(s_name, s_type, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td_df = ft.TensorDefinition('All', [fc])
            df = e.from_csv(td_df, file, inference=False)
            td = ft.TensorDefinition('Derived', [fs])
            df_1 = e.from_df(td, df, td_df, inference=False)
            # Now remove a line and run in inference mode
            df = df.iloc[:-1]
            df_1 = df_1.iloc[:-1]
            df_2 = e.from_df(td, df, td_df, inference=True)
            self.assertEqual(df_1[s_name].equals(df_2[s_name]), True, f'Inference problem. Series not the same')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')


class TestIndex(unittest.TestCase):
    """Index Feature Testcases"""
    def test_read_base_non_inference(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        ind_name = 'MCC_ID'
        fi = ft.FeatureIndex(ind_name, ft.FEATURE_TYPE_INT_16, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fc])
            df = e.from_csv(td, file, inference=False)
            mcc_v = df['MCC'].unique()
            td2 = ft.TensorDefinition('Derived', [fi])
            df = e.from_df(td2, df, td, inference=False)
            self.assertEqual(len(df.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(df.columns[0], ind_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(fi.inference_ready, True, f'Index feature should be ready for inference')
            self.assertEqual(len(fi.dictionary), len(mcc_v), f'Dictionary length not correct {len(fi.dictionary)}')
            self.assertListEqual(list(fi.dictionary.keys()), list(mcc_v), f'Dictionary values don not match')
            self.assertEqual(td.inference_ready, True, f'Tensor should have been ready for inference now')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertListEqual(td2.categorical_features(True), [fi], f'Expanded Feature not correct')

    def test_read_remove_element(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        ind_name = 'MCC_ID'
        fi = ft.FeatureIndex(ind_name, ft.FEATURE_TYPE_INT_16, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td_df = ft.TensorDefinition('All', [fc])
            df = e.from_csv(td_df, file, inference=False)
            td = ft.TensorDefinition('Derived', [fi])
            df_1 = e.from_df(td, df, td_df, inference=False)
            # Now remove a line and run in inference mode
            df = df.iloc[1:]
            df_1 = df_1.iloc[1:]
            df_2 = e.from_df(td, df, td_df, inference=True)
            mcc_v = df_2[ind_name].unique()
            self.assertEqual(df_1[ind_name].equals(df_2[ind_name]), True, f'Inference problem. Series not the same')
            self.assertEqual(len(fi), len(mcc_v) + 1, f'Length of dictionary changed {len(fi)}')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')

    def test_read_add_element(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        ind_name = 'MCC_ID'
        fi = ft.FeatureIndex(ind_name, ft.FEATURE_TYPE_INT_16, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td_c = ft.TensorDefinition('All', [fc])
            df_c = e.from_csv(td_c, file, inference=False)
            # Remove row and set variables. Also remove from categories!
            df_r = df_c.iloc[1:].copy()
            df_r['MCC'] = df_c['MCC'].cat.remove_categories(df_c['MCC'][0])
            mcc_v = df_r['MCC'].unique()
            # Run non-inference on removed panda
            td = ft.TensorDefinition('Derived', [fi])
            e.from_df(td, df_r, td_c, inference=False)
            # And inference on original
            df = e.from_df(td, df_c, td_c, inference=True)
            self.assertEqual(len(fi), len(mcc_v), f'Length of dictionary changed {len(fi)}')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertEqual(df[ind_name][0], 0, f'Missing row should have been default {df[ind_name][0]}')


class TestFeatureBin(unittest.TestCase):
    def test_read_base_non_inference(self):
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        bin_name = 'Amount_bin'
        nr_bins = 3
        fb = ft.FeatureBin(bin_name, ft.FEATURE_TYPE_INT_16, fa, nr_bins)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fa])
            df = e.from_csv(td, file, inference=False)
            mx = np.finfo(df['Amount'].dtype).max
            mn = df['Amount'].min()
            md = df['Amount'].mean()
            td2 = ft.TensorDefinition('Derived', [fb])
            df = e.from_df(td2, df, td, inference=False)
            self.assertEqual(len(df.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(df.columns[0], bin_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(fb.inference_ready, True, f'Index feature should be ready for inference')
            self.assertEqual(fb.bins, [mn, md, mx], f'Bins not set as expected. Got {fb.bins}')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')
            self.assertListEqual(td2.categorical_features(), [fb], f'Expanded Feature not correct')
            self.assertEqual(sorted(list(df[bin_name].unique())), list(range(0, fb.number_of_bins)))

    def test_read_remove_element(self):
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        bin_name = 'Amount_bin'
        nr_bins = 3
        fb = ft.FeatureBin(bin_name, ft.FEATURE_TYPE_INT_16, fa, nr_bins)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td_df = ft.TensorDefinition('All', [fa])
            df = e.from_csv(td_df, file, inference=False)
            td = ft.TensorDefinition('Derived', [fb])
            _ = e.from_df(td, df, td_df, inference=False)
            mx = np.finfo(df['Amount'].dtype).max
            mn = df['Amount'].min()
            md = df['Amount'].mean()
            # Now remove a line and run in inference mode
            df = df.iloc[1:]
            df_2 = e.from_df(td, df, td_df, inference=True)
            bin_v = df_2[bin_name].unique()
            # Should not have a 0 bin. As we removed the 0.0 amount
            self.assertNotIn(0, bin_v, f'Should not have had a 0 value')
            self.assertEqual(fb.bins, [mn, md, mx], f'Bins probably changed. Got {fb.bins}')
            self.assertEqual(len(bin_v), fb.number_of_bins - 1, f'Not missing a value. Got len{len(bin_v)}')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')

    def test_read_add_element(self):
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        bin_name = 'Amount_bin'
        nr_bins = 3
        fb = ft.FeatureBin(bin_name, ft.FEATURE_TYPE_INT_16, fa, nr_bins)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td_c = ft.TensorDefinition('All', [fa])
            df_c = e.from_csv(td_c, file, inference=False)
            # Remove first and last row and set variables.
            df_r = df_c.iloc[1:-1].copy()
            # Run non-inference on removed panda
            td = ft.TensorDefinition('Derived', [fb])
            df_n = e.from_df(td, df_r, td_c, inference=False)
            bin_v = df_n[bin_name].unique()
            # And inference on original
            df = e.from_df(td, df_c, td_c, inference=True)
            self.assertEqual(fb.number_of_bins, len(bin_v), f'Number of bins changed {fb.number_of_bins}')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertEqual(df[bin_name][0], 0, f'Missing row should have been in 0 bin {df[bin_name][0]}')
            self.assertEqual(df[bin_name].iloc[-1], nr_bins-1, f'Last should have max bin {df[bin_name].iloc[-1]}')


def test_expr(x: float) -> float:
    return x + 1


def zero_expr(x: float) -> float:
    return 0.0


def test_date_expr(x: int) -> datetime:
    y = pd.to_datetime(x)
    y = y + timedelta(days=1)
    return y


class TestFeatureExpression(unittest.TestCase):
    """Test cases for Expression Feature"""
    def test_expr_lambda_float(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fe = ft.FeatureExpression('AddAmount', ft.FEATURE_TYPE_FLOAT_32, lambda x: x+1, [fa])
        td1 = ft.TensorDefinition('base', [fa])
        td2 = ft.TensorDefinition('derived', [fe])
        with en.EnginePandasNumpy() as e:
            df1 = e.from_csv(td1, file, inference=False)
            df2 = e.from_csv(td2, file, inference=False)
        df1['Amount'] = df1['Amount'] + 1
        self.assertTrue(df1['Amount'].equals(df2['AddAmount']), f'Amounts should have been equal')

    def test_expr_non_lambda_float(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fe = ft.FeatureExpression('AddAmount', ft.FEATURE_TYPE_FLOAT_32, test_expr, [fa])
        td1 = ft.TensorDefinition('base', [fa])
        td2 = ft.TensorDefinition('derived', [fe])
        with en.EnginePandasNumpy() as e:
            df1 = e.from_csv(td1, file, inference=False)
            df2 = e.from_csv(td2, file, inference=False)
        df1['Amount'] = df1['Amount'] + 1
        self.assertTrue(df1['Amount'].equals(df2['AddAmount']), f'Amounts should have been equal')

    def test_expr_non_lambda_date(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fe = ft.FeatureExpression('Add1ToDate', ft.FEATURE_TYPE_DATE, test_date_expr, [fd])
        td1 = ft.TensorDefinition('base', [fa, fd])
        td2 = ft.TensorDefinition('derived', [fe])
        with en.EnginePandasNumpy() as e:
            df1 = e.from_csv(td1, file, inference=False)
            df2 = e.from_csv(td2, file, inference=False)
            df1['Date'] = df1['Date'] + timedelta(days=1)
            self.assertTrue(df1['Date'].equals(df2['Add1ToDate']), f'Dates should have been equal')


class TestFeatureRatio(unittest.TestCase):
    def test_base_ratio(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        ratio_name = 'ratio'
        fd = ft.FeatureExpression('AddAmount', ft.FEATURE_TYPE_FLOAT, test_expr, [fa])
        fr = ft.FeatureRatio(ratio_name, ft.FEATURE_TYPE_FLOAT, fa, fd)
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fa, fd, fr])
            df = e.from_csv(td, file, inference=False)
            df['ratio-2'] = df[fa.name].div(df[fd.name])
            self.assertTrue(df[fr.name].equals(df['ratio-2']), f'Ratios not equal')

    def test_zero_denominator(self):
        # Test if the zero division return 0 instead of an error or np.inf
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        ratio_name = 'ratio'
        fd = ft.FeatureExpression('ZeroAmount', ft.FEATURE_TYPE_FLOAT, zero_expr, [fa])
        fr = ft.FeatureRatio(ratio_name, ft.FEATURE_TYPE_FLOAT, fa, fd)
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fa, fd, fr])
            df = e.from_csv(td, file, inference=False)
            self.assertTrue((df[fr.name] == 0.0).all(), f'Ratios not all zero')


class TestFeatureConcat(unittest.TestCase):
    def test_base_concat(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_STRING)
        concat_name = 'concat'
        fn = ft.FeatureConcat(concat_name, ft.FEATURE_TYPE_STRING, fc, fm)
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fc, fm, fn])
            df = e.from_csv(td, file, inference=False)
            df['concat-1'] = df[fc.name].astype(str) + df[fm.name].astype(str)
            self.assertTrue(df[fn.name].equals(df['concat-1']), f'Concat columns not equal')

    def test_multiple_concat(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_STRING)
        concat_name_1 = 'concat_1'
        concat_name_2 = 'concat_2'
        fn1 = ft.FeatureConcat(concat_name_1, ft.FEATURE_TYPE_STRING, fc, fm)
        fn2 = ft.FeatureConcat(concat_name_2, ft.FEATURE_TYPE_STRING, fr, fr)
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fc, fm, fr, fn1, fn2])
            df = e.from_csv(td, file, inference=False)
            df['concat-1'] = df[fc.name].astype(str) + df[fm.name].astype(str)
            df['concat-2'] = df[fr.name].astype(str) + df[fr.name].astype(str)
            self.assertTrue(df[fn1.name].equals(df['concat-1']), f'Concat columns not equal')
            self.assertTrue(df[fn2.name].equals(df['concat-2']), f'Concat columns not equal')


class TestReshape(unittest.TestCase):
    """Reshaping Tests
    """
    def test_reshape(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        td1 = ft.TensorDefinition('All', [f for f, d in TestReading.features])
        td2 = ft.TensorDefinition('All', [f for f, d in TestReading.features])
        rf = TestReading.features[0][0]
        td2.remove(rf)
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            df = e.reshape(td2, df)
            self.assertEqual(len(df.columns), len(td1)-1, f'Reshape Panda wrong length. gor {len(df.columns)}')
            self.assertNotIn(rf.name, df.columns, f'{rf.name} should not have been in reshaped Panda')

    def test_reshape_bad(self):
        # Should fail because the data frame is missing a column requested by reshape.
        file = FILES_DIR + 'engine_test_base_comma.csv'
        td1 = ft.TensorDefinition('All', [f for f, d in TestReading.features])
        td2 = ft.TensorDefinition('All', [f for f, d in TestReading.features])
        rf = TestReading.features[0][0]
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            df = df.drop(columns=[rf.name])
            with self.assertRaises(en.EnginePandaNumpyException):
                _ = e.reshape(td2, df)


class TestToNumpy(unittest.TestCase):
    """Test for conversion to_numpy_list
    """
    def test_create(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL, default='NA')
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fa, fm, fc, ff])
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fc)
        fi = ft.FeatureIndex('MCC_ID', ft.FEATURE_TYPE_INT_16, fm)
        fl = ft.FeatureLabelBinary('Fraud_Label', ft.FEATURE_TYPE_INT_8, ff)
        td2 = ft.TensorDefinition('Derived', [fa, fo, fi, fl])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            df = e.from_df(td2, df, td1, inference=False)
            npl = e.to_numpy_list(td2, df)
            self.assertEqual(len(npl), len(df), f'length of numpy list incorrect Got {len(npl)}. Expected {len(df)}')
            self.assertEqual(len(npl.lists), len(td2.learning_categories), f'Wrong number of lists {len(npl.lists)}')
            for lc, l in zip(td2.learning_categories, npl.lists):
                self.assertEqual(l.shape[0], len(df), f'In correct shape 0 for List {lc}')
            for lc, l in zip(td2.learning_categories, npl.lists):
                if lc == ft.LEARNING_CATEGORY_CONTINUOUS:
                    self.assertEqual(len(l.shape), 1, 'Continuous List dim 1 incorrect. Should have been 1')
                elif lc == ft.FEATURE_TYPE_CATEGORICAL:
                    self.assertEqual(len(l.shape), 1, 'Categorical List dim 1 incorrect. Should have been 1')
                elif lc == ft.LEARNING_CATEGORY_BINARY:
                    self.assertEqual(l.shape[1], len(fo.expand_names), f'Binary List dim 1 incorrect. {l.shape[1]}')
                elif lc == ft.LEARNING_CATEGORY_LABEL:
                    self.assertEqual(len(l.shape), 1, 'Label List dim 1 incorrect. Should have been 1')
            for lc, l in zip(td2.learning_categories, npl.lists):
                if lc == ft.LEARNING_CATEGORY_CONTINUOUS:
                    self.assertEqual(l.min(initial=1000), df[fa.name].min(), f'Min do not match {l.min(initial=1000)}')
                    self.assertEqual(l.max(initial=0), df[fa.name].max(), f'Max do not match {l.max(initial=0)}')
                    # Numpy Uses ddof=0 by default.
                    self.assertEqual(l.std(), df[fa.name].std(ddof=False), f'Std do not match {l.std()}')
                elif lc == ft.FEATURE_TYPE_CATEGORICAL:
                    self.assertEqual(l.min(initial=100), min(fi.dictionary.values()), f'Min wrong {l.min(initial=100)}')
                    self.assertEqual(l.max(initial=0), max(fi.dictionary.values()), f'Max wrong {l.max(initial=0)}')
                elif lc == ft.LEARNING_CATEGORY_BINARY:
                    self.assertEqual(l.min(initial=10), 0, f'Min wrong {l.min(initial=10)}')
                    self.assertEqual(l.max(initial=0), 1, f'Min wrong {l.min(initial=0)}')
                if lc == ft.LEARNING_CATEGORY_LABEL:
                    self.assertEqual(int(l.min(initial=10)), df[fl.name].min(), f'Min problem {l.min(initial=1000)}')
                    self.assertEqual(int(l.max(initial=0)), df[fl.name].max(), f'Max do not match {l.max(initial=0)}')


class TestIsBuiltFrom(unittest.TestCase):
    def test_create(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL, default='NA')
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fa, fm, fc, ff])
        fo = ft.FeatureOneHot('MCC_OH', ft.FEATURE_TYPE_INT_8, fm)
        fi = ft.FeatureIndex('Country_ID', ft.FEATURE_TYPE_INT_16, fc)
        fl = ft.FeatureLabelBinary('Fraud_Label', ft.FEATURE_TYPE_INT_8, ff)
        td2 = ft.TensorDefinition('Derived', [fa, fo, fi, fl])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            df = e.from_df(td2, df, td1, inference=False)
            npl = e.to_numpy_list(td2, df)
            self.assertEqual(npl.is_built_from(td2), True, f'Should have yielded true')
            self.assertEqual(npl.is_built_from(td1), False, f'Should have yielded False')
            # Strip off element from the binary List
            lists = npl.lists
            lists[0] = lists[0][:, 0:3]
            npl2 = en.NumpyList(lists)
            self.assertEqual(npl2.is_built_from(td2), False, f'Should have yielded False')
            # Remove entire list
            npl.pop(0)
            self.assertEqual(npl.is_built_from(td2), False, f'Should have yielded False')


def test_calc_delta(dates):
    if isinstance(dates, pd.DataFrame):
        res = dates.diff() / np.timedelta64(1, 'D')
        res = res.fillna(0).abs()
        return res
    else:
        # There was only 1 row
        return 0


class TestSeriesStacked(unittest.TestCase):
    def test_create(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL, default='NA')
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fa, fm, fc, ff])
        fi = ft.FeatureIndex('MCC_ID', ft.FEATURE_TYPE_INT_16, fm)
        fl = ft.FeatureLabelBinary('Fraud_Label', ft.FEATURE_TYPE_INT_8, ff)
        td2 = ft.TensorDefinition('Derived', [fd, fr, fa, fi, fl])
        with en.EnginePandasNumpy() as e:
            s = e.to_series_stacked(td2, file, fr, fd, 3, inference=False)
        self.assertTrue(td2.inference_ready, f'TensorDefinition should have been ready for inference')
        self.assertEqual(td2.rank, 3, f'Rank of TensorDefinition should have been 3')
        # TODO actually add tests

    def test_series_expression(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fa, ff])
        fe = ft.FeatureExpressionSeries('DateDelta', ft.FEATURE_TYPE_FLOAT_32, test_calc_delta, [fd])
        td2 = ft.TensorDefinition('Derived', [fe])
        window = 3
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            s = e.to_series_stacked(td2, file, fr, fd, window, inference=False)
        self.assertTrue(td2.inference_ready, f'TensorDefinition should have been ready for inference')
        self.assertEqual(td2.rank, 3, f'Rank of TensorDefinition should have been 3')
        # Iterate over the card-id's
        for c in df['Card'].unique():
            # Filter by card, sort by date and add the time delta
            f = df.loc[df['Card'] == c]
            f = f.sort_values(by=['Date'])
            f['Delta'] = f.loc[:, ['Date']].diff() / np.timedelta64(1, 'D')
            f['Delta'] = f.loc[:, ['Delta']].fillna(0).abs()
            for i, (index, row) in enumerate(f.iterrows()):
                # This will be a (1 x series x features) shape numpy, one single row of the series we created
                n = s[index].lists[0]
                for j, r in enumerate(range(window-1, max(window-i-2, -1), -1)):
                    b = n[0, r, 0]
                    v = f['Delta'].iloc[i-j]
                    self.assertEqual(b, v, f'Problem for key {c} at row {i} for window position {r}')


class TestGrouperFeature(unittest.TestCase):
    def test_grouped_bad_no_time_feature(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fx = ft.FeatureExpression('DateDerived', ft.FEATURE_TYPE_DATE, fn_one, [fd])
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        fg = ft.FeatureGrouper(
            '2_day_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_SUM)
        td2 = ft.TensorDefinition('Derived', [fd, fr, fa, ff, fg])
        tdx = ft.TensorDefinition('Derived', [fx, fr, fa, ff, fg])
        with en.EnginePandasNumpy() as e:
            # No time feature is bad
            with self.assertRaises(en.EnginePandaNumpyException):
                _ = e.from_csv(td2, file, inference=False)
            # No time feature is bad. if it is derived also; i.e. embedded.
            with self.assertRaises(en.EnginePandaNumpyException):
                _ = e.from_csv(tdx, file, inference=False)
            # Time Feature not of datetime type is also bad
            with self.assertRaises(en.EnginePandaNumpyException):
                _ = e.from_csv(td2, file, time_feature=fa, inference=False)

    def test_grouped_bad_base_not_float(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fa, ff])
        fg = ft.FeatureGrouper(
            '2_day_sum', ft.FEATURE_TYPE_FLOAT_32, ff, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_SUM)
        td2 = ft.TensorDefinition('Derived', [fd, fr, fa, ff, fg])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            with self.assertRaises(en.EnginePandaNumpyException):
                _ = e.from_df(td2, df, td1, inference=False)

    def test_grouped_single_window_all_aggregates(self):
        # Base test. Create single aggregate daily sum on card
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        time_window = 2

        feature_def = [
            ('2_day_sum', ft.AGGREGATOR_SUM, sum),
            ('2_day_cnt', ft.AGGREGATOR_COUNT, len),
            ('2_day_avg', ft.AGGREGATOR_AVG, lambda x: sum(x) / len(x)),
            ('2_day_min', ft.AGGREGATOR_MIN, min),
            ('2_day_max', ft.AGGREGATOR_MAX, max),
            ('2_day_std', ft.AGGREGATOR_STDDEV, lambda x: 0 if len(x) == 1 else stdev(x))
        ]
        group_features = [
            ft.FeatureGrouper(name, ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, time_window, agg)
            for name, agg, _ in feature_def
        ]
        features: List[ft.Feature] = [
            fd, fr, fa, ff
        ]
        features.extend(group_features)
        td = ft.TensorDefinition('Features', features)
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td, file, inference=False, time_feature=fd)

        # Check that all GrouperFeature have been created.
        for grouper_name, _, _ in feature_def:
            self.assertIn(grouper_name, df.columns, f'The aggregated feature {grouper_name} not found in Pandas')

        # Check that all types are correct. They should all have been created as float 32
        for f in group_features:
            self.assertEqual(df[f.name].dtype.name, 'float32', f'Type not FP32 for {df[f.name]}, {df[f.name].dtype}')

        # Check the aggregates. Iterate over the Card-id (the key) and check each aggregate
        for c in df[fr.name].unique():
            prev_dt = None
            amounts = []
            for _, row in df.iterrows():
                if row[fr.name] == c:
                    if prev_dt is None:
                        amounts.append(row[fa.name])
                        for grouper_name, _, list_fn in feature_def:
                            amt = list_fn(amounts)
                            g_amt = row[grouper_name]
                            self.assertAlmostEqual(amt, g_amt, 6, f'Expected {grouper_name} to be same as the amount')
                    elif row[fd.name] >= (prev_dt + timedelta(days=time_window)):
                        amounts = [row[fa.name]]
                        for grouper_name, _, list_fn in feature_def:
                            amt = list_fn(amounts)
                            g_amt = row[grouper_name]
                            self.assertAlmostEqual(amt, g_amt, 6, f'Expected {grouper_name} to be {g_amt}')
                    else:
                        amounts.append(row[fa.name])
                        for grouper_name, _, list_fn in feature_def:
                            amt = list_fn(amounts)
                            g_amt = row[grouper_name]
                            # Do almost equal. There's a slight difference in stddev between pandas and the stats impl
                            self.assertAlmostEqual(amt, g_amt, 6, f'Expected {grouper_name} to be {g_amt}')
                    prev_dt = row[fd.name]

    def test_grouped_multiple_groups(self):
        # Base test. Create single all aggregates on 2 groups
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fc, fa, ff])
        fga_1 = ft.FeatureGrouper(
            'card_2d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_SUM
        )
        fga_2 = ft.FeatureGrouper(
            'card_2d_count', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_COUNT
        )
        fga_3 = ft.FeatureGrouper(
            'card_2d_avg', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_AVG
        )
        fga_4 = ft.FeatureGrouper(
            'card_2d_std', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_STDDEV
        )
        fga_5 = ft.FeatureGrouper(
            'card_2d_min', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_MIN
        )
        fga_6 = ft.FeatureGrouper(
            'card_2d_max', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_MAX
        )
        fgc_1 = ft.FeatureGrouper(
            'country_2d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_SUM
        )
        fgc_2 = ft.FeatureGrouper(
            'country_2d_count', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_COUNT
        )
        fgc_3 = ft.FeatureGrouper(
            'country_2d_avg', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_AVG
        )
        fgc_4 = ft.FeatureGrouper(
            'country_2d_std', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_STDDEV
        )
        fgc_5 = ft.FeatureGrouper(
            'country_2d_min', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_MIN
        )
        fgc_6 = ft.FeatureGrouper(
            'country_2d_max', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, ft.TIME_PERIOD_DAY, 2, ft.AGGREGATOR_MAX
        )

        td2 = ft.TensorDefinition('Derived', [fga_1, fga_2, fga_3, fga_4, fga_5, fga_6])
        td3 = ft.TensorDefinition('Derived', [fgc_1, fgc_2, fgc_3, fgc_4, fgc_5, fgc_6])
        td4 = ft.TensorDefinition('Derived', [fga_1, fga_2, fga_3, fga_4, fga_5, fga_6,
                                              fgc_1, fgc_2, fgc_3, fgc_4, fgc_5, fgc_6])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            df_card = e.from_df(td2, df, td1, inference=False, time_feature=fd)
            df_country = e.from_df(td3, df, td1, inference=False, time_feature=fd)
            df_comb = e.from_df(td4, df, td1, inference=False, time_feature=fd)
            # The resulting df_comb (with 2 groups) should be the same and doing each individual and concatenating
            self.assertTrue(df_comb.equals(pd.concat([df_card, df_country], axis=1)), f'Concatenate dataframe not ==')

    def test_filter(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fl = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        f_not_one = ft.FeatureFilter('Not_Fraud', ft.FEATURE_TYPE_BOOL, fn_not_one, [fl])
        f_is_one = ft.FeatureFilter('Is_Fraud', ft.FEATURE_TYPE_BOOL, fn_one, [fl])
        fg_not_one = ft.FeatureGrouper(
            'card_not_one', ft.FEATURE_TYPE_FLOAT_32, fa, fr, f_not_one, ft.TIME_PERIOD_DAY, 1, ft.AGGREGATOR_SUM
        )
        fg_is_one = ft.FeatureGrouper(
            'card_one', ft.FEATURE_TYPE_FLOAT_32, fa, fr, f_is_one, ft.TIME_PERIOD_DAY, 1, ft.AGGREGATOR_SUM
        )
        fg_no_filter = ft.FeatureGrouper(
            'card_no_filter', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, ft.TIME_PERIOD_DAY, 1, ft.AGGREGATOR_SUM
        )
        td = ft.TensorDefinition('Derived', [fd, fr, fc, fa, f_not_one, f_is_one, fg_not_one, fg_is_one, fg_no_filter])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td, file, inference=False, time_feature=fd)
        # We are using a 1-day window so the filtered records on (fraud == 1) added to (fraud == 2)
        # should be the same a no filter
        self.assertTrue((df['card_no_filter'].equals(df['card_not_one'] + df['card_one'])))


# Some tests to see if from_csv can figure out and build features that depend on one another
class TestBuildEmbedded(unittest.TestCase):
    def test_derived_base(self):
        mcc_id = 'MCC_ID'
        label = 'Fraud_Label'
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_STRING)
        fl = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        fi = ft.FeatureIndex(mcc_id, ft.FEATURE_TYPE_INT_16, fm)
        fl = ft.FeatureLabelBinary(label, ft.FEATURE_TYPE_INT_8, fl)
        td1 = ft.TensorDefinition('Derived_Only', [fi, fl])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
        self.assertEqual(len(df.columns), len(td1), f'Were expecting {len(td1)} columns')
        self.assertIn(mcc_id, df.columns, f'Expected a column named {mcc_id} in the DataFrame')
        self.assertIn(label, df.columns, f'Expected a column named {mcc_id} in the DataFrame')
        self.assertTrue(td1.inference_ready, f'Tensor Definition should have been ready for inference')
        self.assertEqual(td1.rank, 2, f'TensorDefinition should have Rank 2')

    def test_derived_expander(self):
        mcc = 'MCC'
        mcc_oh = 'MCC_OH'
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fm = ft.FeatureSource(mcc, ft.FEATURE_TYPE_STRING)
        fo = ft.FeatureOneHot(mcc_oh, ft.FEATURE_TYPE_INT_16, fm)
        td1 = ft.TensorDefinition('Derived_Only', [fm, fo])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            mcc_v = df[mcc].unique()
            mcc_c = [mcc] + ['MCC' + '__' + m for m in mcc_v if isinstance(m, str)]

        self.assertEqual(len(df.columns), len(mcc_c), f'Col number must match values {len(df.columns), len(mcc_c)}')
        self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')

    def test_derived_3_levels_idx_exp(self):
        mcc_id = 'MCC_ID'
        mcc_test = 'MCC_ID_1'
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fm = ft.FeatureSource('MCC', ft.FEATURE_TYPE_STRING)
        fi = ft.FeatureIndex(mcc_id, ft.FEATURE_TYPE_INT_16, fm)
        fe = ft.FeatureExpression(mcc_test, ft.FEATURE_TYPE_BOOL, lambda x: x == 1, [fi])
        td1 = ft.TensorDefinition('Derived_Expression', [fe])
        td2 = ft.TensorDefinition('Derived', [fi])
        with en.EnginePandasNumpy() as e:
            df_i = e.from_csv(td2, file, inference=False)
            df_e = e.from_csv(td1, file, inference=False)

        self.assertEqual(len(df_e.columns), len(td1), f'Expected only {len(td1)} feature')
        self.assertIn(mcc_test, df_e.columns, f'Expected a column named {mcc_test} in the DataFrame')
        self.assertTrue(td1.inference_ready, f'Tensor Definition should have been ready for inference')
        self.assertEqual(td1.rank, 2, f'TensorDefinition should have Rank 2')
        self.assertTrue((df_i[mcc_id] == 1).equals((df_e[mcc_test] == 1)), f'Dataframes not equal')

    # Test OneHot logic at various levels. These will have various iterations that need to do OH logic.
    def test_derived_3_levels_bin_oh(self):
        amt = 'Amount'
        frd = 'Fraud'
        mcc = 'MCC'
        mcc_oh = 'MCC_OH'
        frd_label = 'Fraud_lbl'
        amount_bin = 'amount_bin'
        bins = 30
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource(amt, ft.FEATURE_TYPE_FLOAT_32)
        fm = ft.FeatureSource(mcc, ft.FEATURE_TYPE_STRING)
        fl = ft.FeatureSource(frd, ft.FEATURE_TYPE_FLOAT_32)
        fab = ft.FeatureBin(amount_bin, ft.FEATURE_TYPE_INT_16, fa, bins)
        flb = ft.FeatureLabelBinary(frd_label, ft.FEATURE_TYPE_INT_8, fl)
        foh = ft.FeatureOneHot('amount_one_hot', ft.FEATURE_TYPE_INT_8, fab)
        fmo = ft.FeatureOneHot(mcc_oh, ft.FEATURE_TYPE_INT_16, fm)
        td1 = ft.TensorDefinition('Input', [fm])
        td2 = ft.TensorDefinition('Derived', [foh, fmo, flb])
        with en.EnginePandasNumpy() as e:
            df_i = e.from_csv(td1, file, inference=False)
            df_e = e.from_csv(td2, file, inference=False)
            mcc_v = df_i[mcc].dropna().unique()
            mcc_c = [f'{mcc}__{c}'for c in mcc_v.tolist()]
            amt_b = [f'{amount_bin}__{i}' for i in range(bins)]

        self.assertEqual(len(df_i), len(df_e), f'Weird, lengths not equal {len(df_i)}, {len(df_e)}')
        self.assertEqual(len(df_e.columns), bins + 1 + len(mcc_v), f'Expected only {bins + 1 + len(mcc_v)} feature')
        self.assertIn(frd_label, df_e.columns, f'Expected {frd_label} to be in the list of columns')
        self.assertTrue(set(mcc_c).issubset(set(df_e.columns)), f'Did not find columns {mcc_c}')
        self.assertTrue(set(amt_b).issubset(set(df_e.columns)), f'Did not find columns {amt_b}')


class TestSeriesFrequencies(unittest.TestCase):
    # Not good if not all features have the same type.
    def test_not_all_same_type(self):
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        tp = ft.TIME_PERIOD_DAY
        freq = 3
        td1 = ft.TensorDefinition('Base', [fd, fr, fa])
        fg_1 = ft.FeatureGrouper(
            'card_2d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM
        )
        # This is a float64, the previous a float32
        fg_2 = ft.FeatureGrouper(
            'card_2d_count', ft.FEATURE_TYPE_FLOAT, fa, fr, None, tp, freq, ft.AGGREGATOR_COUNT
        )
        td2 = ft.TensorDefinition('Frequencies', [fg_1, fg_2])
        with en.EnginePandasNumpy() as e:
            with self.assertRaises(en.EnginePandaNumpyException):
                _ = e.to_series_frequencies(td2, file, fd, inference=False)

    # Run some base day tests
    def test_create_base_day(self):
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        tp = ft.TIME_PERIOD_DAY
        freq = 3
        td1 = ft.TensorDefinition('Base', [fd, fr, fa])
        fg_1 = ft.FeatureGrouper(
            'card_2d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM
        )
        fg_2 = ft.FeatureGrouper(
            'card_2d_count', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_COUNT
        )
        td2 = ft.TensorDefinition('Frequencies', [fg_1, fg_2])
        # With 2 features this will create a single numpy with size (Batch x freq x 2)
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            self.assertEqual(type(n), en.NumpyList, f'expected a return of Numpy List. Got {type(n)} ')
            self.assertEqual(len(n.lists), 1, f'Expected the NumpyList to contain a single array')
            self.assertEqual(type(n.lists[0]), np.ndarray, f'Expected a numpy array return. Got {type(n.lists)}')
            self.assertEqual(n.lists[0].shape[0], len(df), f'The same number of lines as the input records')
            self.assertEqual(n.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(n.lists[0].shape[2], len(td2.features), f'Expected 3rd dim to be the number of features.')
            amount_sum_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            amount_cnt_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            date_dict = {c: min(df[df['Card'] == c]['Date']).to_pydatetime() for c in df['Card'].unique()}
            for i in range(len(df)):
                d = tp.delta_between(date_dict[df.iloc[i]['Card']], df.iloc[i]['Date'].to_pydatetime())
                # Shift numbers up by delta
                s = amount_sum_dict.get(df.iloc[i]['Card'])
                c = amount_cnt_dict.get(df.iloc[i]['Card'])
                s.extend([0 for _ in range(d)])
                c.extend([0 for _ in range(d)])
                date_dict[df.iloc[i]['Card']] = df.iloc[i]['Date'].to_pydatetime()
                # Add and count
                s[freq - 1] = s[freq - 1] + float(df.iloc[i]['Amount'])
                c[freq - 1] = c[freq - 1] + 1
                self.assertTrue(np.array_equal(np.array(s), n.lists[0][i, :, 1]), f'Sums not same {i}')
                self.assertTrue(np.array_equal(np.array(c), n.lists[0][i, :, 0]), f'Counts not same {i}')

    # Run some base week tests
    def test_create_base_week(self):
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        tp = ft.TIME_PERIOD_WEEK
        freq = 3
        td1 = ft.TensorDefinition('Base', [fd, fr, fa])
        fg_1 = ft.FeatureGrouper(
            'card_2d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM
        )
        fg_2 = ft.FeatureGrouper(
            'card_2d_count', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_COUNT
        )
        td2 = ft.TensorDefinition('Frequencies', [fg_1, fg_2])
        # With 2 features this will create a single numpy with size (Batch x freq x 2)
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            self.assertEqual(type(n), en.NumpyList, f'expected a return of Numpy List. Got {type(n)} ')
            self.assertEqual(len(n.lists), 1, f'Expected the NumpyList to contain a single array')
            self.assertEqual(type(n.lists[0]), np.ndarray, f'Expected a numpy array return. Got {type(n.lists)}')
            self.assertEqual(n.lists[0].shape[0], len(df), f'The same number of lines as the input records')
            self.assertEqual(n.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(n.lists[0].shape[2], len(td2.features), f'Expected 3rd dim to be the number of features.')
            amount_sum_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            amount_cnt_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            date_dict = {c: min(df[df['Card'] == c]['Date']).to_pydatetime() for c in df['Card'].unique()}
            for i in range(len(df)):
                s = amount_sum_dict.get(df.iloc[i]['Card'])
                c = amount_cnt_dict.get(df.iloc[i]['Card'])
                d1 = date_dict[df.iloc[i]['Card']]
                d2 = df.iloc[i]['Date'].to_pydatetime()
                d = tp.delta_between(tp.start_period(d1), tp.start_period(d2))
                # Shift numbers up by delta
                if d > 0:
                    s.extend([0 for _ in range(d)])
                    c.extend([0 for _ in range(d)])
                    date_dict[df.iloc[i]['Card']] = df.iloc[i]['Date'].to_pydatetime()
                # Add and count
                s[freq - 1] = s[freq - 1] + float(df.iloc[i]['Amount'])
                c[freq - 1] = c[freq - 1] + 1
                self.assertTrue(np.array_equal(np.array(s), n.lists[0][i, :, 1]), f'Sums not same {i}')
                self.assertTrue(np.array_equal(np.array(c), n.lists[0][i, :, 0]), f'Counts not same {i}')

    # Run some base month tests
    def test_create_base_month(self):
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        tp = ft.TIME_PERIOD_MONTH
        freq = 3
        td1 = ft.TensorDefinition('Base', [fd, fr, fa])
        fg_1 = ft.FeatureGrouper(
            'card_2d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM
        )
        fg_2 = ft.FeatureGrouper(
            'card_2d_count', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_COUNT
        )
        td2 = ft.TensorDefinition('Frequencies', [fg_1, fg_2])
        # With 2 features this will create a single numpy with size (Batch x freq x 2)
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            self.assertEqual(type(n), en.NumpyList, f'expected a return of Numpy List. Got {type(n)} ')
            self.assertEqual(len(n.lists), 1, f'Expected the NumpyList to contain a single array')
            self.assertEqual(type(n.lists[0]), np.ndarray, f'Expected a numpy array return. Got {type(n.lists)}')
            self.assertEqual(n.lists[0].shape[0], len(df), f'The same number of lines as the input records')
            self.assertEqual(n.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(n.lists[0].shape[2], len(td2.features), f'Expected 3rd dim to be the number of features.')
            amount_sum_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            amount_cnt_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            date_dict = {c: min(df[df['Card'] == c]['Date']).to_pydatetime() for c in df['Card'].unique()}
            for i in range(len(df)):
                s = amount_sum_dict.get(df.iloc[i]['Card'])
                c = amount_cnt_dict.get(df.iloc[i]['Card'])
                d1 = date_dict[df.iloc[i]['Card']]
                d2 = df.iloc[i]['Date'].to_pydatetime()
                d = tp.delta_between(tp.start_period(d1), tp.start_period(d2))
                # Shift numbers up by delta
                if d > 0:
                    s.extend([0 for _ in range(d)])
                    c.extend([0 for _ in range(d)])
                    date_dict[df.iloc[i]['Card']] = df.iloc[i]['Date'].to_pydatetime()
                # Add and count
                s[freq - 1] = s[freq - 1] + float(df.iloc[i]['Amount'])
                c[freq - 1] = c[freq - 1] + 1
                self.assertTrue(np.array_equal(np.array(s), n.lists[0][i, :, 1]), f'Sums not same {i}')
                self.assertTrue(np.array_equal(np.array(c), n.lists[0][i, :, 0]), f'Counts not same {i}')

    # Combine various time-periods
    def test_create_base_multiple_tp(self):
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        tp_day = ft.TIME_PERIOD_DAY
        tp_week = ft.TIME_PERIOD_WEEK
        tp_month = ft.TIME_PERIOD_MONTH
        freq = 3
        td1 = ft.TensorDefinition('Base', [fd, fr, fa])
        fg_1 = ft.FeatureGrouper(
            'card_3d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp_day, freq, ft.AGGREGATOR_SUM
        )
        fg_2 = ft.FeatureGrouper(
            'card_3w_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp_week, freq, ft.AGGREGATOR_SUM
        )
        fg_3 = ft.FeatureGrouper(
            'card_3m_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp_month, freq, ft.AGGREGATOR_SUM
        )
        td2 = ft.TensorDefinition('Frequencies', [fg_1, fg_2, fg_3])
        # With 2 features this will create a single numpy with size (Batch x freq x 2)
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            self.assertEqual(type(n), en.NumpyList, f'expected a return of Numpy List. Got {type(n)} ')
            self.assertEqual(len(n.lists), 3, f'Expected the NumpyList to contain 3 arrays')
            self.assertEqual(type(n.lists[0]), np.ndarray, f'Expected a list 0 numpy array return. Got {type(n.lists)}')
            self.assertEqual(type(n.lists[1]), np.ndarray, f'Expected a list 1 numpy array return. Got {type(n.lists)}')
            self.assertEqual(type(n.lists[2]), np.ndarray, f'Expected a list 2 numpy array return. Got {type(n.lists)}')
            self.assertEqual(n.lists[0].shape[0], len(df), f'The same number of lines as the input records. List 0')
            self.assertEqual(n.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies. List 0')
            self.assertEqual(n.lists[0].shape[2], 1, f'Expected 3rd dim to be 1. List 0')
            self.assertEqual(n.lists[1].shape[0], len(df), f'The same number of lines as the input records. List 1')
            self.assertEqual(n.lists[1].shape[1], freq, f'Expected 2nd dim to be the number of frequencies. List 1')
            self.assertEqual(n.lists[1].shape[2], 1, f'Expected 3rd dim to be 1. List 1')
            self.assertEqual(n.lists[2].shape[0], len(df), f'The same number of lines as the input records. List 2')
            self.assertEqual(n.lists[2].shape[1], freq, f'Expected 2nd dim to be the number of frequencies. List 2')
            self.assertEqual(n.lists[2].shape[2], 1, f'Expected 3rd dim to be 1. List 2')
            amount_d_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            amount_w_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            amount_m_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            date_d_dict = {c: min(df[df['Card'] == c]['Date']).to_pydatetime() for c in df['Card'].unique()}
            date_w_dict = {c: min(df[df['Card'] == c]['Date']).to_pydatetime() for c in df['Card'].unique()}
            date_m_dict = {c: min(df[df['Card'] == c]['Date']).to_pydatetime() for c in df['Card'].unique()}
            for i in range(len(df)):
                for ad, dd, tp, j in zip([amount_d_dict, amount_w_dict, amount_m_dict],
                                         [date_d_dict, date_w_dict, date_m_dict],
                                         [tp_day, tp_week, tp_month],
                                         [0, 1, 2]):
                    s = ad.get(df.iloc[i]['Card'])
                    d1 = dd[df.iloc[i]['Card']]
                    d2 = df.iloc[i]['Date'].to_pydatetime()
                    d = tp.delta_between(tp.start_period(d1), tp.start_period(d2))
                    # Shift numbers up by delta
                    if d > 0:
                        s.extend([0 for _ in range(d)])
                        dd[df.iloc[i]['Card']] = df.iloc[i]['Date'].to_pydatetime()
                    # Add and count
                    s[freq - 1] = s[freq - 1] + float(df.iloc[i]['Amount'])
                    self.assertTrue(np.array_equal(np.array(s), n.lists[j][i, :, 0]), f'Sums not same {i} {j}')

    def test_create_filter(self):
        # Base test. See if the filtering works.
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        freq = 5
        tp = ft.TIME_PERIOD_DAY
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        f_is_one = ft.FeatureFilter('Is_Fraud', ft.FEATURE_TYPE_BOOL, fn_one, [ff])
        td1 = ft.TensorDefinition('Source', [fd, fr, fa, ff, f_is_one])
        fg_1 = ft.FeatureGrouper('crd_5d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM)
        fg_2 = ft.FeatureGrouper('flt_5d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, f_is_one, tp, freq, ft.AGGREGATOR_SUM)
        td2 = ft.TensorDefinition('Frequencies', [fg_1, fg_2])
        # With 2 features this will create 1 numpy arrays with size (Batch x freq x 2)
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            self.assertEqual(type(n), en.NumpyList, f'expected a return of Numpy List. Got {type(n)} ')
            self.assertEqual(len(n.lists), 1, f'Expected the NumpyList to contain 1 arrays')
            self.assertEqual(type(n.lists[0]), np.ndarray, f'Expected a list 0 numpy array return. Got {type(n.lists)}')
            self.assertEqual(n.lists[0].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(n.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(n.lists[0].shape[2], 2, f'Expected 3rd dim to be 2.')
            amount_flt_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            amount_n_flt_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            date_flt_dict = {
                c: min(df[(df['Card'] == c) & (df['Fraud'] == 1)]['Date']).to_pydatetime() for c in df['Card'].unique()
            }
            date_n_flt_dict = {
                c: min(df[df['Card'] == c]['Date']).to_pydatetime() for c in df['Card'].unique()
            }
            for i in range(len(df)):
                for ad, dd, ff, j in zip([amount_n_flt_dict, amount_flt_dict],
                                         [date_n_flt_dict, date_flt_dict],
                                         [False, True],
                                         [0, 1]):
                    s = ad.get(df.iloc[i]['Card'])
                    d1 = dd[df.iloc[i]['Card']]
                    d2 = df.iloc[i]['Date'].to_pydatetime()
                    d = tp.delta_between(tp.start_period(d1), tp.start_period(d2))
                    if d > 0:
                        # Shift numbers up by delta
                        s.extend([0 for _ in range(d)])
                        dd[df.iloc[i]['Card']] = df.iloc[i]['Date'].to_pydatetime()
                    # Only actually add if for the non-filtered feature or if the filter is True
                    f = df.iloc[i]['Fraud']
                    if not ff or (ff and f == 1):
                        s[freq - 1] = s[freq - 1] + float(df.iloc[i]['Amount'])
                    self.assertTrue(np.array_equal(np.array(s), n.lists[0][i, :, j]), f'Sums not same {i} {j}')

    def test_grouped_multiple_groups(self):
        # Test if we can have multiple group features. Basically have multiple entities.
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        freq = 5
        tp = ft.TIME_PERIOD_DAY
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fc, fa, ff])
        fg_1 = ft.FeatureGrouper('crd_5d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM)
        fg_2 = ft.FeatureGrouper('cty_5d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fc, None, tp, freq, ft.AGGREGATOR_SUM)
        td2 = ft.TensorDefinition('Frequencies', [fg_1, fg_2])
        # With 2 features this will create 1 numpy arrays with size (Batch x freq x 2). One for each group
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            self.assertEqual(type(n), en.NumpyList, f'expected a return of Numpy List. Got {type(n)} ')
            self.assertEqual(len(n.lists), 1, f'Expected the NumpyList to contain 1 arrays')
            self.assertEqual(type(n.lists[0]), np.ndarray, f'Expected a list 0 numpy array return. Got {type(n.lists)}')
            self.assertEqual(n.lists[0].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(n.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(n.lists[0].shape[2], 2, f'Expected 3rd dim to be 2.')
            amount_cty_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Country'].unique()}
            amount_crd_dict = {c: deque([0.0 for _ in range(freq)], maxlen=freq) for c in df['Card'].unique()}
            date_cty_dict = {c: min(df[df['Country'] == c]['Date']).to_pydatetime() for c in df['Country'].unique()}
            date_crd_dict = {c: min(df[df['Card'] == c]['Date']).to_pydatetime() for c in df['Card'].unique()}
            for i in range(len(df)):
                for ad, dd, key, j in zip([amount_cty_dict, amount_crd_dict],
                                          [date_cty_dict, date_crd_dict],
                                          ['Country', 'Card'],
                                          [1, 0]):
                    s = ad.get(df.iloc[i][key])
                    d1 = dd[df.iloc[i][key]]
                    d2 = df.iloc[i]['Date'].to_pydatetime()
                    d = tp.delta_between(tp.start_period(d1), tp.start_period(d2))
                    # Shift numbers up by delta
                    if d > 0:
                        s.extend([0 for _ in range(d)])
                        dd[df.iloc[i][key]] = df.iloc[i]['Date'].to_pydatetime()
                    # Add and count
                    s[freq - 1] = s[freq - 1] + float(df.iloc[i]['Amount'])
                    self.assertTrue(np.array_equal(np.array(s), n.lists[0][i, :, j]), f'Sums not same {i} {j}')

    def test_normalizer_scale(self):
        # Test Frequency Normalization.
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        freq = 5
        tp = ft.TIME_PERIOD_DAY
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fa])
        fg_1 = ft.FeatureGrouper('crd_5d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM)
        fg_2 = ft.FeatureNormalizeScale('crd_5d_sum_n', ft.FEATURE_TYPE_FLOAT_32, fg_1)
        td2 = ft.TensorDefinition('Frequencies', [fg_1])
        td3 = ft.TensorDefinition('FrequenciesNormalized', [fg_2])
        # With 2 features this will create 1 numpy arrays with size (Batch x freq x 1). One for each group
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            mn = np.min(n.lists[0], axis=(0, 1))
            mx = np.max(n.lists[0], axis=(0, 1))
            self.assertFalse(td3.inference_ready, f'Should not be inference ready before series build')
            nn = e.to_series_frequencies(td3, file, fd, inference=False)
            self.assertTrue(td3.inference_ready, f'Should have been inference ready after series build')
            self.assertEqual(type(nn), en.NumpyList, f'expected a return of Numpy List. Got {type(nn)} ')
            self.assertEqual(len(nn.lists), 1, f'Expected the NumpyList to contain 1 arrays')
            self.assertEqual(type(nn.lists[0]), np.ndarray, f'Expected list 0 numpy array return. Got {type(nn.lists)}')
            self.assertEqual(nn.lists[0].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(nn.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(nn.lists[0].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(fg_2.minimum, mn.item(), f'Minimum not set correctly {fg_2.minimum} {mn.item()}')
            self.assertEqual(fg_2.maximum, mx.item(), f'Maximum not set correctly {fg_2.maximum} {mx.item()}')
            nr = (n.lists[0] - mn.item()) / (mx.item() - mn.item())
            self.assertTrue(np.array_equal(nn.lists[0], nr), f'Calculated array should be same as frequency output')

    def test_normalizer_scale_log(self):
        # Test Frequency Normalization.
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        freq = 5
        s_name = 'scale_log'
        s_type = ft.FEATURE_TYPE_FLOAT
        tp = ft.TIME_PERIOD_DAY
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fg_1 = ft.FeatureGrouper('crd_5d_sum', s_type, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM)
        for log, log_fn in (('e', np.log), ('2', np.log2), ('10', np.log10)):
            fs = ft.FeatureNormalizeScale(s_name, s_type, fg_1, log)
            with en.EnginePandasNumpy() as e:
                td_df = ft.TensorDefinition('All', [fg_1])
                na = e.to_series_frequencies(td_df, file, fd, inference=False)
                self.assertEqual(len(na.lists), 1, f'Expected to get 1 list, got {len(na.lists)}')
                mn = log_fn(na.lists[0]+fs.delta).min(axis=(0, 1))
                mx = log_fn(na.lists[0]+fs.delta).max(axis=(0, 1))
                td2 = ft.TensorDefinition('Derived', [fs])
                nas = e.to_series_frequencies(td2, file, fd, inference=False)
                self.assertTrue(td2.inference_ready, f'Should have been inference ready after series build')
                self.assertEqual(type(nas), en.NumpyList, f'expected a return of Numpy List. Got {type(nas)} ')
                self.assertEqual(len(nas.lists), 1, f'Expected the NumpyList to contain 1 arrays')
                self.assertEqual(type(nas.lists[0]), np.ndarray,
                                 f'Expected list 0 numpy array return. Got {type(nas.lists)}')
                self.assertEqual(nas.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
                self.assertEqual(nas.lists[0].shape[2], 1, f'Expected 3rd dim to be 1.')
                self.assertEqual(fs.minimum, mn.item(), f'Minimum not set correctly {fs.minimum} {mn.item()}')
                self.assertEqual(fs.maximum, mx.item(), f'Maximum not set correctly {fs.maximum} {mx.item()}')
                nr = (log_fn(na.lists[0]+fs.delta) - mn.item()) / (mx.item() - mn.item())
                self.assertTrue(np.array_equal(nas.lists[0], nr), f'Calculated should be same as frequency output')

    def test_normalizer_standard(self):
        # Test Frequency Normalization.
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        freq = 5
        tp = ft.TIME_PERIOD_DAY
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource('Country', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        ff = ft.FeatureSource('Fraud', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fc, fa, ff])
        fg_1 = ft.FeatureGrouper('crd_5d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM)
        fg_2 = ft.FeatureNormalizeStandard('crd_5d_sum_n', ft.FEATURE_TYPE_FLOAT_32, fg_1)
        td2 = ft.TensorDefinition('Frequencies', [fg_1])
        td3 = ft.TensorDefinition('FrequenciesNormalized', [fg_2])
        # With 2 features this will create 1 numpy arrays with size (Batch x freq x 1). One for each group
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            mn = np.mean(n.lists[0], axis=(0, 1))
            st = np.std(n.lists[0], axis=(0, 1))
            self.assertFalse(td3.inference_ready, f'Should not be inference ready before series build')
            nn = e.to_series_frequencies(td3, file, fd, inference=False)
            self.assertTrue(td3.inference_ready, f'Should have been inference ready after series build')
            self.assertEqual(type(nn), en.NumpyList, f'expected a return of Numpy List. Got {type(nn)} ')
            self.assertEqual(len(nn.lists), 1, f'Expected the NumpyList to contain 1 arrays')
            self.assertEqual(type(nn.lists[0]), np.ndarray, f'Expected list 0 numpy array return. Got {type(nn.lists)}')
            self.assertEqual(nn.lists[0].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(nn.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(nn.lists[0].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(fg_2.mean, mn.item(), f'Minimum not set correctly {fg_2.mean} {mn.item()}')
            self.assertEqual(fg_2.stddev, st.item(), f'Maximum not set correctly {fg_2.stddev} {st.item()}')
            nr = (n.lists[0] - mn.item()) / st.item()
            self.assertTrue(np.array_equal(nn.lists[0], nr), f'Calculated array should be same as frequency output')

    def test_normalizer_standard_log(self):
        # Test Frequency Normalization.
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        freq = 5
        s_name = 'standard_log'
        s_type = ft.FEATURE_TYPE_FLOAT
        tp = ft.TIME_PERIOD_DAY
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fg_1 = ft.FeatureGrouper('crd_5d_sum', s_type, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM)
        for log, log_fn in (('e', np.log), ('2', np.log2), ('10', np.log10)):
            fs = ft.FeatureNormalizeStandard(s_name, s_type, fg_1, log)
            with en.EnginePandasNumpy() as e:
                td_df = ft.TensorDefinition('All', [fg_1])
                na = e.to_series_frequencies(td_df, file, fd, inference=False)
                self.assertEqual(len(na.lists), 1, f'Expected to get 1 list, got {len(na.lists)}')
                mn = log_fn(na.lists[0]+fs.delta).mean(axis=(0, 1))
                st = log_fn(na.lists[0]+fs.delta).std(axis=(0, 1))
                td2 = ft.TensorDefinition('Derived', [fs])
                nas = e.to_series_frequencies(td2, file, fd, inference=False)
                self.assertTrue(td2.inference_ready, f'Should have been inference ready after series build')
                self.assertEqual(type(nas), en.NumpyList, f'expected a return of Numpy List. Got {type(nas)} ')
                self.assertEqual(len(nas.lists), 1, f'Expected the NumpyList to contain 1 arrays')
                self.assertEqual(type(nas.lists[0]), np.ndarray,
                                 f'Expected list 0 numpy array return. Got {type(nas.lists)}')
                self.assertEqual(nas.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
                self.assertEqual(nas.lists[0].shape[2], 1, f'Expected 3rd dim to be 1.')
                self.assertEqual(fs.mean, mn.item(), f'Mean not set correctly {fs.mean} {mn.item()}')
                self.assertEqual(fs.stddev, st.item(), f'Stddev not set correctly {fs.stddev} {st.item()}')
                nr = (log_fn(na.lists[0]+fs.delta) - mn.item()) / st.item()
                self.assertTrue(np.array_equal(nas.lists[0], nr), f'Calculated should be same as frequency output')

    def test_normalizer_combine_scale_standard(self):
        # Test Frequency Normalization. See if a combination of Standard and Scaling works.
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        freq = 5
        tp = ft.TIME_PERIOD_DAY
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fa])
        fg_1 = ft.FeatureGrouper('crd_5d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq, ft.AGGREGATOR_SUM)
        fg_2 = ft.FeatureNormalizeStandard('crd_5d_sum_st', ft.FEATURE_TYPE_FLOAT_32, fg_1)
        fg_3 = ft.FeatureNormalizeScale('crd_5d_sum_sc', ft.FEATURE_TYPE_FLOAT_32, fg_1)
        td2 = ft.TensorDefinition('FrequenciesBase', [fg_1])
        td3 = ft.TensorDefinition('FrequenciesNormalizedStandard', [fg_2])
        td4 = ft.TensorDefinition('FrequenciesNormalizedScale', [fg_3])
        td5 = ft.TensorDefinition('FrequenciesAll', [fg_1, fg_2, fg_3])
        # With 1 features this will create 1 numpy arrays with size (Batch x freq x 1). One for each group
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            self.assertFalse(td3.inference_ready, f'Should not be inference ready before series build')
            nst = e.to_series_frequencies(td3, file, fd, inference=False)
            nsc = e.to_series_frequencies(td4, file, fd, inference=False)
            nsa = e.to_series_frequencies(td5, file, fd, inference=False)
            self.assertTrue(td5.inference_ready, f'Should have been inference ready after series build')
            self.assertEqual(type(nsa), en.NumpyList, f'expected a return of Numpy List. Got {type(nsa)} ')
            self.assertEqual(len(nsa.lists), 1, f'Expected the NumpyList to contain 1 arrays')
            self.assertEqual(type(nsa.lists[0]), np.ndarray, f'Expected list 0 array return. Got {type(nsa.lists)}')
            self.assertEqual(nsa.lists[0].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(nsa.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(nsa.lists[0].shape[2], 3, f'Expected 3rd dim to be 3.')
            # First Feature of the combined list should have been the same as the base list
            self.assertTrue(np.array_equal(np.squeeze(n.lists[0]), nsa.lists[0][:, :, 0]), f'Base Feature not the same')
            # Second Feature of the combined list should have been the same as the Standard list
            self.assertTrue(np.array_equal(np.squeeze(nst.lists[0]), nsa.lists[0][:, :, 1]), f'Standard not the same')
            # Third Feature of the combined list should have been the same as the Scale list
            self.assertTrue(np.array_equal(np.squeeze(nsc.lists[0]), nsa.lists[0][:, :, 2]), f'Scale not the same')

    def test_normalizer_multiple_time_windows(self):
        # Test Frequency Normalization. See if a combination of Standard and Scaling works.
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        freq1 = 5
        freq2 = 10
        tp = ft.TIME_PERIOD_DAY
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fa])
        fg_1 = ft.FeatureGrouper('crd_10d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq2, ft.AGGREGATOR_SUM)
        fg_2 = ft.FeatureGrouper('crd_5d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp, freq1, ft.AGGREGATOR_SUM)
        fg_3 = ft.FeatureNormalizeScale('crd_10d_sum_sc', ft.FEATURE_TYPE_FLOAT_32, fg_1)
        fg_4 = ft.FeatureNormalizeStandard('crd_5d_sum_st', ft.FEATURE_TYPE_FLOAT_32, fg_2)
        td2 = ft.TensorDefinition('FrequenciesBase', [fg_1, fg_2])
        td3 = ft.TensorDefinition('FrequenciesNormalizedStandard', [fg_3, fg_4])
        # With 2 features this will create 2 numpy arrays with size (Batch x freq x 1). One for each time period
        # Note that in the Frequency the output is ordered. So fg_2, the 5 time period will come first
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            nn = e.to_series_frequencies(td3, file, fd, inference=False)
            mn5 = np.mean(n.lists[0], axis=(0, 1))
            st5 = np.std(n.lists[0], axis=(0, 1))
            mn10 = np.min(n.lists[1], axis=(0, 1))
            mx10 = np.max(n.lists[1], axis=(0, 1))
            self.assertEqual(type(n), en.NumpyList, f'expected a return of Numpy List. Got {type(n)} ')
            self.assertEqual(len(n.lists), 2, f'Expected the NumpyList to contain 2 arrays')
            self.assertEqual(type(n.lists[0]), np.ndarray, f'Expected list 0 numpy array return. Got {type(n.lists)}')
            self.assertEqual(n.lists[0].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(n.lists[0].shape[1], freq1, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(n.lists[0].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(type(n.lists[1]), np.ndarray, f'Expected list 0 numpy array return. Got {type(n.lists)}')
            self.assertEqual(n.lists[1].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(n.lists[1].shape[1], freq2, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(n.lists[1].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(type(nn), en.NumpyList, f'expected a return of Numpy List. Got {type(nn)} ')
            self.assertEqual(len(nn.lists), 2, f'Expected the NumpyList to contain 2 arrays')
            self.assertEqual(type(nn.lists[0]), np.ndarray, f'Expected list 0 numpy array return. Got {type(nn.lists)}')
            self.assertEqual(nn.lists[0].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(nn.lists[0].shape[1], freq1, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(nn.lists[0].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(type(nn.lists[1]), np.ndarray, f'Expected list 0 numpy array return. Got {type(nn.lists)}')
            self.assertEqual(nn.lists[1].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(nn.lists[1].shape[1], freq2, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(nn.lists[1].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(fg_4.mean, mn5.item(), f'Minimum not set correctly {fg_4.mean} {mn5.item()}')
            self.assertEqual(fg_4.stddev, st5.item(), f'Maximum not set correctly {fg_4.stddev} {st5.item()}')
            nr = (n.lists[0] - mn5.item()) / st5.item()
            self.assertTrue(np.array_equal(nn.lists[0], nr), f'Calculated array should be same as frequency output')
            self.assertEqual(fg_3.minimum, mn10.item(), f'Minimum not set correctly {fg_3.minimum} {mn10.item()}')
            self.assertEqual(fg_3.maximum, mx10.item(), f'Maximum not set correctly {fg_3.maximum} {mx10.item()}')
            nr = (n.lists[1] - mn10.item()) / (mx10.item() - mn10.item())
            self.assertTrue(np.array_equal(nn.lists[1], nr), f'Calculated array should be same as frequency output')

    def test_normalizer_multiple_time_periods(self):
        # Test Frequency Normalization. See if multiple time periods work
        file = FILES_DIR + 'engine_test_freq_comma.csv'
        freq = 5
        tp2 = ft.TIME_PERIOD_WEEK
        tp1 = ft.TIME_PERIOD_DAY
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        fr = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        td1 = ft.TensorDefinition('Source', [fd, fr, fa])
        fg_1 = ft.FeatureGrouper('crd_5w_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp2, freq, ft.AGGREGATOR_SUM)
        fg_2 = ft.FeatureGrouper('crd_5d_sum', ft.FEATURE_TYPE_FLOAT_32, fa, fr, None, tp1, freq, ft.AGGREGATOR_SUM)
        fg_3 = ft.FeatureNormalizeScale('crd_5w_sum_sc', ft.FEATURE_TYPE_FLOAT_32, fg_1)
        fg_4 = ft.FeatureNormalizeStandard('crd_5d_sum_st', ft.FEATURE_TYPE_FLOAT_32, fg_2)
        td2 = ft.TensorDefinition('FrequenciesBase', [fg_1, fg_2])
        td3 = ft.TensorDefinition('FrequenciesNormalized', [fg_3, fg_4])
        # With 2 features this will create 2 numpy array with size (Batch x freq x 1). One for each time period
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            n = e.to_series_frequencies(td2, file, fd, inference=False)
            nn = e.to_series_frequencies(td3, file, fd, inference=False)
            mnd = np.mean(n.lists[0], axis=(0, 1))
            std = np.std(n.lists[0], axis=(0, 1))
            mnw = np.min(n.lists[1], axis=(0, 1))
            mxw = np.max(n.lists[1], axis=(0, 1))
            self.assertEqual(type(n), en.NumpyList, f'expected a return of Numpy List. Got {type(n)} ')
            self.assertEqual(len(n.lists), 2, f'Expected the NumpyList to contain 2 arrays')
            self.assertEqual(type(n.lists[0]), np.ndarray, f'Expected list 0 numpy array return. Got {type(n.lists)}')
            self.assertEqual(n.lists[0].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(n.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(n.lists[0].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(n.lists[1].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(n.lists[1].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(n.lists[1].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(type(nn), en.NumpyList, f'expected a return of Numpy List. Got {type(nn)} ')
            self.assertEqual(len(nn.lists), 2, f'Expected the NumpyList to contain 2 array')
            self.assertEqual(type(nn.lists[0]), np.ndarray, f'Expected list 0 numpy array return. Got {type(nn.lists)}')
            self.assertEqual(nn.lists[0].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(nn.lists[0].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(nn.lists[0].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(nn.lists[1].shape[0], len(df), f'The same number of lines as the input records.')
            self.assertEqual(nn.lists[1].shape[1], freq, f'Expected 2nd dim to be the number of frequencies.')
            self.assertEqual(nn.lists[1].shape[2], 1, f'Expected 3rd dim to be 1.')
            self.assertEqual(fg_4.mean, mnd.item(), f'Minimum not set correctly {fg_4.mean} {mnd.item()}')
            self.assertEqual(fg_4.stddev, std.item(), f'Maximum not set correctly {fg_4.stddev} {std.item()}')
            nr = (n.lists[0] - mnd.item()) / std.item()
            self.assertTrue(np.array_equal(nn.lists[0], nr), f'Calculated array should equal to frequency')
            self.assertEqual(fg_3.minimum, mnw.item(), f'Minimum not set correctly {fg_3.minimum} {mnw.item()}')
            self.assertEqual(fg_3.maximum, mxw.item(), f'Maximum not set correctly {fg_3.maximum} {mxw.item()}')
            nr = (n.lists[1] - mnw.item()) / (mxw.item() - mnw.item())
            self.assertTrue(np.array_equal(nn.lists[1], nr), f'Calculated array should equal to frequency')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
