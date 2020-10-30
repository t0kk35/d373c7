"""
Unit Tests for PandasNumpy Engine
(c) 2020 d373c7
"""
import unittest
import d373c7.features as ft
import d373c7.engines as en

FILES_DIR = './files/'


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
                df = e.from_csv(ft.TensorDefinition(d, [f]), file, inference=False)
                self.assertEqual(len(df.columns), 1, f'Expected a one column panda for read test {d}')
                self.assertEqual(df.columns[0], f.name, f'Wrong panda column for read test {d}. Got {df.columns[0]}')

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
    """"Derived Features. One Hot feature tests. We'll use the from_df function for this
    """
    def test_read_base_non_inference(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            td = ft.TensorDefinition('All', [fc])
            df = e.from_csv(td, file, inference=False)
            mcc_v = df['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            td2 = ft.TensorDefinition('Derived', [fo])
            df = e.from_df(td2, df, inference=False)
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
        fo = ft.FeatureOneHot('MCC_OH', fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [fd]), file, inference=False)
            with self.assertRaises(en.EnginePandaNumpyException):
                e.from_df(ft.TensorDefinition('Derived', [fo]), df, inference=False)

    def test_read_base_inference_removed_element(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df_c = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            mcc_v = df_c['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            # Set derived Feature
            td = ft.TensorDefinition('Derived', [fo])
            e.from_df(td, df_c, inference=False)
            # Now Remove a line from the original csv panda and do an inference run.
            df_c = df_c.iloc[:-1]
            df = e.from_df(td, df_c, inference=True)
            self.assertEqual(len(df.columns), len(mcc_v), f'Col number must match values {len(df.columns), len(mcc_v)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')

    def test_read_base_inference_added_element(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        fo = ft.FeatureOneHot('MCC_OH', fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df_c = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            # Remove row and set variables. Also remove from categories!
            df_r = df_c.iloc[1:].copy()
            df_r['MCC'] = df_c['MCC'].cat.remove_categories(df_c['MCC'][0])
            mcc_v = df_r['MCC'].unique()
            mcc_c = ['MCC' + '__' + m for m in mcc_v]
            # Run non-inference on removed panda
            td = ft.TensorDefinition('Derived', [fo])
            e.from_df(td, df_r, inference=False)
            # And inference on original
            df = e.from_df(td, df_c, inference=True)
            self.assertEqual(len(df.columns), len(mcc_v), f'Col number must match values {len(df.columns), len(mcc_v)}')
            self.assertListEqual(list(df.columns), mcc_c, f'Names of columns must match values {df.columns}')
            self.assertEqual(fo.inference_ready, True, f'One Hot feature should be ready for inference')
            self.assertListEqual(fo.expand_names, mcc_c, f'Expanded names should match the column names')


class TestNormalizeScale(unittest.TestCase):
    """"Derived Features. Normalize Scale feature tests. We'll use the from_df function for this
    """
    def test_read_base_non_inference(self):
        fc = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountScale'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeScale(s_name, s_type, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            mn = df['Amount'].min()
            mx = df['Amount'].max()
            td2 = ft.TensorDefinition('Derived', [fs])
            df = e.from_df(td2, df, inference=False)
            self.assertEqual(len(df.columns), 1, f'Only one columns should have been returned')
            self.assertEqual(df.columns[0], s_name, f'Column name is not correct {df.columns[0]}')
            self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
            self.assertEqual(fs.maximum, mx, f'Maximum not set correctly {fs.maximum}')
            self.assertEqual(fs.minimum, mn, f'Minimum not set correctly {fs.maximum}')
            self.assertListEqual(td2.continuous_features(True), [fs], f'Expanded Feature not correct')

    def test_read_base_inference(self):
        fc = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountScale'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeScale(s_name, s_type, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            td = ft.TensorDefinition('Derived', [fs])
            df_1 = e.from_df(td, df, inference=False)
            # Now remove a line and run in inference mode
            df = df.iloc[:-1]
            df_1 = df_1.iloc[:-1]
            df_2 = e.from_df(td, df, inference=True)
            self.assertEqual(df_1[s_name].equals(df_2[s_name]), True, f'Inference problem. Series not the same')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')


class TestNormalizeStandard(unittest.TestCase):
    """"Derived Features. Normalize Standard feature tests. We'll use the from_df function for this
    """
    def test_read_base_non_inference(self):
        fc = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        s_name = 'AmountScale'
        s_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureNormalizeStandard(s_name, s_type, fc)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            mn = df['Amount'].mean()
            st = df['Amount'].std()
            td2 = ft.TensorDefinition('Derived', [fs])
            df = e.from_df(td2, df, inference=False)
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
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            td = ft.TensorDefinition('Derived', [fs])
            df_1 = e.from_df(td, df, inference=False)
            # Now remove a line and run in inference mode
            df = df.iloc[:-1]
            df_1 = df_1.iloc[:-1]
            df_2 = e.from_df(td, df, inference=True)
            self.assertEqual(df_1[s_name].equals(df_2[s_name]), True, f'Inference problem. Series not the same')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')


class TestIndex(unittest.TestCase):
    """Index Feature Testcases.
    """
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
            df = e.from_df(td2, df, inference=False)
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
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            td = ft.TensorDefinition('Derived', [fi])
            df_1 = e.from_df(td, df, inference=False)
            # Now remove a line and run in inference mode
            df = df.iloc[1:]
            df_1 = df_1.iloc[1:]
            df_2 = e.from_df(td, df, inference=True)
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
            df_c = e.from_csv(ft.TensorDefinition('All', [fc]), file, inference=False)
            # Remove row and set variables. Also remove from categories!
            df_r = df_c.iloc[1:].copy()
            df_r['MCC'] = df_c['MCC'].cat.remove_categories(df_c['MCC'][0])
            mcc_v = df_r['MCC'].unique()
            # Run non-inference on removed panda
            td = ft.TensorDefinition('Derived', [fi])
            e.from_df(td, df_r, inference=False)
            # And inference on original
            df = e.from_df(td, df_c, inference=True)
            self.assertEqual(len(fi), len(mcc_v), f'Length of dictionary changed {len(fi)}')
            self.assertEqual(td.inference_ready, True, f'Tensor should still be ready for inference')
            self.assertEqual(df[ind_name][0], 0, f'Missing row should have been default {df[ind_name][0]}')


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
        fo = ft.FeatureOneHot('MCC_OH', fc)
        fi = ft.FeatureIndex('MCC_ID', ft.FEATURE_TYPE_INT_16, fm)
        fl = ft.FeatureLabelBinary('Fraud_Label', ff)
        td2 = ft.TensorDefinition('Derived', [fa, fo, fi, fl])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            df = e.from_df(td2, df, inference=False)
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
                    # TODO check numpy std <> panda's std?
                    # self.assertEqual(l.std(), df[fa.name].std(), f'Std do not match {l.std()}')
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
        fo = ft.FeatureOneHot('MCC_OH', fc)
        fi = ft.FeatureIndex('MCC_ID', ft.FEATURE_TYPE_INT_16, fm)
        fl = ft.FeatureLabelBinary('Fraud_Label', ff)
        td2 = ft.TensorDefinition('Derived', [fa, fo, fi, fl])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            df = e.from_df(td2, df, inference=False)
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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
