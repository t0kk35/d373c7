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
            df = e.from_df(ft.TensorDefinition('Derived', [fo]), df, inference=False)
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

    def test_rest_base_inference_added_element(self):
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
            df = e.from_df(ft.TensorDefinition('Derived', [fs]), df, inference=False)
            self.assertEqual(len(df.columns), 1, f'Only one columns should have been returned')
            self.assertEqual(df.columns[0], s_name, f'Column name is not correct {df.columns[0]}')
            self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
            self.assertEqual(fs.maximum, mx, f'Maximum not set correctly {fs.maximum}')
            self.assertEqual(fs.minimum, mn, f'Minimum not set correctly {fs.maximum}')

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
            df = e.from_df(ft.TensorDefinition('Derived', [fs]), df, inference=False)
            self.assertEqual(len(df.columns), 1, f'Only one columns should have been returned')
            self.assertEqual(df.columns[0], s_name, f'Column name is not correct {df.columns[0]}')
            self.assertEqual(fs.inference_ready, True, f'Feature should now be inference ready')
            self.assertEqual(fs.mean, mn, f'Mean not set correctly {fs.mean}')
            self.assertEqual(fs.stddev, st, f'Standard Dev not set correctly {fs.stddev}')

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
            df = e.from_df(ft.TensorDefinition('Derived', [fi]), df, inference=False)
            self.assertEqual(len(df.columns), 1, f'Should have gotten one column. Got {len(df.columns)}')
            self.assertEqual(df.columns[0], ind_name, f'Column name incorrect. Got {df.columns[0]}')
            self.assertEqual(fi.inference_ready, True, f'Index feature should be ready for inference')
            self.assertEqual(len(fi.dictionary), len(mcc_v), f'Dictionary length not correct {len(fi.dictionary)}')
            self.assertListEqual(list(fi.dictionary.keys()), list(mcc_v), f'Dictionary values don not match')
            self.assertEqual(td.inference_ready, True, f'Tensor should have been ready for inference now')
            self.assertEqual(td.rank, 2, f'This should have been a rank 2 tensor. Got {td.rank}')

    # TODO More tests here


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
            with self.assertRaises(ft.TensorDefinitionException):
                _ = e.reshape(td2, df)


class TestToNumpy(unittest.TestCase):
    """Test for conversion to_numpy
    """
    def test_create(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT_32)
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default='0000')
        td1 = ft.TensorDefinition('Source', [fa, fc])
        fo = ft.FeatureOneHot('MCC_OH', fc)
        fi = ft.FeatureIndex('MCC_ID', ft.FEATURE_TYPE_INT_16, fc)
        td2 = ft.TensorDefinition('Derived', [fa, fc, fo, fi])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file, inference=False)
            df = e.from_df(td2, df, inference=False)
            x = e.to_numpy_list(td2, df)
            print(x)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
