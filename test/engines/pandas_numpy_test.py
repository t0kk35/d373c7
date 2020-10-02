"""
Unit Tests for PandasNumpy Engine
(c) 2020 d373c7
"""
import unittest
import d373c7.features as ft
import d373c7.engines as en

FILES_DIR = './files/'


class TestReading(unittest.TestCase):
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
                df = e.from_csv(ft.TensorDefinition(d, [f]), file)
                self.assertEqual(len(df.columns), 1, f'Expected a one column panda for read test {d}')
                self.assertEqual(df.columns[0], f.name, f'Wrong panda column for read test {d}. Got {df.columns[0]}')

    def test_read_base_all(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [f for f, d in TestReading.features]), file)
            self.assertEqual(len(df.columns), len(TestReading.features),
                             f'Incorrect number of columns for read all test. got {len(df.columns)}')
            for (f, _), c in zip(TestReading.features, df.columns):
                self.assertEqual(f.name, c, f'Incorrect column name in read test all got {c}, expected {f.name}')

    def test_read_base_all_non_def_delimiter(self):
        file = FILES_DIR + 'engine_test_base_pipe.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [f for f, d in TestReading.features]), file, delimiter='|')
            self.assertEqual(len(df.columns), len(TestReading.features),
                             f'Incorrect number of columns for read all test. got {len(df.columns)}')
            for (f, _), c in zip(TestReading.features, df.columns):
                self.assertEqual(f.name, c, f'Incorrect column name in read test all got {c}, expected {f.name}')


class TestCategorical(unittest.TestCase):
    def test_is_categorical(self):
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file)
            self.assertEqual(df.dtypes[0], 'category', 'Source with f_type should be categorical')

    def test_categorical_default(self):
        default = 'DEF'
        fc = ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL, default=default)
        file = FILES_DIR + 'engine_test_base_comma.csv'
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(ft.TensorDefinition('All', [fc]), file)
            self.assertEqual(fc.default, default, f'Default incorrect. Got {fc.default}')
            self.assertIn(default, list(df['MCC']), f'Default not found in Panda')


class TestReshape(unittest.TestCase):
    def test_reshape(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        td1 = ft.TensorDefinition('All', [f for f, d in TestReading.features])
        td2 = ft.TensorDefinition('All', [f for f, d in TestReading.features])
        rf = TestReading.features[0][0]
        td2.remove(rf)
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(td1, file)
            df = e.reshape(td2, df)
            self.assertEqual(len(df.columns), len(td1)-1, f'Reshape Panda wrong length. gor {len(df.columns)}')
            self.assertNotIn(rf.name, df.columns, f'{rf.name} should not have been in reshaped Panda')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
