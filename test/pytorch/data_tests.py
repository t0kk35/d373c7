"""
Unit Tests for PyTorch Data Features
(c) 2020 d373c7
"""
import unittest
import torch
import torch.utils.data as data
import d373c7.features as ft
import d373c7.engines as en
import d373c7.pytorch as pt
import d373c7.pytorch.data

FILES_DIR = './files/'


class TestNumpyDataSet(unittest.TestCase):
    """Numpy Dataset test cases
    """
    s_features = [
        ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT),
        ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING),
        ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL),
        ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL),
        ft.FeatureSource('Fraud', ft.FEATURE_TYPE_INT_8)
    ]
    d_features = [
        ft.FeatureNormalizeScale('Amount_Scale', ft.FEATURE_TYPE_FLOAT_32, s_features[0]),
        ft.FeatureOneHot('MCC_OH', s_features[2]),
        ft.FeatureIndex('Country_Index', ft.FEATURE_TYPE_INT_16, s_features[3]),
        ft.FeatureSource('Fraud', ft.FEATURE_TYPE_INT_8)
    ]

    def test_creation_base(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tdb = ft.TensorDefinition('Base', self.s_features)
        tdd = ft.TensorDefinition('Derived', self.d_features)
        tdd.set_label(self.d_features[-1])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(tdb, file, inference=False)
            df = e.from_df(tdd, df, inference=False)
            npl = e.to_numpy_list(tdd, df)
            ds = pt.NumpyListDataSet(tdd, npl)
            self.assertEqual(len(ds), len(npl), f'Length of DS is wrong. Got {len(ds)}. Expected {len(npl)}')
            t = ds[0]
            self.assertIsInstance(t, list, f'__get_item__ should have returned a list')
            self.assertIsInstance(t[0], torch.Tensor, f'__get_item__ should have returned a list of Tensors')
            self.assertEqual(len(t), len(tdd.learning_categories), f'Number of list must be number of Learning cats')
            # Test Shapes
            for n, t in zip(npl.lists, ds[0]):
                ns = n.shape[1] if len(n.shape) > 1 else 0
                ts = 0 if len(list(t.shape)) == 0 else list(t.shape)[0]
                self.assertEqual(ns, ts)
            # Test data types.
            for i, d in enumerate(d373c7.pytorch.data._DTypeHelper.get_dtypes(tdd)):
                self.assertEqual(ds[0][i].dtype, d, f'Default data types don not match {i}, expected {d}')

    def test_creation_bad(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tdb = ft.TensorDefinition('Base', self.s_features)
        tdd = ft.TensorDefinition('Derived', self.d_features)
        tdd.set_label(self.d_features[-1])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(tdb, file, inference=False)
            df = e.from_df(tdd, df, inference=False)
            npl = e.to_numpy_list(tdd, df)
            # Try building off of the wrong tensor definition
            with self.assertRaises(pt.PyTorchTrainException):
                _ = pt.NumpyListDataSet(tdb, npl)

    def test_creation_data_loader(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        bs = 3
        tdb = ft.TensorDefinition('Base', self.s_features)
        tdd = ft.TensorDefinition('Derived', self.d_features)
        tdd.set_label(self.d_features[-1])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(tdb, file, inference=False)
            df = e.from_df(tdd, df, inference=False)
            npl = e.to_numpy_list(tdd, df)
            ds = pt.NumpyListDataSet(tdd, npl)
            dl = ds.data_loader(torch.device('cpu'), bs)
            t = next(iter(dl))
            self.assertEqual(len(t), len(tdd.learning_categories))
            # Test data types.
            for i, d in enumerate(d373c7.pytorch.data._DTypeHelper.get_dtypes(tdd)):
                self.assertEqual(t[i].dtype, d, f'Default data types don not match {i}, expected {d}')
            # Check batch-size
            for i, te in enumerate(t):
                self.assertEqual(te.shape[0], bs, f'Batch size does not match item {i}. Got {te.shape[0]}')


class TestClassSampler(unittest.TestCase):
    """Class Sampler test cases
    """
    s_features = [
        ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT),
        ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING),
        ft.FeatureSource('MCC', ft.FEATURE_TYPE_CATEGORICAL),
        ft.FeatureSource('Country', ft.FEATURE_TYPE_CATEGORICAL),
        ft.FeatureSource('Fraud', ft.FEATURE_TYPE_INT_8)
    ]
    d_features = [
        ft.FeatureNormalizeScale('Amount_Scale', ft.FEATURE_TYPE_FLOAT_32, s_features[0]),
        ft.FeatureOneHot('MCC_OH', s_features[2]),
        ft.FeatureIndex('Country_Index', ft.FEATURE_TYPE_INT_16, s_features[3]),
        ft.FeatureSource('Fraud', ft.FEATURE_TYPE_INT_8)
    ]

    def test_creation_base(self):
        file = FILES_DIR + 'engine_test_base_comma.csv'
        bs = 5
        tdb = ft.TensorDefinition('Base', self.s_features)
        tdd = ft.TensorDefinition('Derived', self.d_features)
        tdd.set_label(self.d_features[-1])
        with en.EnginePandasNumpy() as e:
            df = e.from_csv(tdb, file, inference=False)
            df = e.from_df(tdd, df, inference=False)
            npl = e.to_numpy_list(tdd, df)
            cs = pt.ClassSampler(tdd, npl)
            self.assertIsInstance(cs, pt.ClassSampler, f'Was expecting ClassSampler type {type(cs)}')
            sm = cs.over_sampler(bs, replacement=False)
            self.assertIsInstance(sm, data.WeightedRandomSampler, f'Was expecting Weighted Random Sampler {type(sm)}')
            self.assertEqual(len(sm), bs, f'Length not correct {len(sm)}')
            self.assertListEqual(sorted(list(sm)), list(range(len(npl))), f'Each index should be in the weight list')

    def test_creation_bad(self):
        # TODO Make some tests with bad npls and batch sizes.
        pass


def main():
    unittest.main()


if __name__ == '__main__':
    main()
