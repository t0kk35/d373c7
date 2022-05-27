"""
Unit Tests for Profile Numpy Package
(c) 2022 d373c7
"""
import unittest
import numpy as np
import pandas as pd
import statistics as stat

import d373c7.engines as en
import d373c7.features as ft

from d373c7.engines.profile import ProfileNative
from d373c7.engines.profile_numpy import ProfileNumpy, profile_contrib, profile_aggregate, profile_time_logic
from d373c7.engines.profile_numpy import NUMBER_OF_AGGREGATORS
from d373c7.features.group import TIME_PERIODS

from typing import Tuple

FILES_DIR = './files/'


def card_is_1(x: str) -> bool:
    return x == 'CARD-1'


def card_is_2(x: str) -> bool:
    return x == 'CARD-2'


def add_one(x: float) -> float:
    return x + 1


def always_false(x: str) -> float:
    return False


class TestNumpy(unittest.TestCase):
    # Define some dummy FeatureGroupers
    tp = ft.TIME_PERIOD_DAY
    fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
    fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
    fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
    fgc = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_COUNT)
    fgs = ft.FeatureGrouper('S', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_SUM)
    fga = ft.FeatureGrouper('A', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_AVG)
    fgt = ft.FeatureGrouper('ST', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_STDDEV)
    fgx = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_MAX)
    fgm = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_MIN)

    def test_base_setup(self):
        pe = ProfileNumpy([self.fgc, self.fgs, self.fga, self.fgt, self.fgx, self.fgm])
        pea = pe.new_element_array()
        b_flt = pe.base_filters
        f_ind = pe.filter_indexes
        f_flt = pe.feature_filters
        a_ind = pe.aggregator_indexes
        tw = pe.time_windows
        tp_flt = pe.timeperiod_filters
        # Element array Validation
        self.assertEqual(type(pea), np.ndarray, f'Should have been an ndarray')
        self.assertEqual(pea.shape, (1, 1, NUMBER_OF_AGGREGATORS),
                         f'Not expected (1, 1, {NUMBER_OF_AGGREGATORS}) shape. But {pea.shape}')
        self.assertTrue(np.array_equal(pea, np.zeros((1, 1, NUMBER_OF_AGGREGATORS))), f'Array should have been zeros')
        # Base Filter validation
        self.assertEqual(type(b_flt), np.ndarray, f'Expecting an ndarray')
        self.assertEqual(b_flt.shape, (1, 1), f'Expected one element/one filter got {b_flt.shape}')
        self.assertEqual(type(b_flt[0, 0].item()), np.bool, f'Array should contain bools. Not {type(b_flt[0: 0])}')
        # Filter indexes validation
        self.assertEqual(type(f_ind), np.ndarray, f'Expecting an ndarray')
        self.assertEqual(f_ind.shape, (1,), f'Expected one element got {f_ind.shape}')
        self.assertEqual(type(f_ind[0].item()), np.int, f'Array should contain ints. Not {type(f_ind[0])}')
        # Feature Filter validation
        self.assertEqual(type(f_flt), np.ndarray, f'Expecting an ndarray')
        self.assertEqual(f_flt.shape, (6, 1), f'Expected six features/one element got {f_flt.shape}')
        self.assertEqual(type(f_flt[0].item()), np.bool, f'Array should contain bools. Not {type(f_flt[0])}')
        # Aggregator indexes validation
        self.assertEqual(type(a_ind), np.ndarray, f'Expecting an ndarray')
        self.assertEqual(a_ind.shape, (6,), f'Expected six features got {a_ind.shape}')
        self.assertEqual(type(a_ind[0].item()), np.int, f'Array should contain ints. Not {type(a_ind[0])}')
        # Time Window validation
        self.assertEqual(type(tw), np.ndarray, f'Expecting an ndarray')
        self.assertEqual(tw.shape, (6,), f'Expected six features got {tw.shape}')
        self.assertEqual(type(tw[0].item()), np.int, f'Array should contain ints. Not {type(tw[0])}')
        # Time Period validation
        self.assertEqual(type(tp_flt), np.ndarray, f'Expecting an ndarray')
        self.assertEqual(tp_flt.shape, (len(TIME_PERIODS), 1), f'Expected ({len(TIME_PERIODS)}, 1) got {tp_flt.shape}')
        self.assertEqual(type(tp_flt[0, 0].item()), np.bool, f'Array should contain bools. Not {type(tp_flt[0])}')

    def test_base_contrib(self):
        lst = [[1.0], [2.0], [2.5], [3], [5], [5.5]]
        array = np.array(lst)
        pe = ProfileNumpy([self.fgc, self.fgs, self.fga, self.fgt, self.fgx, self.fgm])
        pea = pe.new_element_array()
        b_flt = pe.base_filters
        f_ind = pe.filter_indexes
        f_flt = pe.feature_filters
        a_ind = pe.aggregator_indexes
        tw = pe.time_windows
        for i in range(len(array)):
            profile_contrib(b_flt, f_ind, array[i], np.array([]), pea)
            agg = profile_aggregate(f_flt, a_ind, tw, pea)
            x = len(lst[:i+1])
            self.assertEqual(x, agg[0].item(), f'Counts should have been equal {x} {agg[0]}')
            x = sum([i for li in lst[:i+1] for i in li])
            self.assertEqual(x, agg[1].item(), f'Sums should have been equal {x} {agg[1]}')
            x = stat.mean([i for li in lst[:i+1] for i in li])
            self.assertAlmostEqual(x, agg[2].item(), places=15, msg=f'Averages should have been equal {x} {agg[2]}')
            if len(lst[:i+1]) > 1:
                x = stat.stdev([i for li in lst[:i+1] for i in li])
                self.assertAlmostEqual(x, agg[3].item(), places=15, msg=f'Stddev should have been equal {x} {agg[3]}')
            x = max([i for li in lst[:i+1] for i in li])
            self.assertEqual(x, agg[4].item(), f'Maximums should have been equal {x} {agg[4]}')
            x = min([i for li in lst[:i+1] for i in li])
            self.assertEqual(x, agg[5].item(), f'Minimums should have been equal {x} {agg[5]}')


class TestNumpyHelpers(unittest.TestCase):
    def test_get_deltas(self):
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, ft.TIME_PERIOD_DAY, 5, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        pny = ProfileNumpy([fgc])
        delta1 = pny.get_deltas(df, fd)

        # Reconstruct deltas with TimePeriod Logic
        def calc_delta(d1: pd.Timestamp, d2: pd.Timestamp) -> Tuple[int, ...]:
            return tuple([0 for _ in TIME_PERIODS]) if pd.isnull(d2) else tuple(
                [
                    tp.delta_between(tp.start_period(d2.to_pydatetime()), tp.start_period(d1.to_pydatetime()))
                    for tp in TIME_PERIODS
                ]
            )

        delta2 = np.vectorize(calc_delta, otypes=[int for _ in TIME_PERIODS])(
                    df[fd.name], df[fd.name].reset_index(drop=True).shift(1))

        delta2 = np.stack(delta2, axis=1).astype(np.int16)
        # Now test if d1 and d2 are equal
        self.assertTrue(np.array_equal(delta1, delta2), f'Arrays should have been equal')


class TestNumpyProfile(unittest.TestCase):
    def test_base_aggregators(self):
        # The dates range for over 5 days. So a 5-day profile should give the full range of the data.
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 5, ft.AGGREGATOR_COUNT)
        fgs = ft.FeatureGrouper('S', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 5, ft.AGGREGATOR_SUM)
        fga = ft.FeatureGrouper('A', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 5, ft.AGGREGATOR_AVG)
        fgt = ft.FeatureGrouper('ST', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 5, ft.AGGREGATOR_STDDEV)
        fgx = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 5, ft.AGGREGATOR_MAX)
        fgm = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 5, ft.AGGREGATOR_MIN)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        pnt = ProfileNative([fgc, fgs, fga, fgt, fgx, fgm])
        pny = ProfileNumpy([fgc, fgs, fga, fgt, fgx, fgm])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        for i in range(len(amounts)):
            pnt.contribute(([amounts[i].item()], df[fd.name].iloc[i].to_pydatetime(), []))
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], np.array([]), pea)
        x = np.array(pnt.list(df[fd.name].iloc[-1].to_pydatetime()))
        y = profile_aggregate(f_flt, a_ind, tw, pea)
        self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y}')

    def test_base_one_contribution(self):
        # Do one contribution in a multi period grouper feature
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_COUNT)
        fgs = ft.FeatureGrouper('S', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_SUM)
        fga = ft.FeatureGrouper('A', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_AVG)
        fgt = ft.FeatureGrouper('ST', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_STDDEV)
        fgx = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_MAX)
        fgm = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_MIN)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        pny = ProfileNumpy([fgc, fgs, fga, fgt, fgx, fgm])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        profile_contrib(b_flt, f_ind, amounts[0], np.array([]), pea)
        y = profile_aggregate(f_flt, a_ind, tw, pea)
        # All, except the stddev should be the contributed amount
        ind = [i for i, f in enumerate(pny.features) if f.aggregator != ft.AGGREGATOR_STDDEV]
        self.assertTrue(np.all(y[ind] == amounts[0]), f'{y[ind]} is not all {amounts[0]}')
        # The stddev should be 0
        ind = [i for i, f in enumerate(pny.features) if f.aggregator == ft.AGGREGATOR_STDDEV]
        self.assertTrue(np.all(y[ind] == 0), f'{y[ind]} should have been 0, it is the stddev')

    def test_base_filter(self):
        # Let's repeat and see if the filtering works. Filter out the entries of CARD-1
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        ff = ft.FeatureFilter('Filter', ft.FEATURE_TYPE_BOOL, card_is_1, [fc])
        fg = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, ff, tp, 5, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd, ff])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        pnt = ProfileNative([fg])
        pny = ProfileNumpy([fg])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        dates = df[fd.name]
        deltas = pny.get_deltas(df, fd)
        for i in range(len(amounts)):
            pnt.contribute(([amounts[i].item()], dates[i].to_pydatetime(), [filters[i].item()]))
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
        x = pnt.list(df[fd.name].iloc[-1])
        y = profile_aggregate(f_flt, a_ind, tw, pea)
        self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y}')

    def test_all_zero(self):
        # Let's repeat with a filter that is always False, this should return all zeros
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        ff = ft.FeatureFilter('Filter', ft.FEATURE_TYPE_BOOL, always_false, [fc])
        fg = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, ff, tp, 5, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd, ff])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        pny = ProfileNumpy([fg])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        for i in range(len(amounts)):
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
        y = profile_aggregate(f_flt, a_ind, tw, pea)
        self.assertEqual(np.count_nonzero(y), 0, f'Array should have been all zeros {y}')

    def test_multiple_base(self):
        # Let's repeat and see if we can use 2 different base features. We'll use the sum aggregator
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tp = ft.TIME_PERIOD_DAY
        fa1 = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fa2 = ft.FeatureExpression('Amount2', ft.FEATURE_TYPE_FLOAT, add_one, [fa1])
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fg1 = ft.FeatureGrouper('C1', ft.FEATURE_TYPE_FLOAT, fa1, fc, None, tp, 5, ft.AGGREGATOR_SUM)
        fg2 = ft.FeatureGrouper('C2', ft.FEATURE_TYPE_FLOAT, fa2, fc, None, tp, 5, ft.AGGREGATOR_SUM)
        td = ft.TensorDefinition('Source', [fa1, fa2, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        pnt = ProfileNative([fg1, fg2])
        pny = ProfileNumpy([fg1, fg2])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        dates = df[fd.name]
        for i in range(len(amounts)):
            pnt.contribute(([amounts[i][0].item(), amounts[i][1].item()], dates[i].to_pydatetime(), []))
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
        x = pnt.list(df[fa2.name].iloc[-1])
        y = profile_aggregate(f_flt, a_ind, tw, pea)
        self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y}')

    def test_multiple_filters(self):
        # Let's repeat and see if multiple filters work. Filter out the entries of CARD-1
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        ff1 = ft.FeatureFilter('Filter1', ft.FEATURE_TYPE_BOOL, card_is_1, [fc])
        ff2 = ft.FeatureFilter('Filter2', ft.FEATURE_TYPE_BOOL, card_is_2, [fc])
        fg1 = ft.FeatureGrouper('C1', ft.FEATURE_TYPE_FLOAT, fa, fc, ff1, tp, 5, ft.AGGREGATOR_COUNT)
        fg2 = ft.FeatureGrouper('C2', ft.FEATURE_TYPE_FLOAT, fa, fc, ff2, tp, 5, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd, ff1, ff2])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        pnt = ProfileNative([fg1, fg2])
        pny = ProfileNumpy([fg1, fg2])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        dates = df[fd.name]
        for i in range(len(amounts)):
            pnt.contribute(
                ([amounts[i][0].item()], dates[i].to_pydatetime(), [filters[i][0].item(), filters[i][1].item()])
            )
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
        x = pnt.list(df[fd.name].iloc[-1])
        y = profile_aggregate(f_flt, a_ind, tw, pea)
        self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y}')

    def test_multiple_base_and_filters(self):
        # Let's repeat and see if multiple bases and multiple filers work
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tp = ft.TIME_PERIOD_DAY
        fa1 = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fa2 = ft.FeatureExpression('Amount2', ft.FEATURE_TYPE_FLOAT, add_one, [fa1])
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        ff1 = ft.FeatureFilter('Filter1', ft.FEATURE_TYPE_BOOL, card_is_1, [fc])
        ff2 = ft.FeatureFilter('Filter2', ft.FEATURE_TYPE_BOOL, card_is_2, [fc])
        fg1 = ft.FeatureGrouper('C1', ft.FEATURE_TYPE_FLOAT, fa1, fc, ff1, tp, 5, ft.AGGREGATOR_COUNT)
        fg2 = ft.FeatureGrouper('C2', ft.FEATURE_TYPE_FLOAT, fa1, fc, ff2, tp, 5, ft.AGGREGATOR_COUNT)
        fg3 = ft.FeatureGrouper('C3', ft.FEATURE_TYPE_FLOAT, fa2, fc, ff1, tp, 5, ft.AGGREGATOR_COUNT)
        fg4 = ft.FeatureGrouper('C4', ft.FEATURE_TYPE_FLOAT, fa2, fc, ff2, tp, 5, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa1, fa2, fc, fd, ff1, ff2])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        pnt = ProfileNative([fg1, fg2, fg3, fg4])
        pny = ProfileNumpy([fg1, fg2, fg3, fg4])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        dates = df[fd.name]
        for i in range(len(amounts)):
            pnt.contribute(
                ([amounts[i][0].item(), amounts[i][1].item()], dates[i].to_pydatetime(),
                 [filters[i][0].item(), filters[i][1].item()])
            )
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
        x = pnt.list(df[fd.name].iloc[-1])
        y = profile_aggregate(f_flt, a_ind, tw, pea)
        self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y}')

    def test_base_same_week(self):
        # Let's do some week testing
        threads = 1
        tp = ft.TIME_PERIOD_WEEK
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        # Aggregate for 1 week only.
        fg = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        # Reset the dates so the months shift
        df.at[2, 'Date'] = pd.Timestamp(year=2020, month=1, day=5)  # This is a Sunday
        df.at[3, 'Date'] = pd.Timestamp(year=2020, month=1, day=6)  # This is a Monday, count should reset
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=1, day=7)  # This is a Tuesday
        pnt = ProfileNative([fg])
        pny = ProfileNumpy([fg])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        dates = df[fd.name]
        for i in range(len(amounts)):
            pnt.contribute(([amounts[i].item()], dates[i].to_pydatetime(), []))
            x = np.array(pnt.list(dates[i].to_pydatetime()))
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
            y = profile_aggregate(f_flt, a_ind, tw, pea)
            self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y} row {i}')

    def test_base_same_month(self):
        # Let's do some month testing
        threads = 1
        tp = ft.TIME_PERIOD_MONTH
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        # Aggregate for 1 month only.
        fg = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        # Reset the dates so the months shift
        df.at[2, 'Date'] = pd.Timestamp(year=2020, month=1, day=31)
        df.at[3, 'Date'] = pd.Timestamp(year=2020, month=2, day=1)
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=2, day=2)
        pnt = ProfileNative([fg])
        pny = ProfileNumpy([fg])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        dates = df[fd.name]
        for i in range(len(amounts)):
            pnt.contribute(([amounts[i].item()], dates[i].to_pydatetime(), []))
            x = np.array(pnt.list(dates[i].to_pydatetime()))
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
            y = profile_aggregate(f_flt, a_ind, tw, pea)
            self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y} row {i}')

    # Multiple day periods
    def test_multiple_day_periods(self):
        # The dates range for over 5 days. So a 5-day profile should give the full range of the data.
        # The 2-day profile should reset
        threads = 1
        tp = ft.TIME_PERIOD_DAY
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc5 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 5, ft.AGGREGATOR_COUNT)
        fgc2 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        pnt = ProfileNative([fgc5, fgc2])
        pny = ProfileNumpy([fgc5, fgc2])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        dates = df[fd.name]
        for i in range(len(amounts)):
            pnt.contribute(([amounts[i].item()], dates[i].to_pydatetime(), []))
            x = np.array(pnt.list(dates[i].to_pydatetime()))
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
            y = profile_aggregate(f_flt, a_ind, tw, pea)
            self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y} row {i}')

    # Multiple week periods
    def test_multiple_week_periods(self):
        # The dates range for over 6 days. So a 2-week profile should give the full range of the data.
        # The 1-week profile should reset
        threads = 1
        tp = ft.TIME_PERIOD_WEEK
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc2 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_COUNT)
        fgc1 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=1, day=6)  # This is a Monday, count should reset
        pnt = ProfileNative([fgc2, fgc1])
        pny = ProfileNumpy([fgc2, fgc1])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        dates = df[fd.name]
        for i in range(len(amounts)):
            pnt.contribute(([amounts[i].item()], dates[i].to_pydatetime(), []))
            x = np.array(pnt.list(dates[i].to_pydatetime()))
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
            y = profile_aggregate(f_flt, a_ind, tw, pea)
            self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y} row {i}')

    # Multiple month periods
    def test_multiple_month_periods(self):
        # The dates range for over 2 months. So a 2-month profile should give the full range of the data.
        # The 1-month profile should reset
        threads = 1
        tp = ft.TIME_PERIOD_MONTH
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc2 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_COUNT)
        fgc1 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=2, day=1)
        pnt = ProfileNative([fgc2, fgc1])
        pny = ProfileNumpy([fgc2, fgc1])
        pea = pny.new_element_array()
        b_flt = pny.base_filters
        f_ind = pny.filter_indexes
        f_flt = pny.feature_filters
        a_ind = pny.aggregator_indexes
        tw = pny.time_windows
        tp_flt = pny.timeperiod_filters
        amounts = df[[f.name for f in pny.base_features]].to_numpy()
        filters = df[[f.name for f in pny.filter_features]].to_numpy()
        deltas = pny.get_deltas(df, fd)
        dates = df[fd.name]
        for i in range(len(amounts)):
            pnt.contribute(([amounts[i].item()], dates[i].to_pydatetime(), []))
            x = np.array(pnt.list(dates[i].to_pydatetime()))
            profile_time_logic(tp_flt, deltas[i], pea)
            profile_contrib(b_flt, f_ind, amounts[i], filters[i], pea)
            y = profile_aggregate(f_flt, a_ind, tw, pea)
            self.assertTrue(np.array_equal(x, y), f'Lists should be the same {x} {y} row {i}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
