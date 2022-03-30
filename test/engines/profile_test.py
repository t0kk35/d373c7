"""
Unit Tests for Profile Package
(c) 2022 d373c7
"""
import unittest
import numpy as np
import statistics as stat

import pandas as pd

import d373c7.engines as en
import d373c7.features as ft

from d373c7.features.group import Aggregator, AGGREGATOR_SUM, AGGREGATOR_COUNT, AGGREGATOR_MIN, AGGREGATOR_MAX
from d373c7.features.group import AGGREGATOR_AVG, AGGREGATOR_STDDEV
from d373c7.engines.profile import ProfileNative, ProfileException, ProfileElementNative, ProfileElementNativeDict

FILES_DIR = './files/'


def card_is_1(x: str) -> bool:
    return x == 'CARD-1'


class TestNativeElement(unittest.TestCase):
    def test_base_native_element(self):
        lst = [1.0, 2.0, 2.5, 3, 5, 5.5]
        pe = ProfileElementNative()
        for e in lst:
            pe.contribute(e)
        x = len(lst)
        y = pe.aggregate(0, AGGREGATOR_COUNT)
        self.assertEqual(x, y, f'Counts should have been equal {x} {y}')
        x = sum(lst)
        y = pe.aggregate(0, AGGREGATOR_SUM)
        self.assertEqual(x, y, f'Sums should have been equal {x} {y}')
        x = stat.mean(lst)
        y = pe.aggregate(0, AGGREGATOR_AVG)
        self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y}')
        x = stat.stdev(lst)
        y = pe.aggregate(0, AGGREGATOR_STDDEV)
        self.assertAlmostEqual(x, y, places=15, msg=f'Standard deviation should have been equal {x} {y}')
        x = max(lst)
        y = pe.aggregate(0, AGGREGATOR_MAX)
        self.assertEqual(x, y, f'Maximums should have been equal {x} {y}')
        x = min(lst)
        y = pe.aggregate(0, AGGREGATOR_MIN)
        self.assertEqual(x, y, f'Minimums should have been equal {x} {y}')

    def test_element_merge_native_element(self):
        lst1 = [1.0, 2.0, 2.5, 3, 5, 5.5]
        lst2 = [6.0, 7.5, 8.0]
        lst3 = lst1 + lst2
        pe1 = ProfileElementNative()
        for e in lst1:
            pe1.contribute(e)
        pe2 = ProfileElementNative()
        for e in lst2:
            pe2.contribute(e)
        pe3 = ProfileElementNative()
        pe3.merge(pe1)
        pe3.merge(pe2)
        x = len(lst3)
        y = pe3.aggregate(0, AGGREGATOR_COUNT)
        self.assertEqual(x, y, f'Counts should have been equal {x} {y}')
        x = sum(lst3)
        y = pe3.aggregate(0, AGGREGATOR_SUM)
        self.assertEqual(x, y, f'Sums should have been equal {x} {y}')
        x = stat.mean(lst3)
        y = pe3.aggregate(0, AGGREGATOR_AVG)
        self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y}')
        x = stat.stdev(lst3)
        y = pe3.aggregate(0, AGGREGATOR_STDDEV)
        self.assertAlmostEqual(x, y, places=15, msg=f'Standard deviation should have been equal {x} {y}')
        x = max(lst3)
        y = pe3.aggregate(0, AGGREGATOR_MAX)
        self.assertEqual(x, y, f'Maximums should have been equal {x} {y}')
        x = min(lst3)
        y = pe3.aggregate(0, AGGREGATOR_MIN)
        self.assertEqual(x, y, f'Minimums should have been equal {x} {y}')

    def merge_empty_element(self):
        lst1 = [1.0, 2.0, 2.5, 3, 5, 5.5]
        pe1 = ProfileElementNative()
        for e in lst1:
            pe1.contribute(e)
        pe2 = ProfileElementNative()
        pe3 = ProfileElementNative()
        pe3.merge(pe1)
        pe3.merge(pe2)
        x = len(lst1)
        y = pe3.aggregate(0, AGGREGATOR_COUNT)
        self.assertEqual(x, y, f'Counts should have been equal {x} {y}')
        x = sum(lst1)
        y = pe3.aggregate(0, AGGREGATOR_SUM)
        self.assertEqual(x, y, f'Sums should have been equal {x} {y}')
        x = stat.mean(lst1)
        y = pe3.aggregate(0, AGGREGATOR_AVG)
        self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y}')
        x = stat.stdev(lst1)
        y = pe3.aggregate(0, AGGREGATOR_STDDEV)
        self.assertAlmostEqual(x, y, places=15, msg=f'Standard deviation should have been equal {x} {y}')
        x = max(lst1)
        y = pe3.aggregate(0, AGGREGATOR_MAX)
        self.assertEqual(x, y, f'Maximums should have been equal {x} {y}')
        x = min(lst1)
        y = pe3.aggregate(0, AGGREGATOR_MIN)
        self.assertEqual(x, y, f'Minimums should have been equal {x} {y}')


class TestNativeDictElement(unittest.TestCase):
    def test_base_native_dict_element(self):
        lst = [('a', 1.0), ('b', 2.0), ('b', 2.5), ('a', 3), ('a', 5), ('c', 5.5)]
        pe = ProfileElementNativeDict()
        for k, v in lst:
            pe.contribute((k, v))

        for key in set([e for e, _ in lst]):
            # Make a list with just one of the keys
            f_lst = [v for k, v in lst if k == key]
            x = len(f_lst)
            y = pe.aggregate((key, 0), AGGREGATOR_COUNT)
            self.assertEqual(x, y, f'Counts should have been equal {x} {y} for key {key}')
            x = sum(f_lst)
            y = pe.aggregate((key, 0), AGGREGATOR_SUM)
            self.assertEqual(x, y, f'Sums should have been equal {x} {y} for key {key}')
            x = stat.mean(f_lst)
            y = pe.aggregate((key, 0), AGGREGATOR_AVG)
            if len(f_lst) > 1:
                self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y} for key {key}')
                x = stat.stdev(f_lst)
                y = pe.aggregate((key, 0), AGGREGATOR_STDDEV)
                self.assertAlmostEqual(x, y, places=15, msg=f'Std dev should have been equal {x} {y} for key {key}')
            x = max(f_lst)
            y = pe.aggregate((key, 0), AGGREGATOR_MAX)
            self.assertEqual(x, y, f'Maximums should have been equal {x} {y} for key {key}')
            x = min(f_lst)
            y = pe.aggregate((key, 0), AGGREGATOR_MIN)
            self.assertEqual(x, y, f'Minimums should have been equal {x} {y} for key {key}')

    def test_element_merge_native_element(self):
        lst1 = [('a', 1.0), ('b', 2.0), ('b', 2.5), ('a', 3)]
        lst2 = [('a', 5), ('c', 5.5)]
        lst3 = lst1 + lst2
        pe1 = ProfileElementNativeDict()
        for k, v in lst1:
            pe1.contribute((k, v))
        pe2 = ProfileElementNativeDict()
        for k, v in lst2:
            pe2.contribute((k, v))
        pe3 = ProfileElementNativeDict()
        pe3.merge(pe1)
        pe3.merge(pe2)

        for key in set([e for e, _ in lst3]):
            # Make a list with just one of the keys
            f_lst = [v for k, v in lst3 if k == key]
            x = len(f_lst)
            y = pe3.aggregate((key, 0), AGGREGATOR_COUNT)
            self.assertEqual(x, y, f'Counts should have been equal {x} {y} for key {key}')


class TestNativeProfile(unittest.TestCase):
    def test_base_base_feature_not_in_td(self):
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 5, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fc])
        with self.assertRaises(ProfileException):
            _ = ProfileNative([fgc], fd, td)

    def test_base_filter_not_in_td(self):
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        ff = ft.FeatureFilter('Filter', ft.FEATURE_TYPE_BOOL, card_is_1, [fc])
        fg = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, ff, tp, 5, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with self.assertRaises(ProfileException):
            _ = ProfileNative([fg], fd, td)

    def test_base_time_feature_not_in_td(self):
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        ff = ft.FeatureFilter('Filter', ft.FEATURE_TYPE_BOOL, card_is_1, [fc])
        fg = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, ff, tp, 5, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc])
        with self.assertRaises(ProfileException):
            _ = ProfileNative([fg], fd, td)

    def test_base_bad_aggregator(self):
        tp = ft.TIME_PERIOD_DAY
        bag = Aggregator(0, 'Bad', 'bad')
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 5, bag)
        td = ft.TensorDefinition('Source', [fa, fc])
        with self.assertRaises(ProfileException):
            _ = ProfileNative([fgc], fd, td)

    def test_base_aggregators(self):
        # The dates range for over 5 days. So a 5-day profile should give the full range of the data.
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 5, ft.AGGREGATOR_COUNT)
        fgs = ft.FeatureGrouper('S', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 5, ft.AGGREGATOR_SUM)
        fga = ft.FeatureGrouper('A', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 5, ft.AGGREGATOR_AVG)
        fgt = ft.FeatureGrouper('ST', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 5, ft.AGGREGATOR_STDDEV)
        fgx = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 5, ft.AGGREGATOR_MAX)
        fgm = ft.FeatureGrouper('M', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 5, ft.AGGREGATOR_MIN)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        p = ProfileNative([fgc, fgs, fga, fgt, fgx, fgm], fd, td)
        a = df.to_numpy()
        for i in range(len(a)):
            p.contribute(a[i])
        x = p.list(a[len(a)-1])
        self.assertEqual(x[0], len(a[:, 0]), f'Counts do not match {x[0]} {len(a[:, 0])} ')
        self.assertEqual(x[1], np.sum(a[:, 0]), f'Sums do not match {x[1]} {np.sum(a[:, 0])}')
        self.assertEqual(x[2], np.average(a[:, 0]), f'Averages do not match {x[2]} {np.average(a[:, 0])}')
        self.assertEqual(x[3], np.std(a[:, 0], ddof=1), f'Stddev does not match {x[3]} {np.std(a[:, 0], ddof=1)}')
        self.assertEqual(x[4], np.amax(a[:, 0]), f'Maximums do not match {x[4]} {np.amax(a[:, 0])}')
        self.assertEqual(x[5], np.amin(a[:, 0]), f'Averages do not match {x[5]} {np.amin(a[:, 0])}')

    def test_base_filter(self):
        # Let's repeat and see if the filtering works. Filter out the entries of CARD-1
        threads = 1
        file = FILES_DIR + 'engine_test_base_comma.csv'
        tp = ft.TIME_PERIOD_DAY
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        ff = ft.FeatureFilter('Filter', ft.FEATURE_TYPE_BOOL, card_is_1, [fc])
        fg = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, ff, tp, 5, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd, ff])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        p = ProfileNative([fg], fd, td)
        a = df.to_numpy()
        for i in range(len(a)):
            p.contribute(a[i])
        x = p.list(a[len(a)-1])
        # Filter the input, this should give the same result as the profile. (If the profile applies the filter)
        y = df[df['Filter'] == 1].to_numpy()
        self.assertEqual(x[0], len(y[:, 0]), f'Counts do not match {x[0]} {len(y[:, 0])} ')

    def test_base_same_week(self):
        # Let's do some week testing
        threads = 1
        tp = ft.TIME_PERIOD_WEEK
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        # Aggregate for 1 week only.
        fg = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        # Reset the dates so the months shift
        df.at[2, 'Date'] = pd.Timestamp(year=2020, month=1, day=5)  # This is a Sunday
        df.at[3, 'Date'] = pd.Timestamp(year=2020, month=1, day=6)  # This is a Monday, count should reset
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=1, day=7)  # This is a Tuesday
        p = ProfileNative([fg], fd, td)
        a = df.to_numpy()
        c = 0
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for i in range(len(a)):
            p.contribute(a[i])
            d = tp.start_period(df.iloc[i]['Date'].to_pydatetime())
            if tp.delta_between(start_date, d) == 0:
                c += 1
            else:
                # week changed, count should have been reset.
                c = 1
                start_date = d
            self.assertEqual(p.list(a[i])[0], float(c), f'Counts do not match {p.list(a[i])[0]} {float(c)}')

    def test_base_same_month(self):
        # Let's do some month testing
        threads = 1
        tp = ft.TIME_PERIOD_MONTH
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        # Aggregate for 1 month only.
        fg = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        # Reset the dates so the months shift
        df.at[2, 'Date'] = pd.Timestamp(year=2020, month=1, day=31)
        df.at[3, 'Date'] = pd.Timestamp(year=2020, month=2, day=1)
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=2, day=2)
        p = ProfileNative([fg], fd, td)
        a = df.to_numpy()
        c = 0
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for i in range(len(a)):
            p.contribute(a[i])
            d = tp.start_period(df.iloc[i]['Date'].to_pydatetime())
            if tp.delta_between(start_date, d) == 0:
                c += 1
            else:
                # Month changed, count should have been reset.
                c = 1
                start_date = d
            self.assertEqual(p.list(a[i])[0], float(c), f'Counts do not match {p.list(a[i])[0]} {float(c)}')

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
        fgc5 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 5, ft.AGGREGATOR_COUNT)
        fgc2 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 2, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        p = ProfileNative([fgc5, fgc2], fd, td)
        a = df.to_numpy()
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for i in range(len(a)):
            p.contribute(a[i])
            d = tp.start_period(df.iloc[i]['Date'].to_pydatetime())
            g = tp.delta_between(start_date, d) - 1 if i > 1 else 0
            x = p.list(a[i])
            self.assertEqual(x[0], i+1, f'5 day counts do not match {x[0]} {i+1} ')
            self.assertEqual(x[1], i+1-g, f'2 day counts do not match {x[1]} {i+1-g} ')

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
        fgc2 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 2, ft.AGGREGATOR_COUNT)
        fgc1 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=1, day=6)  # This is a Monday, count should reset
        p = ProfileNative([fgc2, fgc1], fd, td)
        a = df.to_numpy()
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for i in range(len(a)):
            p.contribute(a[i])
            d = tp.start_period(df.iloc[i]['Date'].to_pydatetime())
            g = i+1 if tp.delta_between(start_date, d) == 0 else 1
            x = p.list(a[i])
            self.assertEqual(x[0], i+1, f'2 week counts do not match {x[0]} {i+1} ')
            self.assertEqual(x[1], g, f'1 week counts do not match {x[1]} {g} ')

    # Multiple month periods
    def test_multiple_month_periods(self):
        # The dates range for over 6 days. So a 2-week profile should give the full range of the data.
        # The 1-week profile should reset
        threads = 1
        tp = ft.TIME_PERIOD_MONTH
        file = FILES_DIR + 'engine_test_base_comma.csv'
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fc = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
        fd = ft.FeatureSource('Date', ft.FEATURE_TYPE_DATE_TIME, format_code='%Y%m%d')
        fgc2 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 2, ft.AGGREGATOR_COUNT)
        fgc1 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=2, day=1)
        p = ProfileNative([fgc2, fgc1], fd, td)
        a = df.to_numpy()
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for i in range(len(a)):
            p.contribute(a[i])
            d = tp.start_period(df.iloc[i]['Date'].to_pydatetime())
            g = i+1 if tp.delta_between(start_date, d) == 0 else 1
            x = p.list(a[i])
            self.assertEqual(x[0], i+1, f'2 month counts do not match {x[0]} {i+1} ')
            self.assertEqual(x[1], g, f'1 month counts do not match {x[1]} {g} ')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
