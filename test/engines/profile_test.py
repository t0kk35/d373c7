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

from d373c7.engines.profile import ProfileNative, ProfileElementNative


FILES_DIR = './files/'


def card_is_1(x: str) -> bool:
    return x == 'CARD-1'


def card_is_2(x: str) -> bool:
    return x == 'CARD-2'


def add_one(x: float) -> float:
    return x + 1


def always_false(x: str) -> float:
    return False


class TestNativeElement(unittest.TestCase):
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

    def test_base_native_element(self):
        lst = [1.0, 2.0, 2.5, 3, 5, 5.5]
        pe = ProfileElementNative()
        for e in lst:
            pe.contribute(e)
        x = len(lst)
        y = pe.aggregate(self.fgc)
        self.assertEqual(x, y, f'Counts should have been equal {x} {y}')
        x = sum(lst)
        y = pe.aggregate(self.fgs)
        self.assertEqual(x, y, f'Sums should have been equal {x} {y}')
        x = stat.mean(lst)
        y = pe.aggregate(self.fga)
        self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y}')
        x = stat.stdev(lst)
        y = pe.aggregate(self.fgt)
        self.assertAlmostEqual(x, y, places=15, msg=f'Standard deviation should have been equal {x} {y}')
        x = max(lst)
        y = pe.aggregate(self.fgx)
        self.assertEqual(x, y, f'Maximums should have been equal {x} {y}')
        x = min(lst)
        y = pe.aggregate(self.fgm)
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
        y = pe3.aggregate(self.fgc)
        self.assertEqual(x, y, f'Counts should have been equal {x} {y}')
        x = sum(lst3)
        y = pe3.aggregate(self.fgs)
        self.assertEqual(x, y, f'Sums should have been equal {x} {y}')
        x = stat.mean(lst3)
        y = pe3.aggregate(self.fga)
        self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y}')
        x = stat.stdev(lst3)
        y = pe3.aggregate(self.fgt)
        self.assertAlmostEqual(x, y, places=15, msg=f'Standard deviation should have been equal {x} {y}')
        x = max(lst3)
        y = pe3.aggregate(self.fgx)
        self.assertEqual(x, y, f'Maximums should have been equal {x} {y}')
        x = min(lst3)
        y = pe3.aggregate(self.fgm)
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
        y = pe3.aggregate(self.fgc)
        self.assertEqual(x, y, f'Counts should have been equal {x} {y}')
        x = sum(lst1)
        y = pe3.aggregate(self.fgs)
        self.assertEqual(x, y, f'Sums should have been equal {x} {y}')
        x = stat.mean(lst1)
        y = pe3.aggregate(self.fga)
        self.assertAlmostEqual(x, y, places=15, msg=f'Averages should have been equal {x} {y}')
        x = stat.stdev(lst1)
        y = pe3.aggregate(self.fgt)
        self.assertAlmostEqual(x, y, places=15, msg=f'Standard deviation should have been equal {x} {y}')
        x = max(lst1)
        y = pe3.aggregate(self.fgx)
        self.assertEqual(x, y, f'Maximums should have been equal {x} {y}')
        x = min(lst1)
        y = pe3.aggregate(self.fgm)
        self.assertEqual(x, y, f'Minimums should have been equal {x} {y}')


class TestNativeProfile(unittest.TestCase):
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
        p = ProfileNative([fgc, fgs, fga, fgt, fgx, fgm])
        for amount, date in zip(df[fa.name], df[fd.name]):
            p.contribute(([amount], date, []))
        x = p.list(df[fd.name].iloc[-1])
        af = df[fa.name].to_numpy()
        self.assertEqual(x[0], len(af), f'Counts do not match {x[0]} {len(af)} ')
        self.assertEqual(x[1], np.sum(af), f'Sums do not match {x[1]} {np.sum(af)}')
        self.assertEqual(x[2], np.average(af), f'Averages do not match {x[2]} {np.average(af)}')
        self.assertEqual(x[3], np.std(af, ddof=1), f'Stddev does not match {x[3]} {np.std(af, ddof=1)}')
        self.assertEqual(x[4], np.amax(af), f'Maximums do not match {x[4]} {np.amax(af)}')
        self.assertEqual(x[5], np.amin(af), f'Averages do not match {x[5]} {np.amin(af)}')

    def test_base_one_contribution(self):
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
        p = ProfileNative([fgc, fgs, fga, fgt, fgx, fgm])
        p.contribute(([df[fa.name].iloc[0]], df[fd.name].iloc[0], []))
        x = np.array(p.list(df[fd.name].iloc[0]))
        # All, except the stddev should be the contributed amount
        ind = [i for i, f in enumerate(p.features) if f.aggregator != ft.AGGREGATOR_STDDEV]
        self.assertTrue(np.all(x[ind] == df[fa.name].iloc[0]), f'{x[ind]} is not all {df[fa.name].iloc[0]}')
        # The stddev should be 0
        ind = [i for i, f in enumerate(p.features) if f.aggregator == ft.AGGREGATOR_STDDEV]
        self.assertTrue(np.all(x[ind] == 0), f'{x[ind]} should have been 0, it is the stddev')

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
        p = ProfileNative([fg])
        for amount, date, flt in zip(df[fa.name], df[fd.name], df[ff.name]):
            p.contribute(([amount], date, [flt]))
        x = p.list(df[fd.name].iloc[-1])
        # Filter the input, this should give the same result as the profile. (If the profile applies the filter)
        y = df[df['Filter'] == 1].to_numpy()
        self.assertEqual(x[0], len(y[:, 0]), f'Counts do not match {x[0]} {len(y[:, 0])} ')

    def test_all_zero(self):
        # Let's repeat and see if we apply a filter that is always False, that should return all zero
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
        p = ProfileNative([fg])
        for amount, date, flt in zip(df[fa.name], df[fd.name], df[ff.name]):
            p.contribute(([amount], date, [flt]))
        x = p.list(df[fd.name].iloc[-1])
        self.assertEqual(np.count_nonzero(x), 0, f'Array should have return all zeros {x}')

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
        p = ProfileNative([fg1, fg2])
        for amount1, amount2, date in zip(df[fa1.name], df[fa2.name], df[fd.name]):
            p.contribute(([amount1, amount2], date, []))
        x = p.list(df[fa1.name].iloc[-1])
        af1 = df[fa1.name].to_numpy()
        af2 = df[fa2.name].to_numpy()
        self.assertEqual(x[0], sum(af1), f'sum1 does not match {x[0]} {sum(af1)} ')
        self.assertEqual(x[1], sum(af2), f'sum2 does not match {x[1]} {sum(af2)} ')

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
        p = ProfileNative([fg1, fg2])
        for amount, date, flt1, flt2 in zip(df[fa.name], df[fd.name], df[ff1.name], df[ff2.name]):
            p.contribute(([amount], date, [flt1, flt2]))
        x = p.list(df[fd.name].iloc[-1])
        # Filter the input, this should give the same result as the profile. (If the profile applies the filter)
        y = df[df['Filter1'] == 1].to_numpy()
        self.assertEqual(x[0], len(y[:, 0]), f'Counts do not match {x[0]} {len(y[:, 0])} ')
        y = df[df['Filter2'] == 1].to_numpy()
        self.assertEqual(x[1], len(y[:, 0]), f'Counts do not match {x[1]} {len(y[:, 0])} ')

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
        p = ProfileNative([fg])
        c = 0
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for amount, date in zip(df[fa.name], df[fd.name]):
            p.contribute(([amount], date, []))
            d = tp.start_period(date)
            if tp.delta_between(start_date, d) == 0:
                c += 1
            else:
                # week changed, count should have been reset.
                c = 1
                start_date = d
            lr = df[fd.name].iloc[-1]
            self.assertEqual(p.list(lr)[0], float(c), f'Counts do not match {p.list(lr)[0]} {float(c)}')

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
        p = ProfileNative([fg])
        c = 0
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for amount, date in zip(df[fa.name], df[fd.name]):
            p.contribute(([amount], date, []))
            d = tp.start_period(date)
            if tp.delta_between(start_date, d) == 0:
                c += 1
            else:
                # Month changed, count should have been reset.
                c = 1
                start_date = d
            lr = df[fd.name].iloc[-1]
            self.assertEqual(p.list(lr)[0], float(c), f'Counts do not match {p.list(lr)[0]} {float(c)}')

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
        p = ProfileNative([fgc5, fgc2])
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for i, (amount, date) in enumerate(zip(df[fa.name], df[fd.name])):
            p.contribute(([amount], date, []))
            d = tp.start_period(date)
            g = tp.delta_between(start_date, d) - 1 if i > 1 else 0
            x = p.list(df[fd.name].iloc[-1])
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
        fgc2 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_COUNT)
        fgc1 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=1, day=6)  # This is a Monday, count should reset
        p = ProfileNative([fgc2, fgc1])
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for i, (amount, date) in enumerate(zip(df[fa.name], df[fd.name])):
            p.contribute(([amount], date, []))
            d = tp.start_period(date)
            g = i+1 if tp.delta_between(start_date, d) == 0 else 1
            x = p.list(df[fd.name].iloc[-1])
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
        fgc2 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 2, ft.AGGREGATOR_COUNT)
        fgc1 = ft.FeatureGrouper('C', ft.FEATURE_TYPE_FLOAT, fa, fc, None, tp, 1, ft.AGGREGATOR_COUNT)
        td = ft.TensorDefinition('Source', [fa, fc, fd])
        with en.EnginePandasNumpy(num_threads=threads) as e:
            df = e.from_csv(td, file, inference=False)
        df.at[4, 'Date'] = pd.Timestamp(year=2020, month=2, day=1)
        p = ProfileNative([fgc2, fgc1])
        start_date = tp.start_period(df.iloc[0]['Date'].to_pydatetime())
        for i, (amount, date) in enumerate(zip(df[fa.name], df[fd.name])):
            p.contribute(([amount], date, []))
            d = tp.start_period(date)
            g = i+1 if tp.delta_between(start_date, d) == 0 else 1
            x = p.list(df[fd.name].iloc[-1])
            self.assertEqual(x[0], i+1, f'2 month counts do not match {x[0]} {i+1}')
            self.assertEqual(x[1], g, f'1 month counts do not match {x[1]} {g}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
