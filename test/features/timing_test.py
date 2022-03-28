"""
Unit Tests for the timing functions
(c) 2022 d373c7
"""
import unittest
from datetime import datetime
import d373c7.features as ft


class LearningCategoriesTest(unittest.TestCase):
    def test_timeperiod_daily_delta(self):
        tp = ft.TIME_PERIOD_DAY
        t1 = datetime(2022, 1, 1)
        d = tp.delta_between(t1, t1)
        self.assertEqual(d, 0, f'Delta should have been 0. Was {d}')
        t2 = datetime(2022, 1, 2)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 1, f'Delta should have been one. Was {d}')
        t2 = datetime(2022, 2, 1)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 31, f'Delta should have been 31. Was {d}')
        t2 = datetime(2023, 1, 1)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 365, f'Delta should have been 365. Was {d}')

    def test_timeperiod_daily_start_period(self):
        tp = ft.TIME_PERIOD_DAY
        t1 = datetime(2022, 1, 1, hour=12, minute=5, second=30)
        d = tp.start_period(t1)
        self.assertEqual(d.year, t1.year, f'Years should have been the same. {t1.year} {d.year}')
        self.assertEqual(d.month, t1.month, f'Months should have been the same. {t1.month} {d.month}')
        self.assertEqual(d.day, t1.day, f'Months should have been the same. {t1.day} {d.day}')
        self.assertEqual(d.hour, 0, f'Hour should have been 0. {t1.hour} {d.hour}')
        self.assertEqual(d.minute, 0, f'Minute should have been 0. {t1.minute} {d.min}')
        self.assertEqual(d.second, 0, f'Second should have been 0. {t1.second} {d.second}')
        self.assertEqual(d.microsecond, 0, f'Mirco seconds should have been 0. {t1.microsecond} {d.microsecond}')

    def test_timeperiod_week(self):
        tp = ft.TIME_PERIOD_WEEK
        t1 = datetime(2022, 1, 1)
        d = tp.delta_between(t1, t1)
        self.assertEqual(d, 0, f'Delta should have been 0. Was {d}')
        t2 = datetime(2022, 1, 7)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 0, f'Delta should have been 0. Was {d}')
        t2 = datetime(2022, 1, 8)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 1, f'Delta should have been 1. Was {d}')
        t2 = datetime(2022, 1, 14)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 1, f'Delta should have been 1. Was {d}')
        t2 = datetime(2022, 1, 15)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 2, f'Delta should have been 2. Was {d}')
        t2 = datetime(2022, 2, 1)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 4, f'Delta should have been 4. Was {d}')
        t2 = datetime(2022, 2, 5)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 5, f'Delta should have been 5. Was {d}')
        t2 = datetime(2022, 12, 25)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 51, f'Delta should have been 51. Was {d}')
        t2 = datetime(2023, 1, 1)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 52, f'Delta should have been 52. Was {d}')

    def test_timeperiod_week_start_period(self):
        tp = ft.TIME_PERIOD_WEEK
        t1 = datetime(2022, 1, 1, hour=12, minute=5, second=30)
        d = tp.start_period(t1)
        self.assertEqual(d.year, t1.year-1, f'Years should have gone back -1. {t1.year} {d.year}')
        self.assertEqual(d.month, 12, f'Months should have been 12. {d.month}')
        self.assertEqual(d.weekday(), 0, f'Weekday should have been 0. {d.weekday()}')
        self.assertEqual(d.hour, 0, f'Hour should have been 0. {t1.hour} {d.hour}')
        self.assertEqual(d.minute, 0, f'Minute should have been 0. {t1.minute} {d.min}')
        self.assertEqual(d.second, 0, f'Second should have been 0. {t1.second} {d.second}')
        self.assertEqual(d.microsecond, 0, f'Mirco seconds should have been 0. {t1.microsecond} {d.microsecond}')
        # This is a Sunday
        t1 = datetime(2022, 1, 2, hour=12, minute=5, second=30)
        d = tp.start_period(t1)
        self.assertEqual(d.year, t1.year-1, f'Years should have gone back -1. {t1.year} {d.year}')
        self.assertEqual(d.month, 12, f'Months should have been 12. {d.month}')
        self.assertEqual(d.weekday(), 0, f'Weekday should have been 0. {d.weekday()}')
        self.assertEqual(d.hour, 0, f'Hour should have been 0. {t1.hour} {d.hour}')
        self.assertEqual(d.minute, 0, f'Minute should have been 0. {t1.minute} {d.min}')
        self.assertEqual(d.second, 0, f'Second should have been 0. {t1.second} {d.second}')
        self.assertEqual(d.microsecond, 0, f'Mirco seconds should have been 0. {t1.microsecond} {d.microsecond}')
        # This is a Monday, so day should not change
        t1 = datetime(2022, 1, 3, hour=12, minute=5, second=30)
        d = tp.start_period(t1)
        self.assertEqual(d.year, t1.year, f'Years should have been the same. {t1.year} {d.year}')
        self.assertEqual(d.month, t1.month, f'Months should have been the same. {t1.month} {d.month}')
        self.assertEqual(d.day, t1.day, f'Days should have been the same. {t1.day} {d.day}')
        self.assertEqual(d.weekday(), 0, f'Weekday should have been 0. {d.weekday()}')
        self.assertEqual(d.hour, 0, f'Hour should have been 0. {t1.hour} {d.hour}')
        self.assertEqual(d.minute, 0, f'Minute should have been 0. {t1.minute} {d.min}')
        self.assertEqual(d.second, 0, f'Second should have been 0. {t1.second} {d.second}')
        self.assertEqual(d.microsecond, 0, f'Mirco seconds should have been 0. {t1.microsecond} {d.microsecond}')

    def test_timeperiod_month(self):
        tp = ft.TIME_PERIOD_MONTH
        t1 = datetime(2022, 1, 1)
        d = tp.delta_between(t1, t1)
        self.assertEqual(d, 0, f'Delta should have been 0. Was {d}')
        t2 = datetime(2022, 1, 31)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 0, f'Delta should have been 0. Was {d}')
        t2 = datetime(2022, 2, 1)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 1, f'Delta should have been 1. Was {d}')
        t2 = datetime(2022, 2, 10)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 1, f'Delta should have been 1. Was {d}')
        t2 = datetime(2022, 3, 1)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 2, f'Delta should have been 2. Was {d}')
        t2 = datetime(2022, 12, 31)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 11, f'Delta should have been 11. Was {d}')
        t2 = datetime(2023, 1, 1)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 12, f'Delta should have been 12. Was {d}')
        t2 = datetime(2023, 1, 31)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 12, f'Delta should have been 12. Was {d}')
        t2 = datetime(2023, 2, 1)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 13, f'Delta should have been 13. Was {d}')
        t2 = datetime(2023, 12, 31)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 23, f'Delta should have been 23. Was {d}')
        t2 = datetime(2024, 1, 1)
        d = tp.delta_between(t1, t2)
        self.assertEqual(d, 24, f'Delta should have been 24. Was {d}')

    def test_timeperiod_month_start_period(self):
        tp = ft.TIME_PERIOD_MONTH
        t1 = datetime(2022, 1, 1, hour=12, minute=5, second=30)
        d = tp.start_period(t1)
        self.assertEqual(d.year, t1.year, f'Years should have been the same. {t1.year} {d.year}')
        self.assertEqual(d.month, t1.month, f'Months should have been the same. {t1.month} {d.month}')
        self.assertEqual(d.day, t1.day, f'Months should have been the same. {t1.day} {d.day}')
        self.assertEqual(d.hour, 0, f'Hour should have been 0. {t1.hour} {d.hour}')
        self.assertEqual(d.minute, 0, f'Minute should have been 0. {t1.minute} {d.min}')
        self.assertEqual(d.second, 0, f'Second should have been 0. {t1.second} {d.second}')
        self.assertEqual(d.microsecond, 0, f'Mirco seconds should have been 0. {t1.microsecond} {d.microsecond}')
        t1 = datetime(2022, 1, 31, hour=12, minute=5, second=30)
        d = tp.start_period(t1)
        self.assertEqual(d.year, t1.year, f'Years should have been the same. {t1.year} {d.year}')
        self.assertEqual(d.month, t1.month, f'Months should have been the same. {t1.month} {d.month}')
        self.assertEqual(d.day, 1, f'Months should have been 1. {d.day}')
        self.assertEqual(d.hour, 0, f'Hour should have been 0. {t1.hour} {d.hour}')
        self.assertEqual(d.minute, 0, f'Minute should have been 0. {t1.minute} {d.min}')
        self.assertEqual(d.second, 0, f'Second should have been 0. {t1.second} {d.second}')
        self.assertEqual(d.microsecond, 0, f'Mirco seconds should have been 0. {t1.microsecond} {d.microsecond}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
