"""
Definition of grouped features.
(c) 2021 d373c7
"""
import logging
from abc import ABC, abstractmethod
from datetime import timedelta, datetime
from dataclasses import dataclass, field
from functools import total_ordering

from .common import LearningCategory
from ..common import enforce_types
from ..features.common import Feature, FeatureWithBaseFeature
from ..features.expressions import FeatureFilter
from typing import Optional

logger = logging.getLogger(__name__)


@enforce_types
@total_ordering
@dataclass(frozen=True, eq=False)
class TimePeriod(ABC):
    key: int = field(repr=False)
    name: str = field(compare=False)
    pandas_window: str = field(repr=False, compare=False)
    numpy_window: str = field(repr=False, compare=False)
    datetime_window: str = field(repr=False, compare=False)

    def __eq__(self, other):
        if isinstance(other, TimePeriod):
            return self.key == other.key
        else:
            raise TypeError(
                f'Can not check equality of TimePeriod object and non-TimePeriod object Got a {type(other)}'
            )

    def __lt__(self, other):
        if isinstance(other, TimePeriod):
            return self.key < other.key
        else:
            raise TypeError(
                f'Can not do < of TimePeriod object and non-TimePeriod object Got a {type(other)}'
            )

    def time_delta(self, number_of_periods: int):
        kw = {self.datetime_window: number_of_periods}
        return timedelta(**kw)

    @abstractmethod
    def delta_between(self, dt1: datetime, dt2: datetime) -> int:
        pass

    @abstractmethod
    def start_period(self, d: datetime) -> datetime:
        pass


@enforce_types
@dataclass(frozen=True)
class TimePeriodDay(TimePeriod):
    def delta_between(self, dt1: datetime, dt2: datetime) -> int:
        return (dt2 - dt1).days

    def start_period(self, d: datetime) -> datetime:
        # Remove time part
        return datetime(year=d.year, month=d.month, day=d.day)


@enforce_types
@dataclass(frozen=True)
class TimePeriodWeek(TimePeriod):
    def delta_between(self, dt1: datetime, dt2: datetime) -> int:
        return (dt2 - dt1).days // 7

    def start_period(self, d: datetime) -> datetime:
        # Go back to previous monday
        r = TIME_PERIOD_DAY.start_period(d)
        return r - timedelta(days=r.weekday())


@enforce_types
@dataclass(frozen=True)
class TimePeriodMonth(TimePeriod):
    def delta_between(self, dt1: datetime, dt2: datetime) -> int:
        return (dt2.year - dt1.year) * 12 + dt2.month - dt1.month

    def start_period(self, d: datetime) -> datetime:
        # Remove time and go to first day of month
        return datetime(year=d.year, month=d.month, day=1)


TIME_PERIOD_DAY = TimePeriodDay(0, 'Day', 'd', 'D', 'd')
TIME_PERIOD_WEEK = TimePeriodWeek(1, 'Week', 'w', 'W', 'w')
TIME_PERIOD_MONTH = TimePeriodMonth(2, 'Month', 'm', 'M', 'm')

TIME_PERIODS = [
    TIME_PERIOD_DAY,
    TIME_PERIOD_WEEK,
    TIME_PERIOD_MONTH
]


@enforce_types
@dataclass(frozen=True, order=True)
class Aggregator:
    key: int = field(repr=False)
    name: str = field(compare=False)
    panda_agg_func: str = field(repr=False, compare=False)


AGGREGATOR_SUM = Aggregator(0, 'Sum', 'sum')
AGGREGATOR_COUNT = Aggregator(1, 'Count', 'count')
AGGREGATOR_MIN = Aggregator(2, 'Minimum', 'min')
AGGREGATOR_MAX = Aggregator(3, 'Maximum', 'max')
AGGREGATOR_AVG = Aggregator(4, 'Average', 'mean')
AGGREGATOR_STDDEV = Aggregator(5, 'Standard Deviation', 'std')


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureGrouper(FeatureWithBaseFeature):
    group_feature: Feature
    filter_feature: Optional[FeatureFilter] = field(compare=False)
    time_period: TimePeriod
    time_window: int
    aggregator: Aggregator

    def __post_init__(self):
        # Make sure the type float based.
        self.val_float_type()
        self.val_base_feature_is_float()
        # Embedded features are the base_feature, the group feature, the filter (if set) + their embedded features.
        eb = [self.group_feature, self.base_feature]
        eb.extend(self.group_feature.embedded_features + self.base_feature.embedded_features)
        if self.filter_feature is not None:
            eb.append(self.filter_feature)
            eb.extend(self.filter_feature.embedded_features)
        self.embedded_features = list(set(eb))

    @property
    def inference_ready(self) -> bool:
        # This feature is inference ready if all its embedded features are ready for inference.
        return all([f.inference_ready for f in self.embedded_features])

    @property
    def learning_category(self) -> LearningCategory:
        return self.type.learning_category
