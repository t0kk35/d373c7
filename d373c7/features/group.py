"""
Definition of grouped features.
(c) 2021 d373c7
"""
import logging
from abc import ABC, abstractmethod
from datetime import timedelta, datetime
from dataclasses import dataclass, field

from .common import LearningCategory
from ..common import enforce_types
from ..features.common import Feature, FeatureTypeString, FeatureWithBaseFeature, FeatureHelper
from ..features.common import FeatureDefinitionException
from ..features.expressions import FeatureFilter
from typing import Optional

logger = logging.getLogger(__name__)


@enforce_types
@dataclass(frozen=True)
class TimePeriod(ABC):
    key: int = field(repr=False)
    name: str
    pandas_window: str = field(repr=False)
    numpy_window: str = field(repr=False)
    datetime_window: str = field(repr=False)

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
@dataclass(frozen=True)
class Aggregator:
    key: int = field(repr=False)
    name: str
    panda_agg_func: str = field(repr=False)


AGGREGATOR_SUM = Aggregator(0, 'Sum', 'sum')
AGGREGATOR_COUNT = Aggregator(1, 'Count', 'count')
AGGREGATOR_MIN = Aggregator(2, 'Minimum', 'min')
AGGREGATOR_MAX = Aggregator(3, 'Maximum', 'max')
AGGREGATOR_AVG = Aggregator(4, 'Average', 'mean')
AGGREGATOR_STDDEV = Aggregator(5, 'Standard Deviation', 'std')


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureGrouper(FeatureWithBaseFeature):
    group_feature: Feature
    dimension_feature: Optional[Feature] = field(compare=False)
    filter_feature: Optional[FeatureFilter] = field(compare=False)
    time_period: TimePeriod
    time_window: int
    aggregator: Aggregator

    def __post_init__(self):
        # Make sure the type float based.
        self.val_float_type()
        self.val_base_feature_is_float()
        self._val_dimension_feature_is_str()
        # Embedded features are the base_feature, the group feature, the filter (if set) and the dimension
        # feature (if set) + their embedded features.
        eb = [self.group_feature, self.base_feature]
        eb.extend(self.group_feature.embedded_features + self.base_feature.embedded_features)
        for f in (self.filter_feature, self.dimension_feature):
            if f is not None:
                eb.append(f)
                eb.extend(f.embedded_features)
        self.embedded_features = list(set(eb))

    def _val_dimension_feature_is_str(self) -> None:
        """
        Validation method to check if the dimension_feature is of type str.
        Dimension features must be of type string as the profiling will build a dict with this feature as key
        Will throw a FeatureDefinitionException if the base feature is NOT a float.

        @return: None
        """
        if self.dimension_feature is not None and \
                not FeatureHelper.is_feature_of_type(self.dimension_feature, FeatureTypeString):
            raise FeatureDefinitionException(
                f'Dimension feature of a {self.__class__.__name__} must be a string type. ' +
                f'Got <{type(self.dimension_feature.type)}>'
            )

    @property
    def inference_ready(self) -> bool:
        # This feature is inference ready if all its embedded features are ready for inference.
        return all([f.inference_ready for f in self.embedded_features])

    @property
    def learning_category(self) -> LearningCategory:
        return self.type.learning_category
