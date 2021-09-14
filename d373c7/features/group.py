"""
Definition of grouped features.
(c) 2021 d373c7
"""
import logging
from datetime import timedelta
from dataclasses import dataclass

from .common import LearningCategory
from ..common import enforce_types
from ..features.common import Feature, FeatureWithBaseFeature
from ..features.expressions import FeatureFilter
from typing import Optional

logger = logging.getLogger(__name__)


@enforce_types
@dataclass(frozen=True)
class TimePeriod:
    key: int
    name: str
    pandas_window: str
    numpy_window: str
    datetime_window: str

    def time_delta(self, number_of_periods: int):
        kw = {self.datetime_window: number_of_periods}
        return timedelta(**kw)


TIME_PERIOD_DAY = TimePeriod(0, 'Day', 'd', 'D', 'd')
TIME_PERIOD_WEEK = TimePeriod(1, 'Week', 'w', 'W', 'w')
TIME_PERIOD_MONTH = TimePeriod(2, 'Month', 'm', 'M', 'm')

TIME_PERIODS = [
    TIME_PERIOD_DAY,
    TIME_PERIOD_WEEK,
    TIME_PERIOD_MONTH
]


@enforce_types
@dataclass(frozen=True)
class Aggregator:
    key: int
    name: str
    panda_agg_func: str


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
    filter_feature: Optional[FeatureFilter]
    time_period: TimePeriod
    time_window: int
    aggregator: Aggregator

    def __post_init__(self):
        # Make sure the type float based.
        self.val_float_type()
        self.val_base_feature_is_float()
        # Embedded features are the base_feature, the group feature and the filter (if set) + their embedded features.
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
