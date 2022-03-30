"""
Profile Base definitions
(c) 2022 d373c7
"""
import logging

import numpy as np
import datetime as dt

from abc import ABC, abstractmethod
from ..features.group import FeatureGrouper, Aggregator, TimePeriod
from ..features.common import Feature
from ..features.tensor import TensorDefinition
from ..features.common import FeatureHelper, FeatureTypeTimeBased
from collections import deque
from math import sqrt
from typing import List, Tuple, TypeVar, Generic, Dict, Callable
from itertools import groupby
logger = logging.getLogger(__name__)


class ProfileException(Exception):
    def __init__(self, message: str):
        super().__init__('Error profiling: ' + message)


IN = TypeVar('IN')  # Type for the input of the contribution. This is what goes into the profile
OUT = TypeVar('OUT')  # Type for the output of the profile. What comes out of the profile
AI = TypeVar('AI')  # Type for the input to the Aggregators.
AO = TypeVar('AO')  # Type for the output of the Aggregators.


class ProfileElement(Generic[AI, AO], ABC):
    @abstractmethod
    def contribute(self, contribution: AI):
        pass

    @abstractmethod
    def aggregate(self, contribution: AI, aggregator: Aggregator) -> AO:
        pass

    @abstractmethod
    def merge(self, pe: 'ProfileElement[AI, AO]') -> 'ProfileElement[AI, AO]':
        pass


class ProfileField(Generic[IN, OUT], ABC):
    @abstractmethod
    def contribute(self, contribution: IN):
        pass

    @abstractmethod
    def run_time_logic(self, current_time: dt.datetime):
        pass


class Profile(Generic[IN, OUT], ABC):
    def __init__(self, features: List[FeatureGrouper]):
        self.features = features

    def contribute(self, contribution: IN):
        for pf in self.profile_fields:
            pf.run_time_logic(self.extract_date_time(contribution))
            pf.contribute(contribution)

    @property
    @abstractmethod
    def profile_fields(self) -> List[ProfileField[IN, OUT]]:
        pass

    @abstractmethod
    def extract_date_time(self, contribution: IN) -> dt.datetime:
        pass

    @abstractmethod
    def list(self, contribution: IN) -> OUT:
        pass

    @staticmethod
    def feature_index(feature: Feature, contrib_tensor_definition: TensorDefinition) -> int:
        if feature is None:
            return -1
        try:
            index = contrib_tensor_definition.features.index(feature)
        except ValueError:
            raise ProfileException(
                f'Could not find feature {feature.name} ' +
                f'in Tensor Definition {contrib_tensor_definition.name}'
            )
        return index


class ProfileFieldNativeBase(ProfileField[np.ndarray, np.ndarray], Generic[AI, AO], ABC):
    def __init__(self, base_feature: Feature, filter_feature: Feature, time_period: TimePeriod,
                 time_window: int, contrib_tensor_definition: TensorDefinition):
        self.base_feature_index = Profile.feature_index(base_feature, contrib_tensor_definition)
        self._filter_feature_index = Profile.feature_index(filter_feature, contrib_tensor_definition)
        self._time_slots = deque([self.create_element() for _ in range(time_window)], maxlen=time_window)
        self._time_period = time_period
        self._time_window = time_window
        self._start_period_time = None

    def contribute(self, contribution: np.ndarray):
        # Only contribute if filter is True or None
        if self._filter_feature_index == -1 or contribution[self._filter_feature_index] == 1:
            self._time_slots[self._time_window-1].contribute(self.extract_contribution(contribution))

    def run_time_logic(self, current_time: dt.datetime):
        stp = self._time_period.start_period(current_time)
        previous_time = self._start_period_time if self._start_period_time is not None else stp
        self._start_period_time = stp
        delta = self._time_period.delta_between(previous_time, stp)
        self._time_slots.extend([self.create_element() for _ in range(delta)])

    def merged_profile_element(self, feature: FeatureGrouper) -> ProfileElement[AI, AO]:
        pf = self.create_element()
        for i, e in enumerate(reversed(self._time_slots)):
            # Only go back the time window of the current feature, as we keep max time windows for each time period,
            # there could be more time slots than the requested feature time window
            if i < feature.time_window:
                pf.merge(e)
        return pf

    @abstractmethod
    def create_element(self) -> ProfileElement[AI, AO]:
        pass

    @abstractmethod
    def extract_contribution(self, contribution: np.ndarray) -> AI:
        pass


class ProfileFieldNative(ProfileFieldNativeBase[float, float]):
    def __init__(self, base_feature: Feature, filter_feature: Feature, time_period: TimePeriod,
                 time_window: int, contrib_tensor_definition: TensorDefinition):
        super(ProfileFieldNative, self).__init__(
            base_feature, filter_feature, time_period, time_window, contrib_tensor_definition
        )

    def create_element(self) -> ProfileElement[float, float]:
        return ProfileElementNative()

    def extract_contribution(self, contribution: np.ndarray) -> float:
        return contribution[self.base_feature_index]


class ProfileFieldNativeDict(ProfileFieldNativeBase[Tuple[str, float], float]):
    def __init__(self, base_feature: Feature, filter_feature: Feature, dimension_feature: Feature,
                 time_period: TimePeriod, time_window: int, contrib_tensor_definition: TensorDefinition):
        super(ProfileFieldNativeDict, self).__init__(
            base_feature, filter_feature, time_period, time_window, contrib_tensor_definition
        )
        self._dimension_feature_index = Profile.feature_index(dimension_feature, contrib_tensor_definition)

    def create_element(self) -> ProfileElement[Tuple[str, float], float]:
        return ProfileElementNativeDict()

    def extract_contribution(self, contribution: np.ndarray) -> Tuple[str, float]:
        return contribution[(self._dimension_feature_index, self.base_feature_index)]


class ProfileNative(Profile[np.ndarray, np.ndarray]):
    def __init__(self, features: List[FeatureGrouper], time_feature: Feature,
                 contrib_tensor_definition: TensorDefinition):
        super(ProfileNative, self).__init__(features)
        # Make a dict that keeps track of which feature is stored in which field.
        # We only need 1 profile field per unique base_feature, filter_feature and time period. It will calculate
        # all the aggregates for the max time window
        self._val_time_field_is_time_based(time_feature)
        self._time_feature_index = Profile.feature_index(time_feature, contrib_tensor_definition)
        uniq_f: Dict[Tuple[Feature, Feature, TimePeriod], List[FeatureGrouper]] = {
            k: list(v) for k, v in
            groupby(features, lambda x: (x.base_feature, x.filter_feature, x.time_period))
        }
        self._profile_fields: List[ProfileField[np.ndarray, np.ndarray]] = []
        self._features_to_field: Dict[FeatureGrouper, ProfileFieldNativeBase] = {}
        for (bf, ff, tp), v in uniq_f.items():
            tw = max([f.time_window for f in v])
            pf = ProfileFieldNative(bf, ff, tp, tw, contrib_tensor_definition)
            self._profile_fields.append(pf)
            for f in v:
                self._features_to_field[f] = pf

    def _val_time_field_is_time_based(self, feature: Feature):
        if not FeatureHelper.is_feature_of_type(feature, FeatureTypeTimeBased):
            raise ProfileException(
                f'The time-field to a {self.__class__.__name__} must be fime time based. Got {feature.name} which is ' +
                f'of type {feature.type.name}'
            )

    @property
    def profile_fields(self) -> List[ProfileField[np.ndarray, np.ndarray]]:
        return self._profile_fields

    def extract_date_time(self, contribution: np.ndarray) -> dt.datetime:
        cdt = contribution[self._time_feature_index]
        return cdt

    def _get_and_aggregate(self, f: FeatureGrouper, contribution: np.ndarray) -> float:
        pf = self._features_to_field[f]
        pe = pf.merged_profile_element(f)
        return pe.aggregate(pf.extract_contribution(contribution), f.aggregator)

    def list(self, contribution: np.ndarray) -> np.ndarray:
        out = [self._get_and_aggregate(f, contribution) for f in self.features]
        return np.array(out, dtype=float)


class ProfileElementNative(ProfileElement[float, float]):
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0
        self.min = 0
        self.max = 0

    def contribute(self, contribution: float):
        self.count += 1
        delta = contribution - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (contribution - self.mean)
        if contribution < self.min or self.count == 1:
            self.min = contribution
        if contribution > self.max:
            self.max = contribution

    def aggregate(self, contribution: float, aggregator: Aggregator) -> float:
        return PROFILE_AGG_HELPER.aggregate(self, aggregator)

    def merge(self, pe: 'ProfileElementNative') -> 'ProfileElementNative':
        if self.count == 0:
            self.count = pe.count
            self.mean = pe.mean
            self.M2 = pe.M2
            self.max = pe.max
            self.min = pe.min
        else:
            count = self.count + pe.count
            delta = self.mean - pe.mean
            delta2 = delta * delta
            self.mean = (self.count * self.mean + pe.count * pe.mean) / count
            self.M2 = self.M2 + pe.M2 + delta2 * (self.count * pe.count) / count
            self.count = count
            if pe.max > self.max:
                self.max = pe.max
            if pe.min < self.min and pe.count > 0:
                self.min = pe.min

        return self


class ProfileAggregatorHelper:
    def __init__(self):
        self._functions: List[Callable] = [
            self._sum,
            self._count,
            self._min,
            self._max,
            self._mean,
            self._stddev
        ]
        self._number_of_functions = len(self._functions)

    def aggregate(self, pe: ProfileElementNative, aggregator: Aggregator):
        if aggregator.key > self._number_of_functions:
            raise ProfileException(
                f'Risk of overflow, looking for aggregator with Key: {aggregator.key}. Probably using an unknown' +
                f' aggregator'
            )
        return self._functions[aggregator.key](pe)

    @staticmethod
    def _count(pe: ProfileElementNative) -> float:
        return pe.count

    @staticmethod
    def _sum(pe: ProfileElementNative) -> float:
        return pe.count * pe.mean

    @staticmethod
    def _mean(pe: ProfileElementNative) -> float:
        return pe.mean

    @staticmethod
    def _stddev(pe: ProfileElementNative) -> float:
        if pe.count < 2:
            return 0
        else:
            return sqrt(pe.M2 / (pe.count-1))

    @staticmethod
    def _min(pe: ProfileElementNative) -> float:
        return pe.min

    @staticmethod
    def _max(pe: ProfileElementNative) -> float:
        return pe.max


PROFILE_AGG_HELPER = ProfileAggregatorHelper()


class ProfileElementNativeDict(ProfileElement[Tuple[str, float], float]):
    def __init__(self):
        self._elements: Dict[str, ProfileElementNative] = {}

    def contribute(self, contribution: Tuple[str, float]):
        key, value = contribution
        self._elements.setdefault(key, ProfileElementNative()).contribute(value)

    def aggregate(self, contribution: Tuple[str, float], aggregator: Aggregator) -> float:
        key, value = contribution
        return self._elements.get(key, ProfileElementNative()).aggregate(value, aggregator)

    def merge(self, pe: 'ProfileElementNativeDict') -> 'ProfileElementNativeDict':
        r = {
            **self._elements,
            **pe._elements,
            **{k: self._elements[k].merge(pe._elements[k]) for k in self._elements.keys() & pe._elements.keys()}
        }
        self._elements = r
        return self
