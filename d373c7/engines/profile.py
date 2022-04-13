"""
Profile Base definitions
(c) 2022 d373c7
"""
import logging

import datetime as dt

from abc import ABC, abstractmethod
from ..features.group import FeatureGrouper, Aggregator, TimePeriod
from ..features.common import Feature
from ..features.expressions import FeatureFilter
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
    def merge(self, pe: 'ProfileElement[AI, AO]'):
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


class ProfileFieldNativeBase(ProfileField[
                                 Tuple[List[float], dt.datetime, List[int], List[str]], List[float]
                             ], Generic[AI, AO], ABC):
    def __init__(self, time_period: TimePeriod, time_window: int, base_feature: Feature, base_order: List[Feature],
                 filter_feature: FeatureFilter, filter_order: List[FeatureFilter]):
        self._time_slots = deque([self.create_element() for _ in range(time_window)], maxlen=time_window)
        self.base_index = self.find_index(base_feature, base_order)
        self._filter_index = self.find_index(filter_feature, filter_order)
        self._time_period = time_period
        self._time_window = time_window
        self._start_period_time = None

    def contribute(self, contribution: Tuple[List[float], dt.datetime, List[int], List[str]]):
        # Only contribute if filter is True or None
        _, _, filters, _ = contribution
        if self._filter_index == -1 or filters[self._filter_index] == 1:
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

    @staticmethod
    def find_index(f: Feature, feature_list: List[Feature]) -> int:
        if f is None:
            return -1
        else:
            try:
                return feature_list.index(f)
            except ValueError:
                raise ProfileException(
                    f'Could not find index of {f.name} in list {[f.name for f in feature_list]}'
                )

    @abstractmethod
    def create_element(self) -> ProfileElement[AI, AO]:
        pass

    @abstractmethod
    def extract_contribution(self, contribution: Tuple[List[float], dt.datetime, List[int], List[str]]) -> AI:
        pass


class ProfileFieldNative(ProfileFieldNativeBase[float, float]):
    def __init__(self, time_period: TimePeriod, time_window: int, base_feature: Feature, base_order: List[Feature],
                 filter_feature: FeatureFilter, filter_order: List[FeatureFilter]):
        super(ProfileFieldNative, self).__init__(time_period, time_window, base_feature, base_order,
                                                 filter_feature, filter_order)

    def create_element(self) -> ProfileElement[float, float]:
        return ProfileElementNative()

    def extract_contribution(self, contribution: Tuple[List[float], dt.datetime, List[int], List[str]]) -> float:
        amount, _, _, _ = contribution
        return amount[self.base_index]


class ProfileFieldNativeDict(ProfileFieldNativeBase[Tuple[str, float], float]):
    def __init__(self, time_period: TimePeriod, time_window: int, base_feature: Feature, base_order: List[Feature],
                 filter_feature: FeatureFilter, filter_order: List[FeatureFilter],
                 key_feature: Feature, key_order: List[Feature]):
        super(ProfileFieldNativeDict, self).__init__(time_period, time_window, base_feature, base_order,
                                                     filter_feature, filter_order)
        self._key_index = self.find_index(key_feature, key_order)

    def create_element(self) -> ProfileElement[Tuple[str, float], float]:
        return ProfileElementNativeDict()

    def extract_contribution(self, contribution: Tuple[List[float], dt.datetime, List[int], List[str]]) \
            -> Tuple[str, float]:
        amount, _, _, key = contribution
        return key[self._key_index], amount[self.base_index]


class ProfileNative(Profile[Tuple[List[float], dt.datetime, List[int], List[str]], List[float]]):
    def __init__(self, features: List[FeatureGrouper]):
        super(ProfileNative, self).__init__(features)
        # Make a dict that keeps track of which feature is stored in which field.
        # We only need 1 profile field per unique base_feature, filter_feature and time period. It will calculate
        # all the aggregates for the max time window
        uniq_f: Dict[Tuple[Feature, Feature, FeatureFilter, TimePeriod], List[FeatureGrouper]] = {
            k: list(v) for k, v in
            groupby(
                sorted(features, key=lambda x: x),
                lambda x: (x.base_feature, x.dimension_feature, x.filter_feature, x.time_period))
        }
        self._profile_fields: \
            List[ProfileField[Tuple[List[float], dt.datetime, List[int], List[str]], List[float]]] = []
        self._features_to_field: Dict[FeatureGrouper, ProfileFieldNativeBase] = {}
        for (bf, df, ff, tp), v in uniq_f.items():
            tw = max([f.time_window for f in v])
            if df is None:
                pf = ProfileFieldNative(tp, tw, bf, self.base_features, ff, self.filter_features)
            else:
                pf = ProfileFieldNativeDict(tp, tw, bf, self.base_features, ff, self.filter_features,
                                            df, self.dimension_features)
            self._profile_fields.append(pf)
            self._features_to_field.update({f: pf for f in v})

    @property
    def profile_fields(self) -> List[ProfileField[Tuple[List[float], dt.datetime, List[int], List[str]], List[float]]]:
        return self._profile_fields

    @property
    def base_features(self) -> List[Feature]:
        lst = [f.base_feature for f in self.features]
        # Not using set here to make unique because we want to keep the order
        return list(dict.fromkeys(lst))

    @property
    def filter_features(self) -> List[FeatureFilter]:
        lst = [f.filter_feature for f in self.features if f.filter_feature is not None]
        # Not using set here to make unique because we want to keep the order
        return list(dict.fromkeys(lst))

    @property
    def dimension_features(self) -> List[Feature]:
        lst = [f.dimension_feature for f in self.features if f.dimension_feature if f.dimension_feature is not None]
        # Not using set here to make unique because we want to keep the order
        return list(dict.fromkeys(lst))

    def extract_date_time(self, contribution: Tuple[List[float], dt.datetime, List[int], List[str]]) -> dt.datetime:
        _, cdt, _, _ = contribution
        return cdt

    def _get_and_aggregate(self, f: FeatureGrouper,
                           contribution: Tuple[List[float], dt.datetime, List[int], List[str]]) -> float:
        pf = self._features_to_field[f]
        pe = pf.merged_profile_element(f)
        return pe.aggregate(pf.extract_contribution(contribution), f.aggregator)

    def list(self, contribution: Tuple[List[float], dt.datetime, List[int], List[str]]) -> List[float]:
        return [self._get_and_aggregate(f, contribution) for f in self.features]


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

    @staticmethod
    def merge_new(pe1: 'ProfileElementNative', pe2: 'ProfileElementNative') -> 'ProfileElementNative':
        r = ProfileElementNative()
        r.merge(pe1)
        r.merge(pe2)
        return r

    def merge(self, pe: 'ProfileElementNative'):
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

    def merge(self, pe: 'ProfileElementNativeDict'):
        r = {
            **self._elements,
            **pe._elements,
            **{k: ProfileElementNative.merge_new(
                self._elements[k], pe._elements[k]
            ) for k in self._elements.keys() & pe._elements.keys()}
        }
        self._elements = r
        return self
