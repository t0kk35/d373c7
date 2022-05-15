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
from typing import List, Tuple, TypeVar, Generic, Dict, Callable, Optional
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
    def aggregate(self, feature: Optional[FeatureGrouper]) -> AO:
        pass

    @abstractmethod
    def merge(self, pe: 'ProfileElement[AI, AO]'):
        pass


class ProfileField(Generic[IN, OUT], ABC):
    @abstractmethod
    def contribute(self, current_time: dt.datetime):
        pass

    @abstractmethod
    def run_time_logic(self, current_time: dt.datetime):
        pass


class Profile(Generic[IN, OUT], ABC):
    """
    Base class for Profiles. It has a generic input (used as parameter for the contribution) and an output (used as
    output for the list method)
    """
    def __init__(self, features: List[FeatureGrouper]):
        self.features = features

    def contribute(self, contribution: IN):
        """
        Method to used to contribute data to the profile.

        Args:
            contribution: (type IN) Data to contribute data to a profile

        Returns: None
        """
        for pf in self.profile_fields:
            pf.run_time_logic(self.extract_date_time(contribution))
            pf.contribute(contribution)

    @property
    def base_features(self) -> List[Feature]:
        """
        Property which return the base features that are needed to contribute to the profile, it is the unique list
        of base_features contained in the FeatureGroupers to

        Returns: (List[Feature]) Unique list of base features.
        """
        lst = [f.base_feature for f in self.features]
        # Not using set here to make unique because we want to keep the order
        return list(dict.fromkeys(lst))

    @property
    def filter_features(self) -> List[FeatureFilter]:
        """
        Property which return the filter features that are needed to contribute to the profile, it is the unique list
        of filter_features contained in the FeatureGroupers to

        Returns: (List[Feature]) Unique list of filter features.
        """
        lst = [f.filter_feature for f in self.features if f.filter_feature is not None]
        # Not using set here to make unique because we want to keep the order
        return list(dict.fromkeys(lst))

    @property
    @abstractmethod
    def profile_fields(self) -> List[ProfileField[IN, OUT]]:
        pass

    @abstractmethod
    def extract_date_time(self, contribution: IN) -> dt.datetime:
        """
        Method to extract the date from the generic contribution.

        Args:
            contribution: (IN Type) The contribution to perform on the profile

        Returns: (datetime) The data time from the contribution
        """
        pass

    @abstractmethod
    def list(self, current_time: dt.datetime) -> OUT:
        """
        Lists the aggregators within the profile for all the FeatureGroupers

        Args:
            current_time: (datetime) DateTime at which to get the list

        Returns: (OUT Type) An object of type OUT containing all the aggregations of the FeatureGroupers
        """
        pass


class ProfileFieldBase(ProfileField[AI, AO], Generic[AI, AO], ABC):
    def __init__(self):
        self.start_period_time = None

    @staticmethod
    def find_index(f: Feature, feature_list: List[Feature]) -> int:
        """
        Static Method which returns the index of a feature in a list. -1 is returned if the feature is not found.

        Args:
            f: (Feature). The feature to look up.
            feature_list: (List[Feature]). A list of features to use a lookup list.

        Returns: (int). Index of f in feature_list
        """
        if f is None:
            return -1
        else:
            try:
                return feature_list.index(f)
            except ValueError:
                raise ProfileException(
                    f'Could not find index of {f.name} in list {[f.name for f in feature_list]}'
                )


class ProfileFieldNative(ProfileFieldBase[Tuple[List[float], dt.datetime, List[int]], List[float]], Generic[AI, AO]):
    def __init__(
            self, time_period: TimePeriod, time_window: int, base_feature: Feature, base_order: List[Feature],
            filter_feature: FeatureFilter, filter_order: List[FeatureFilter]):
        super(ProfileFieldNative, self).__init__()
        self.base_index = self.find_index(base_feature, base_order)
        self._filter_index = self.find_index(filter_feature, filter_order)
        self._time_period = time_period
        self._time_window = time_window
        self._time_slots = deque([ProfileElementNative() for _ in range(time_window)], maxlen=time_window)

    def contribute(self, contribution: Tuple[List[float], dt.datetime, List[int]]):
        # Only contribute if filter is True or None
        _, _, filters = contribution
        if self._filter_index == -1 or filters[self._filter_index] == 1:
            self._time_slots[self._time_window-1].contribute(self.extract_contribution(contribution))

    def run_time_logic(self, current_time: dt.datetime):
        stp = self._time_period.start_period(current_time)
        previous_time = self.start_period_time if self.start_period_time is not None else stp
        self.start_period_time = stp
        delta = self._time_period.delta_between(previous_time, stp)
        self._time_slots.extend([ProfileElementNative() for _ in range(delta)])

    def merged_profile_element(self, feature: FeatureGrouper) -> ProfileElement[AI, AO]:
        pf = ProfileElementNative()
        for i, e in enumerate(reversed(self._time_slots)):
            # Only go back the time window of the current feature, as we keep max time windows for each time period,
            # there could be more time slots than the requested feature time window
            if i < feature.time_window:
                pf.merge(e)
        return pf

    # @abstractmethod
    # def create_element(self) -> ProfileElement[AI, AO]:
    #     return ProfileElementNative()

    def extract_contribution(self, contribution: Tuple[List[float], dt.datetime, List[int]]) -> float:
        amount, _, _ = contribution
        return amount[self.base_index]


class ProfileNative(Profile[Tuple[List[float], dt.datetime, List[int]], List[float]]):
    def __init__(self, features: List[FeatureGrouper]):
        super(ProfileNative, self).__init__(features)
        # Make a dict that keeps track of which feature is stored in which field.
        # We only need 1 profile field per unique base_feature, filter_feature and time period. It will calculate
        # all the aggregates for the max time window
        uniq_f: Dict[Tuple[Feature, FeatureFilter, TimePeriod], List[FeatureGrouper]] = {
            k: list(sorted(v, key=lambda x: x)) for k, v in
            groupby(features, lambda x: (x.base_feature, x.filter_feature, x.time_period))
        }
        self._profile_fields: List[ProfileField[Tuple[List[float], dt.datetime, List[int]], List[float]]] = []
        self._features_to_field: Dict[FeatureGrouper, ProfileFieldNative] = {}
        for (bf, ff, tp), v in uniq_f.items():
            tw = max([f.time_window for f in v])
            pf = ProfileFieldNative(tp, tw, bf, self.base_features, ff, self.filter_features)
            self._profile_fields.append(pf)
            self._features_to_field.update({f: pf for f in v})

    @property
    def profile_fields(self) -> List[ProfileField[Tuple[List[float], dt.datetime, List[int]], List[float]]]:
        return self._profile_fields

    def extract_date_time(self, contribution: Tuple[List[float], dt.datetime, List[int]]) -> dt.datetime:
        _, cdt, _ = contribution
        return cdt

    def _get_and_aggregate(self, f: FeatureGrouper) -> float:
        pf = self._features_to_field[f]
        pe = pf.merged_profile_element(f)
        return pe.aggregate(f)

    def list(self, current_time: dt.datetime) -> List[float]:
        # TODO Run TimeLogic
        return [self._get_and_aggregate(f) for f in self.features]


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

    def aggregate(self, feature: Optional[FeatureGrouper]) -> float:
        if feature is None:
            raise ProfileException(
                f'Aggregate for NativeProfile element should receive single element of FeatureGrouper. Got None'
            )
        return PROFILE_AGG_HELPER.aggregate(self, feature.aggregator)

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
