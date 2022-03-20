"""
Profile Base definitions
(c) 2022 d373c7
"""
import logging

import numpy as np

from abc import ABC, abstractmethod
from ..features.group import FeatureGrouper, Aggregator, TimePeriod
from ..features.common import FeatureHelper, Feature
from ..features.tensor import TensorDefinition
from math import sqrt
from typing import List, Tuple, TypeVar, Generic, Optional, Dict
from itertools import groupby
logger = logging.getLogger(__name__)


class ProfileException(Exception):
    def __init__(self, message: str):
        super().__init__('Error profiling: ' + message)


IN = TypeVar('IN')  # Type for the input of the contribution. This is what goes into the profile
OUT = TypeVar('OUT')  # Type for the output of the profile. What comes out of the profile
AI = TypeVar('AI')  # Type for the input to the Aggregators.
AO = TypeVar('AO')  # Type for the output of the Aggregators.


class ProfileElement(Generic[AO], ABC):
    @abstractmethod
    def contribute(self, contribution: AO):
        pass


class ProfileField(Generic[IN, OUT], ABC):
    @abstractmethod
    def contribute(self, contribution: IN):
        pass

    @abstractmethod
    def list(self) -> OUT:
        pass


class ProfileAggregator(Generic[AO], ABC):
    @abstractmethod
    def aggregate(self, field: ProfileElement[AO]) -> AO:
        pass


class ProfileFieldFactory(Generic[IN, OUT], ABC):
    @abstractmethod
    def get_profile_fields(self, features: List[FeatureGrouper], contrib_tensor_definition: TensorDefinition) \
            -> List[ProfileField[IN, OUT]]:
        pass


class ProfileAggregatorFactory(Generic[AO], ABC):
    @abstractmethod
    def get_aggregator(self, aggregator: Aggregator) -> ProfileAggregator[AO]:
        pass


class Profile(Generic[IN, AI, AO, OUT], ABC):
    def __init__(self, features: List[FeatureGrouper], contrib_tensor_definition: TensorDefinition):
        self.features = features

    def contribute(self, contribution: IN):
        for pf in self.profile_fields:
            pf.contribute(contribution)

    @property
    @abstractmethod
    def profile_fields(self) -> List[ProfileField[IN, OUT]]:
        pass

    @abstractmethod
    def list(self) -> OUT:
        pass

    @abstractmethod
    def get_profile_field_factory(self) -> ProfileFieldFactory[IN, OUT]:
        pass


class ProfileAggregatorFactoryNative(ProfileAggregatorFactory[float]):
    def get_aggregator(self, aggregator: Aggregator) -> ProfileAggregator[float]:
        if aggregator.name == 'Count':
            return ProfileAggregatorNativeCount()
        elif aggregator.name == 'Sum':
            return ProfileAggregatorNativeSum()
        elif aggregator.name == 'Mean':
            return ProfileAggregatorNativeMean()
        elif aggregator.name == 'Stddev':
            return ProfileAggregatorNativeStddev()
        elif aggregator.name == 'Min':
            return ProfileAggregatorNativeMin()
        elif aggregator.name == 'Max':
            return ProfileAggregatorNativeMax()
        else:
            raise ProfileException(f'Unknown Profile aggregator {aggregator.name}')


class ProfileFieldNative(ProfileField[np.ndarray, List[float]]):
    def __init__(self, base_feature: Feature, filter_feature: Feature, contrib_tensor_definition: TensorDefinition):
        self.element = ProfileElementNative()
        self.base_feature_index = self.feature_index(base_feature, contrib_tensor_definition)
        self.filter_feature_index = self.feature_index(filter_feature, contrib_tensor_definition)

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

    def contribute(self, contribution: np.ndarray):
        # Only contribute if filter is True or None
        if self.filter_feature_index == -1 or contribution[self.filter_feature_index] is True:
            self.element.contribute(contribution[self.base_feature_index])

    def list(self) -> List[float]:
        pass


class ProfileFieldFactoryNative(ProfileFieldFactory[np.ndarray, List[float]]):
    def get_profile_fields(self, features: List[FeatureGrouper], contrib_tensor_definition: TensorDefinition) -> \
            List[ProfileField[np.ndarray, List[float]]]:
        # We only need 1 profile field per unique base_feature, filter_feature and time settings. It will calculate
        # all the aggregates.
        uniq_f = {
            k: list(v) for k, v in
            groupby(features, lambda x: (x.base_feature, x.filter_feature, x.time_period, x.time_window))
        }
        fields = []
        for (bf, ff, _, _), v in uniq_f.items():
            fields.append(ProfileFieldNative(bf, ff, contrib_tensor_definition))
        return fields


class ProfileNative(Profile[np.ndarray, float, float, List[float]]):
    def __init__(self, features: List[FeatureGrouper], contrib_tensor_definition: TensorDefinition):
        super(ProfileNative, self).__init__(features, contrib_tensor_definition)
        # Make a dict that keeps track of which feature is stored in which field.
        # We only need 1 profile field per unique base_feature, filter_feature and time settings. It will calculate
        # all the aggregates.
        uniq_f: Dict[Tuple[Feature, Feature, TimePeriod, int], List[FeatureGrouper]] = {
            k: list(v) for k, v in
            groupby(features, lambda x: (x.base_feature, x.filter_feature, x.time_period, x.time_window))
        }
        self._profile_fields: List[ProfileField[np.ndarray, List[float]]] = []
        self._features_to_field: Dict[FeatureGrouper, ProfileField[np.ndarray, List[float]]] = {}
        for (bf, ff, _, _), v in uniq_f.items():
            pf = ProfileFieldNative(bf, ff, contrib_tensor_definition)
            self._profile_fields.append(pf)
            for f in v:
                self._features_to_field[f] = pf

    def get_profile_field_factory(self) -> ProfileFieldFactory:
        return ProfileFieldFactoryNative()

    @property
    def profile_fields(self) -> List[ProfileField[np.ndarray, List[float]]]:
        return self._profile_fields

    def list(self) -> List[float]:
        for f in self.features:
            agg = ProfileAggregatorFactoryNative().get_aggregator(f.aggregator)
            pf = self._features_to_field[f]
            return agg.aggregate(pf.list())


# class ProfileFieldDict(Profile):
#     @abstractmethod
#     def contribute(self, key: str, value: float):
#         pass


class ProfileElementNative(ProfileElement[float]):
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


class ProfileAggregatorNativeCount(ProfileAggregator[float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.count


class ProfileAggregatorNativeSum(ProfileAggregator[float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.count * field.mean


class ProfileAggregatorNativeMean(ProfileAggregator[float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.mean


class ProfileAggregatorNativeStddev(ProfileAggregator[float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        if field.count < 2:
            return float("nan")
        else:
            return sqrt(field.M2 / (field.count + 1))


class ProfileAggregatorNativeMin(ProfileAggregator[float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.min


class ProfileAggregatorNativeMax(ProfileAggregator[float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.max


# class ProfileDictNative(ProfileFieldDict):
#     def __init__(self):
#         self.field_dict: Dict[str, ProfileField] = {}
#
#     def contribute(self, key: str, value: float):
#         fld = self.field_dict.setdefault(key, ProfileFieldNative())
#         fld.contribute(value)
