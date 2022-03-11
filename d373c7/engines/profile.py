"""
Profile Base definitions
(c) 2022 d373c7
"""
import logging
from abc import ABC, abstractmethod
from features import FeatureGrouper, Aggregator, FeatureHelper
from typing import List, Dict, Tuple, TypeVar, Generic
logger = logging.getLogger(__name__)


class ProfileException(Exception):
    def __init__(self, message: str):
        super().__init__('Error profiling: ' + message)


IN = TypeVar('IN')
OUT = TypeVar('OUT')
E = TypeVar('E')


class ProfileElement(Generic[IN, E, OUT], ABC):
    @abstractmethod
    def contribute(self, contribution: IN):
        pass


class ProfileField(Generic[IN, E, OUT], ABC):
    @abstractmethod
    def extract(self, contribution: IN) -> E:
        pass


class ProfileAggregator(Generic[E, OUT], ABC):
    @abstractmethod
    def aggregate(self, field: ProfileElement[IN, E, OUT]) -> OUT:
        pass


class ProfileFieldFactory(Generic[IN, E, OUT], ABC):
    @abstractmethod
    def get_profile_field(self, feature: FeatureGrouper) -> ProfileField[IN, E, OUT]:
        pass


class ProfileAggregatorFactory(Generic[E, OUT], ABC):
    @abstractmethod
    def get_aggregator(self, aggregator: Aggregator) -> ProfileAggregator[E, OUT]:
        pass


class Profile(Generic[IN, E, OUT], ABC):
    def __init__(self, features: List[FeatureGrouper]):
        ff_factory = self.get_profile_field_factory()
        ag_factory = self.get_aggregator_factory()
        self.profile_fields: Dict[
            Tuple[FeatureGrouper, Tuple[ProfileElement[IN, E, OUT], ProfileAggregator[E, OUT]]]
        ] = {
            f: (ff_factory.get_profile_field(f), ag_factory.get_aggregator(f.aggregator)) for f in features
        }

    @abstractmethod
    def get_aggregator_factory(self) -> ProfileAggregatorFactory[E, OUT]:
        pass

    @abstractmethod
    def get_profile_field_factory(self) -> ProfileFieldFactory:
        pass

    def contribute(self, contribution: IN):
        for field, _ in self.field_dict.values():
            field.contribute(contribution)

    @property
    def field_dict(self) -> Dict[FeatureGrouper, Tuple[ProfileElement[IN, OUT], ProfileAggregator[IN, OUT]]]:
        return self.field_dict

    def get_profile_field(self, feature: FeatureGrouper, context: IN) -> OUT:
        try:
            field, agg = self.field_dict[feature]
            x = field.extract(context)
            x = agg.aggregate(x)
            return x
        except KeyError:
            raise ProfileException(f'Could not find key in Profile dict for feature {feature.name}')


class ProfileAggregatorFactoryNative(ProfileAggregatorFactory[str, float]):
    def get_aggregator(self, aggregator: Aggregator) -> ProfileAggregator[str, float]:
        if aggregator.name == 'count':
            return ProfileAggregatorNativeCount()
        elif aggregator.name == 'sum':
            return ProfileAggregatorNativeSum()
        elif aggregator.name == 'mean':
            return ProfileAggregatorNativeMean()
        elif aggregator.name == 'stddev':
            return ProfileAggregatorNativeStddev()
        elif aggregator.name == 'min':
            return ProfileAggregatorNativeMin()
        elif aggregator.name == 'max':
            return ProfileAggregatorNativeMax()
        else:
            raise ProfileException(f'Unknown Profile aggregator {aggregator.name}')


class ProfileFieldFactoryNative(ProfileFieldFactory):
    def get_profile_field(self, feature: FeatureGrouper):
        if FeatureHelper.is_feature(feature, FeatureGrouper):
            return ProfileElementNative()


class ProfileNative(Profile):
    def get_profile_field_factory(self) -> ProfileFieldFactory:
        return ProfileFieldFactoryNative()

    def get_aggregator_factory(self) -> ProfileAggregatorFactory:
        return ProfileAggregatorFactory()


# class ProfileFieldDict(Profile):
#     @abstractmethod
#     def contribute(self, key: str, value: float):
#         pass


class ProfileElementNative(ProfileElement[str, float]):
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


class ProfileAggregatorNativeCount(ProfileAggregator[str, float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.count


class ProfileAggregatorNativeSum(ProfileAggregator[str, float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.count * field.mean


class ProfileAggregatorNativeMean(ProfileAggregator[str, float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.mean


class ProfileAggregatorNativeStddev(ProfileAggregator[str, float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        if field.count < 2:
            return float("nan")
        else:
            return field.M2 / (field.count + 1)


class ProfileAggregatorNativeMin(ProfileAggregator[str, float]):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.min


class ProfileAggregatorNativeMax(ProfileAggregator):
    def aggregate(self, field: ProfileElementNative) -> float:
        return field.max


# class ProfileDictNative(ProfileFieldDict):
#     def __init__(self):
#         self.field_dict: Dict[str, ProfileField] = {}
#
#     def contribute(self, key: str, value: float):
#         fld = self.field_dict.setdefault(key, ProfileFieldNative())
#         fld.contribute(value)
