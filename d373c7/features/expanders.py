"""
Definition of expander features.
(c) 2020 d373c7
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from ..common import enforce_types
from .common import LearningCategory, LEARNING_CATEGORY_BINARY
from ..features.common import FeatureWithBaseFeature
from ..features.common import not_implemented
from ..features.base import FeatureVirtual


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureExpander(FeatureWithBaseFeature, ABC):
    """
    Base class for expander features. Expander features expand when they are built. One feature in an input
    can turn into multiple features in output. For instance a one_hot encoded feature.
    """
    expand_names: List[str] = field(default=None, init=False, hash=False)

    @abstractmethod
    def expand(self) -> List[FeatureVirtual]:
        return not_implemented(self)


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureOneHot(FeatureExpander):
    """
    A One Hot feature. This will take a base feature and one hot encode it. It will create as many additional
    virtual features as there are input values. The virtual feature will have a specific name for instance
    <base_feature>__<input_value>
    """
    def __post_init__(self):
        self.val_int_type()
        self.val_base_feature_is_string_or_integer()
        # By default return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    def expand(self) -> List[FeatureVirtual]:
        if self.expand_names is not None:
            return [FeatureVirtual(name=n, type=self.type) for n in self.expand_names]
        else:
            return []

    @property
    def inference_ready(self) -> bool:
        return self.expand_names is not None

    @property
    def learning_category(self) -> LearningCategory:
        # Treat One Hot Features as 'Binary' learning category. Even though they are encoded as integers.
        return LEARNING_CATEGORY_BINARY
