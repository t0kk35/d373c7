"""
Definition of normaliser features.
(c) 2020 d373c7
"""
from abc import ABC
from dataclasses import dataclass
from ..common import enforce_types
from ..features.common import FeatureDefinitionException
from ..features.common import FeatureWithBaseFeature
from ..features.common import LearningCategory, LEARNING_CATEGORY_CONTINUOUS
from typing import Optional, List


@enforce_types
@dataclass
class FeatureNormalize(FeatureWithBaseFeature, ABC):
    """
    Base class for features with normalizing logic
    """
    def __post_init__(self):
        self.val_float_type()
        self.val_base_feature_is_float()

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_CONTINUOUS


@enforce_types
@dataclass
class FeatureNormalizeLogBase(FeatureNormalize, ABC):
    log_base: Optional[str] = None
    delta: float = 1e-2

    def log_base_valid(self):
        if self.log_base is not None and self.log_base not in self.valid_bases():
            raise FeatureDefinitionException(
                f'Error creating {self.name}. Requested log base {self.log_base}. ' +
                f'Supported bases are {self.valid_bases()}'
            )

    @staticmethod
    def valid_bases() -> List[str]:
        return ['e', '10', '2']


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureNormalizeScale(FeatureNormalizeLogBase):
    """
    Normalizing feature. Feature that scales a base feature between 0 and 1 with a min/max logic.
    """
    minimum: float = None
    maximum: float = None

    def __post_init__(self):
        self.val_float_type()
        self.val_base_feature_is_float()
        self.log_base_valid()
        # By default, return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    @property
    def inference_ready(self) -> bool:
        return self.minimum is not None and self.maximum is not None


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureNormalizeStandard(FeatureNormalizeLogBase):
    """
    Normalizing feature. Feature that standardises a base feature around mean zero and unit standard deviation.
    """
    mean: float = None
    stddev: float = None

    def __post_init__(self):
        self.val_float_type()
        self.val_base_feature_is_float()
        self.log_base_valid()
        # By default, return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    @property
    def inference_ready(self) -> bool:
        return self.mean is not None and self.stddev is not None
