"""
Definition of normaliser features.
(c) 2020 d373c7
"""
from typing import List
from ..features.base import FeatureDefinitionException
from ..features.common import Feature, FeatureInferenceAttributes, FeatureType, FeatureTypeFloat
from ..features.common import not_implemented


class FeatureNormalize(FeatureInferenceAttributes):
    """Base class for features with normalizing logic
    """
    @staticmethod
    def _val_base_type_is_float(base_feature: Feature):
        if not isinstance(base_feature.type, FeatureTypeFloat):
            raise FeatureDefinitionException(
                f'The base feature parameter of a normaliser feature must be a float-type. '
                f'Found [{type(base_feature.type)}]'
            )

    @staticmethod
    def _val_f_type_is_float(f_type: FeatureType):
        if not isinstance(f_type, FeatureTypeFloat):
            raise FeatureDefinitionException(
                f'The f_type of a normaliser feature must be a float-type. '
                f'Found [{type(f_type)}]'
            )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and self.base_feature == other.base_feature
        else:
            return False

    def __hash__(self):
        return hash(self.name) + hash(self.base_feature)

    @property
    def base_feature(self) -> Feature:
        return not_implemented(self)


class FeatureNormalizeScale(FeatureNormalize):
    """Normalizing feature. Feature that scales a base feature between 0 and 1 with a min/max logic.

    Args:
        name: Name of the feature
        f_type: Feature Type. This must be a float type.
        base_feature: The feature to normalize. This must be a float type.
    """
    def __init__(self, name: str, f_type: FeatureType, base_feature: Feature):
        FeatureNormalize._val_f_type_is_float(f_type)
        FeatureNormalize._val_base_type_is_float(base_feature)
        Feature.__init__(self, name, f_type)
        self._base_feature = base_feature
        self._minimum = None
        self._maximum = None

    def __repr__(self):
        return f'Normalise Scaler {self.name}/{self.type}. Base {self.base_feature.name} ' \
               f'Attr {self.minimum}/{self.maximum} '

    @property
    def base_feature(self) -> Feature:
        return self._base_feature

    @property
    def embedded_features(self) -> List[Feature]:
        return [self._base_feature]

    @property
    def inference_ready(self) -> bool:
        return self.minimum is not None and self.maximum is not None

    @property
    def minimum(self) -> float:
        return self._minimum

    @minimum.setter
    def minimum(self, minimum: float):
        self._minimum = minimum

    @property
    def maximum(self) -> float:
        return self._maximum

    @maximum.setter
    def maximum(self, maximum: float):
        self._maximum = maximum


class FeatureNormalizeStandard(FeatureNormalize):
    """Normalizing feature. Feature that standardises a base feature around mean zero and unit standard deviation.

    Args:
        name: Name of the feature
        f_type: Feature Type. This must be a float type.
        base_feature: The feature to normalize. This must be a float type.
    """
    def __init__(self, name: str, f_type: FeatureType, base_feature: Feature):
        FeatureNormalize._val_f_type_is_float(f_type)
        FeatureNormalize._val_base_type_is_float(base_feature)
        Feature.__init__(self, name, f_type)
        self._base_feature = base_feature
        self._mean = None
        self._stddev = None

    def __repr__(self):
        return f'Normalise Scaler {self.name}/{self.type}. Base {self.base_feature.name} ' \
               f'Attr {self.mean}/{self.stddev} '

    @property
    def base_feature(self) -> Feature:
        return self._base_feature

    @property
    def embedded_features(self) -> List[Feature]:
        return [self._base_feature]

    @property
    def inference_ready(self) -> bool:
        return self.mean is not None and self.stddev is not None

    @property
    def mean(self) -> float:
        return self._mean

    @mean.setter
    def mean(self, mean: float):
        self._mean = mean

    @property
    def stddev(self) -> float:
        return self._stddev

    @stddev.setter
    def stddev(self, stddev: float):
        self._stddev = stddev
