"""
Definition of some fairly straight forward Features
(c) 2020 d373c7
"""
import logging
from typing import List, Optional, Union, Dict
from dataclasses import dataclass, field
from .common import enforce_types
from ..features.common import Feature, FeatureCategorical, FeatureDefinitionException
from ..features.common import FeatureWithBaseFeature, LearningCategory, LEARNING_CATEGORY_NONE
from ..features.common import FeatureHelper, FeatureTypeTimeBased, FeatureTypeNumerical, FeatureTypeString

logger = logging.getLogger(__name__)


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureSource(Feature):
    """
    A feature found in a source. I.e a file or message or JSON or other. This is the most basic feature.
    """
    format_code: Optional[str] = None
    default: Optional[Union[str, float, int]] = None

    def __post_init__(self):
        self._val_format_code_not_none_for_time()

    @property
    def inference_ready(self) -> bool:
        # A source feature has no inference attributes
        return True

    @property
    def learning_category(self) -> LearningCategory:
        # Should be the learning category of the type of the source feature
        return self.type.learning_category

    def _val_format_code_not_none_for_time(self):
        if FeatureHelper.is_feature_of_type(self, FeatureTypeTimeBased):
            if self.format_code is None:
                raise FeatureDefinitionException(
                    f'Feature {self.name} is time based, its format_code should not be <None>'
                )


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureVirtual(Feature):
    """
    A placeholder feature without actual definition. Sometimes we might want to refer to a feature that is not
    an actual feature. Fluffy, true, this is a feature without actually being one.
    Virtual features should be created by providing a name and type
    """
    @property
    def inference_ready(self) -> bool:
        # A virtual feature has no inference attributes
        return True

    @property
    def learning_category(self) -> LearningCategory:
        # Virtual features are never used for learning. No matter what their type is.
        return LEARNING_CATEGORY_NONE


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureIndex(FeatureWithBaseFeature, FeatureCategorical):
    """
    Indexer feature. It will turn a specific input field (the base_feature) into an index. For instance 'DE'->1,
    'FR'->2, 'GB'->3 etc... The index will have an integer type and is ideal to model in embeddings.
    """
    dictionary: Dict = field(default=None, init=False, hash=False)

    def __post_init__(self):
        self.val_int_type()
        self.val_base_feature_is_string_or_integer()
        # By default, return set embedded features to be the base feature.
        self.embedded_features.append(self.base_feature)
        self.embedded_features.extend(self.base_feature.embedded_features)

    def __len__(self):
        return len(self.dictionary)

    @property
    def inference_ready(self) -> bool:
        return self.dictionary is not None


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureBin(FeatureWithBaseFeature, FeatureCategorical):
    """
    Feature that will 'bin' a float number. Binning means the float feature will be turned into an int/categorical
    variable. For instance values 0.0 till 0.85 will be bin 1, from 0.85 till 1.7 bin 2 etc
    """
    number_of_bins: int
    scale_type: str = 'linear'
    bins: List[int] = field(default=None, init=False, hash=False)

    def __post_init__(self):
        self.val_int_type()
        self.val_base_feature_is_float()
        # By default; return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    def __len__(self):
        return self.number_of_bins

    @property
    def range(self) -> List[int]:
        return list(range(1, self.number_of_bins))

    @property
    def inference_ready(self) -> bool:
        return self.bins is not None


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureRatio(FeatureWithBaseFeature):
    """
    Feature to calculate a ratio between 2 numbers. It will take the first input number and divide it by the second.
    It will avoid division by 0. If 0 is the denominator, the result will be 0 and not an error.
    """
    denominator_feature: Feature

    def __post_init__(self):
        self.val_float_type()
        self.val_base_feature_is_numerical()
        self._val_denominator_is_numerical()
        # Add base and denominator to the embedded features list
        self.embedded_features = self.get_base_and_base_embedded_features()
        self.embedded_features.append(self.denominator_feature)
        self.embedded_features.extend(self.denominator_feature.embedded_features)

    @property
    def learning_category(self) -> LearningCategory:
        # Should be the learning category of the type of the source feature
        return self.type.learning_category

    @property
    def inference_ready(self) -> bool:
        # A ratio feature has no inference attributes
        return True

    def _val_denominator_is_numerical(self):
        if not FeatureHelper.is_feature_of_type(self.denominator_feature, FeatureTypeNumerical):
            raise FeatureDefinitionException(
                f'The denominator feature {self.denominator_feature.name} of a FeatureRatio should be numerical. ' +
                f'Got {self.denominator_feature.type} '
            )


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureConcat(FeatureWithBaseFeature):
    """
    Feature to concatenate 2 features. Both feature must be string type, the result will be a string
    """
    concat_feature: Feature

    def __post_init__(self):
        self.val_string_type()
        self.val_base_feature_is_string()
        self._val_concat_feature_is_string()
        # Add base and concat to the embedded features list
        self.embedded_features = self.get_base_and_base_embedded_features()
        self.embedded_features.append(self.concat_feature)
        self.embedded_features.extend(self.concat_feature.embedded_features)

    @property
    def learning_category(self) -> LearningCategory:
        # Should be the learning category of the type of the source feature
        return self.type.learning_category

    @property
    def inference_ready(self) -> bool:
        # A concat feature has no inference attributes
        return True

    def _val_concat_feature_is_string(self):
        if not FeatureHelper.is_feature_of_type(self.concat_feature, FeatureTypeString):
            raise FeatureDefinitionException(
                f'The concat feature {self.concat_feature.name} of a FeatureRatio should be a string. ' +
                f'Got {self.concat_feature.type} '
            )
