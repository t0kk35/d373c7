"""
Definition of some fairly straight forward Features
(c) 2020 d373c7
"""
import logging
from typing import List, Optional, Union, Dict
from dataclasses import dataclass, field
from .common import enforce_types
from ..features.common import Feature, FeatureCategorical
from ..features.common import FeatureWithBaseFeature, LearningCategory, LEARNING_CATEGORY_NONE


logger = logging.getLogger(__name__)


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureSource(Feature):
    """
    A feature found in a source. I.e a file or message or JSON or other. This is the most basic feature.
    """
    format_code: Optional[str] = None
    default: Optional[Union[str, float, int]] = None

    @property
    def inference_ready(self) -> bool:
        # A source feature has no inference attributes
        return True

    @property
    def learning_category(self) -> LearningCategory:
        # Should be the learning category of the type of the source feature
        return self.type.learning_category


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureVirtual(Feature):
    """
    A place holder feature without actual definition. Sometimes we might want to refer to a feature that is not
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
        # By default return set embedded features to be the base feature.
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
        # By default return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    def __len__(self):
        return self.number_of_bins

    @property
    def range(self) -> List[int]:
        return list(range(1, self.number_of_bins))

    @property
    def inference_ready(self) -> bool:
        return self.bins is not None
