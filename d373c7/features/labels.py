"""
Definition of label Features. They will be the target during training
(c) 2020 d373c7
"""
import logging
from dataclasses import dataclass
from ..common import enforce_types
from ..features.common import FeatureWithBaseFeature
from ..features.common import LearningCategory, LEARNING_CATEGORY_LABEL

logger = logging.getLogger(__name__)


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureLabel(FeatureWithBaseFeature):
    """
    Base class for all Features that will be used as labels during training
    """
    @property
    def inference_ready(self) -> bool:
        # A label feature has not inference attributes
        return True

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_LABEL


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureLabelBinary(FeatureLabel):
    """
    Feature to indicate what the label needs to be during training. This feature will assume it is binary of type
    INT and will contain values 0 and 1.
    """
    def __post_init__(self):
        # Do post init validation
        self.val_int_type()
        self.val_base_feature_is_numerical()
        # By default, return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()
