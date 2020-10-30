"""
Definition of label Features. They will be the target during training
(c) 2020 d373c7
"""
import logging
from ..features.common import Feature, FeatureType, FeatureTypeNumerical, FeatureDefinitionException, FEATURE_TYPE_INT_8
from ..features.common import LearningCategory, LEARNING_CATEGORY_LABEL
from typing import List

logger = logging.getLogger(__name__)


class FeatureLabel(Feature):
    def __init__(self, name: str, f_type: FeatureType, base_feature: Feature):
        Feature.__init__(self, name, f_type)
        self._base_feature = base_feature

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and self.type == other.type
        else:
            return False

    def __hash__(self):
        return hash(self.name) + hash(self.type)

    @property
    def base_feature(self):
        """Return the base feature on which this Label was built. Will return an instance of type 'Feature'

        :return: The base feature as instance of 'Feature'
        """
        return self._base_feature

    @property
    def embedded_features(self) -> List[Feature]:
        return [self._base_feature]

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_LABEL


class FeatureLabelBinary(FeatureLabel):
    """Feature to indicate what the label needs to be during training. This feature will assume it is binary of type
    INT8 and will contain values 0 and 1.

    Args:
        name: The name for this feature.
        base_feature: The field on which the label should be based.
    """
    @staticmethod
    def _val_is_numerical(base_feature: Feature):
        if not isinstance(base_feature.type, FeatureTypeNumerical):
            raise FeatureDefinitionException(
                f'The base feature of a Binary Label feature must be numerical Got <{base_feature.type}>'
            )

    def __init__(self, name: str, base_feature: Feature):
        FeatureLabelBinary._val_is_numerical(base_feature)
        FeatureLabel.__init__(self, name, FEATURE_TYPE_INT_8, base_feature)

    def __repr__(self):
        return f'Label Feature {self.name}/{self.type}'
