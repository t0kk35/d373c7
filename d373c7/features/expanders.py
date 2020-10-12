"""
Definition of expander features.
(c) 2020 d373c7
"""
from typing import List
from ..features.common import LEARNING_CATEGORY_BINARY, LearningCategory
from ..features.common import Feature, FeatureInferenceAttributes, FeatureTypeString, FEATURE_TYPE_INT_8
from ..features.common import FeatureDefinitionException
from ..features.common import not_implemented
from ..features.base import FeatureVirtual


class FeatureExpander(FeatureInferenceAttributes):
    """ Base class for expander features. Expander features expand when they are built. One feature in an input
    can turn into multiple features in output. For instance a one_hot encoded feature.
    """
    @property
    def base_feature(self) -> Feature:
        return not_implemented(self)

    @property
    def expand_names(self) -> List[str]:
        return not_implemented(self)

    @expand_names.setter
    def expand_names(self, names: List[str]):
        pass

    def expand(self) -> List[FeatureVirtual]:
        return not_implemented(self)


class FeatureOneHot(FeatureExpander):
    """A One Hot feature. This will take a base feature and one hot encode it. It will create as many additional
    virtual features as there are input values. The virtual feature will have a specific name for instance
    <base_feature>__<input_value>

    Args:
        name: A name for the One Hot Feature
        base_feature: The base feature which will be one hot encoded.
    """
    @staticmethod
    def _val_type_is_string(base_feature: Feature):
        if not isinstance(base_feature.type, FeatureTypeString):
            raise FeatureDefinitionException(
                f'The base feature parameter of a one-hot feature must be a string-type. '
                f'Found [{type(base_feature.type)}]'
            )

    def __init__(self, name, base_feature: Feature):
        FeatureOneHot._val_type_is_string(base_feature)
        # Use smallest possible type. Can only take 0 or 1 as value
        Feature.__init__(self, name, FEATURE_TYPE_INT_8)
        self._base_feature = base_feature
        self._e_names = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and \
                   self.base_feature == other.base_feature and \
                   self.type == other.type
        else:
            return False

    def __hash__(self):
        return hash(self.name) + hash(self.base_feature) + hash(self.type)

    def __repr__(self):
        return f'OneHot Feature. {self.name}/{self.type}. Base {self.base_feature.name}'

    @property
    def base_feature(self) -> Feature:
        return self._base_feature

    @property
    def embedded_features(self) -> List[Feature]:
        return [self._base_feature]

    @property
    def expand_names(self) -> List[str]:
        return self._e_names

    @expand_names.setter
    def expand_names(self, names: List[str]):
        self._e_names = names

    def expand(self) -> List[FeatureVirtual]:
        if self.expand_names is not None:
            return [FeatureVirtual(name=n, f_type=FEATURE_TYPE_INT_8) for n in self.expand_names]
        else:
            return []

    @property
    def inference_ready(self) -> bool:
        return self._e_names is not None

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_BINARY
