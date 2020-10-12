"""
Definition of some fairly straight forward Features
(c) 2020 d373c7
"""
import logging
from typing import List
from ..features.common import Feature, FeatureType, FeatureDefinitionException, FeatureInferenceAttributes
from ..features.common import LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_NONE
from ..features.common import FeatureTypeString, FeatureTypeInteger, LearningCategory


logger = logging.getLogger(__name__)


def not_implemented(class_):
    raise NotImplementedError(f'Feature problem. Not defined for class {class_.__class__.name}')


class FeatureSource(Feature):
    """"A feature found in a source. I.e a file or message or JSON or other. This is the most basic feature.
    Args:
        name: A name for the feature
        f_type: The type of the feature. This must be an instance of the FeatureType class
        format_code: A specific format to be applied to the input string. For instance to convert to a date.
        default: A default value. If set, this value will be default if missing in the input source.
    """
    def __init__(self, name: str, f_type: FeatureType, format_code: str = None, default: any = None):
        if type(name) != str:
            raise FeatureDefinitionException('Source Field must have a name parameter of type string')
        Feature.__init__(self, name, f_type)
        self._format_code = format_code
        self._default = default

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and self.type == other.type
        else:
            return False

    def __hash__(self):
        return hash(self.name) + hash(self.type)

    def __repr__(self):
        return f'Source Feature {self.name}/{self.type}'

    @property
    def embedded_features(self) -> List[Feature]:
        return []

    @property
    def format_code(self):
        return self._format_code

    @property
    def default(self) -> any:
        return self._default

    @property
    def learning_category(self) -> LearningCategory:
        if isinstance(self.type, FeatureTypeInteger):
            return LEARNING_CATEGORY_CATEGORICAL
        else:
            return LEARNING_CATEGORY_NONE


class FeatureVirtual(Feature):
    """A place holder feature without actual definition. Sometimes we might want to refer to a feature that is not
    an actual feature. Fluffy, true, this is a feature without actually being one.
    Virtual features should be created by
    - Either providing a base feature to virtualize
    - Or providing a name and f_type

    Args:
        feature: A feature to virtualize
        name: The name for the virtual feature.
        f_type: The type of the virtual feature.
    """
    @staticmethod
    def _val_feature_or_name(feature: Feature, name: str, f_type: FeatureType):
        if feature is not None:
            if name is not None or f_type is not None:
                raise FeatureDefinitionException(
                    f'Virtual feature creation should be done with either a feature or a name and type'
                )
        if feature is None:
            if name is None or f_type is None:
                raise FeatureDefinitionException(
                    f'Virtual Feature creation, if no feature is given, then a name and type should be provided'
                )

    def __init__(self, feature: Feature = None, name: str = None, f_type: FeatureType = None):
        self._val_feature_or_name(feature, name, f_type)
        feature_name = feature.name if feature is not None else name
        feature_type = feature.type if feature is not None else f_type
        Feature.__init__(self, feature_name, feature_type)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and self.type == other.type
        else:
            return False

    def __hash__(self):
        return hash(self.name) + hash(self.type)

    def __repr__(self):
        return f'Virtual Feature {self.name}/{self.type}'

    @property
    def embedded_features(self) -> List[Feature]:
        return []


class FeatureIndex(FeatureInferenceAttributes):
    """Indexer feature. It will turn a specific input field (the base_feature) into an index. For instance 'DE'->1,
    'FR'->2, 'GB'->3 etc... The index will have an integer type and is ideal to model in embeddings.
    Args:
        name: A name for the index feature
        f_type: The type of the feature. This must be an instance of the FeatureTypeInteger class (i.e integer based)
        base_feature: The feature which will be indexed. This should be either string or integer based.
    """
    @staticmethod
    def _val_type_is_string_or_integer(base_feature: Feature):
        ft = base_feature.type
        if not isinstance(ft, FeatureTypeString) and not isinstance(ft, FeatureTypeInteger):
            raise FeatureDefinitionException(
                f'The base feature parameter of an indexing feature must be a string-type or integer-type. '
                f'Found [{type(base_feature.type)}]'
            )

    @staticmethod
    def _val_type_is_integer_based(f_type: FeatureType):
        if not isinstance(f_type, FeatureTypeInteger):
            raise FeatureDefinitionException(
                f'The feature type of an index feature should be integer based, not {f_type}'
            )

    def __init__(self, name: str, f_type: FeatureType, base_feature: Feature):
        self._val_type_is_string_or_integer(base_feature)
        self._val_type_is_integer_based(f_type)
        Feature.__init__(self, name, f_type)
        self._dict = None
        self._base_feature = base_feature

    def __len__(self):
        return len(self.dictionary)

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
        return f'IndexFeature. {self.name}/{self.type}. Base {self.base_feature.name}'

    @property
    def base_feature(self) -> Feature:
        return self._base_feature

    @property
    def dictionary(self) -> dict:
        return self._dict

    @dictionary.setter
    def dictionary(self, dictionary: dict):
        self._dict = dictionary

    @property
    def embedded_features(self) -> List[Feature]:
        return [self._base_feature]

    @property
    def category(self) -> LearningCategory:
        return LEARNING_CATEGORY_CATEGORICAL

    @property
    def inference_ready(self) -> bool:
        return self._dict is not None
