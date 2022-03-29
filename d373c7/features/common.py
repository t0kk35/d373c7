"""
Common classes for all features
(c) 2020 d373c7
"""
from ..common import enforce_types
from dataclasses import dataclass, field
from typing import List, TypeVar, Type
from abc import ABC, abstractmethod


def not_implemented(class_):
    raise NotImplementedError(f'Feature problem. Not defined for class {class_.__class__.name}')


class FeatureDefinitionException(Exception):
    """
    Exception thrown when the Definition of a feature fails
    """
    def __init__(self, message: str):
        super().__init__("Error Defining Feature: " + message)


@enforce_types
@dataclass(frozen=True, order=True)
class LearningCategory:
    """
    Describes the feature category. The category will be used to drive what sort of layers and learning models
    can be used on a specific feature.
    """
    key: int = field(repr=False)
    name: str
    default_panda_type: str = field(repr=False)
    sort_index: int = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.key)


LEARNING_CATEGORY_BINARY: LearningCategory = LearningCategory(0, 'Binary', 'int8')
LEARNING_CATEGORY_CATEGORICAL: LearningCategory = LearningCategory(1, 'Categorical', 'int32')
LEARNING_CATEGORY_CONTINUOUS: LearningCategory = LearningCategory(2, 'Continuous', 'float64')
LEARNING_CATEGORY_LABEL: LearningCategory = LearningCategory(3, 'Label', 'float32')
LEARNING_CATEGORY_NONE: LearningCategory = LearningCategory(4, 'None', 'None')

LEARNING_CATEGORIES: List[LearningCategory] = [
    LEARNING_CATEGORY_BINARY,
    LEARNING_CATEGORY_CONTINUOUS,
    LEARNING_CATEGORY_CATEGORICAL,
    LEARNING_CATEGORY_LABEL,
    LEARNING_CATEGORY_NONE
]

# List of Learning Categories that can be used in models. Should be all LCs excluding the None.
LEARNING_CATEGORIES_MODEL: List[LearningCategory] = [
    lc for lc in LEARNING_CATEGORIES if lc.name != 'None'
]

# List of Learning Categories that can be used in models as input. Should be all LCs excluding the None and the labels
LEARNING_CATEGORIES_MODEL_INPUT: List[LearningCategory] = [
    lc for lc in LEARNING_CATEGORIES_MODEL if lc.name != 'Label'
]


@enforce_types
@dataclass(frozen=True)
class FeatureType:
    """
    Defines the datatype of a particular Feature. It will tell us what sort of data a feature is holding, like
    string values, a float value, an integer value etc... See below for the specific implementations
    """
    key: int = field(repr=False)
    name: str
    learning_category: LearningCategory = field(repr=False)
    precision: int = field(default=0, repr=False)

    @staticmethod
    def max_precision(ft1: 'FeatureType', ft2: 'FeatureType') -> 'FeatureType':
        if ft1.precision > ft2.precision:
            return ft1
        else:
            return ft2


class FeatureTypeString(FeatureType):
    pass


class FeatureTypeNumerical(FeatureType):
    pass


class FeatureTypeInteger(FeatureTypeNumerical):
    pass


class FeatureTypeFloat(FeatureTypeNumerical):
    pass


class FeatureTypeTimeBased(FeatureType):
    pass


class FeatureTypeBool(FeatureTypeNumerical):
    pass


FEATURE_TYPE_STRING: FeatureType = FeatureTypeString(1, 'STRING', LEARNING_CATEGORY_NONE)
FEATURE_TYPE_CATEGORICAL: FeatureType = FeatureTypeString(2, 'CATEGORICAL', LEARNING_CATEGORY_CATEGORICAL)
FEATURE_TYPE_FLOAT: FeatureType = FeatureTypeFloat(3, 'FLOAT', LEARNING_CATEGORY_CONTINUOUS, 64)
FEATURE_TYPE_FLOAT_32: FeatureType = FeatureTypeFloat(4, 'FLOAT_32', LEARNING_CATEGORY_CONTINUOUS, 32)
FEATURE_TYPE_FLOAT_64: FeatureType = FEATURE_TYPE_FLOAT
FEATURE_TYPE_DATE: FeatureType = FeatureTypeTimeBased(5, 'DATE', LEARNING_CATEGORY_NONE)
FEATURE_TYPE_DATE_TIME: FeatureType = FeatureTypeTimeBased(6, 'DATETIME', LEARNING_CATEGORY_NONE)
FEATURE_TYPE_INTEGER: FeatureType = FeatureTypeInteger(7, 'INTEGER', LEARNING_CATEGORY_CATEGORICAL, 32)
FEATURE_TYPE_INT_8: FeatureType = FeatureTypeInteger(8, 'INT_8', LEARNING_CATEGORY_CATEGORICAL, 8)
FEATURE_TYPE_INT_16: FeatureType = FeatureTypeInteger(9, 'INT_16', LEARNING_CATEGORY_CATEGORICAL, 16)
FEATURE_TYPE_INT_32: FeatureType = FEATURE_TYPE_INTEGER
FEATURE_TYPE_INT_64: FeatureType = FeatureTypeInteger(10, 'INT_64', LEARNING_CATEGORY_CATEGORICAL, 64)
FEATURE_TYPE_BOOL: FeatureType = FeatureTypeBool(11, 'INT_8', LEARNING_CATEGORY_BINARY, 8)

T = TypeVar('T')


@enforce_types
@dataclass(unsafe_hash=True)
class Feature(ABC):
    """
    Base Feature class. All features will inherit from this class.
    It is an abstract-ish class that only defines the name and type
    """
    name: str
    type: FeatureType
    embedded_features: List['Feature'] = field(default_factory=list, init=False, hash=False, repr=False)

    def _val_type(self, f_type: Type[FeatureType]) -> None:
        """
        Validation method to check if a feature is of a specific type. Will throw a FeatureDefinitionException
        if the feature is NOT of that type.

        @return: None
        """
        if not FeatureHelper.is_feature_of_type(self, f_type):
            raise FeatureDefinitionException(
                f'The FeatureType of a {self.__class__.__name__} must be {f_type.__name__}. Got <{self.type.name}>'
            )

    def val_int_type(self) -> None:
        """
        Validation method to check if the feature is integer based. Will throw a FeatureDefinitionException
        if the feature is NOT integer based.

        @return: None
        """
        self._val_type(FeatureTypeInteger)

    def val_float_type(self) -> None:
        """
        Validation method to check if the feature is float based. Will throw a FeatureDefinitionException
        if the feature is NOT float based.

        @return: None
        """
        self._val_type(FeatureTypeFloat)

    def val_bool_type(self) -> None:
        """
        Validation method to check if the feature is bool based. Will throw a FeatureDefinitionException
        if the feature is NOT bool based.

        @return: None
        """
        self._val_type(FeatureTypeBool)

    @property
    @abstractmethod
    def learning_category(self) -> LearningCategory:
        """
        Get the learning category of this feature. Will drive the sort of learning operations that are available on
        this feature. Learning categories are 'Continuous','Binary','Categorical' and 'Label

        @return: The Learning Category. An instance of 'LearningCategory'
        """
        return not_implemented(self)

    @property
    @abstractmethod
    def inference_ready(self) -> bool:
        """
        Returns a bool indicating if the feature is ready for inference. Some features need to have been trained
        first or loaded. They need to know the inference attributes they will need to build the feature.

        @return: True if the feature is ready for inference
        """
        return not_implemented(self)


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureWithBaseFeature(Feature, ABC):
    """
    Abstract class for features that have a base feature. These are typically features that are based off of another
    feature. There's a bunch of derived features that will have a base feature.
    """
    base_feature: Feature

    def val_base_feature_is_float(self) -> None:
        """
        Validation method to check if the base_feature is of type float. Will throw a FeatureDefinitionException if the
        base feature is NOT a float.

        @return: None
        """
        if not FeatureHelper.is_feature_of_type(self.base_feature, FeatureTypeFloat):
            raise FeatureDefinitionException(
                f'Base feature of a {self.__class__.__name__} must be a float type. ' +
                f'Got <{type(self.base_feature.type)}>'
            )

    def val_base_feature_is_string_or_integer(self):
        """
        Validation method to check if the base_feature is of type int or string. Will throw a FeatureDefinitionException
        if the base feature is NOT an int or string.

        @return: None
        """
        if not FeatureHelper.is_feature_of_type(self.base_feature, FeatureTypeString) and \
                not FeatureHelper.is_feature_of_type(self.base_feature, FeatureTypeInteger):
            raise FeatureDefinitionException(
                f'The base feature parameter of a {self.__class__.__name__} must be a string-type or integer-type. ' +
                f'Got [{type(self.base_feature.type)}]'
            )

    def val_base_feature_is_numerical(self):
        """
        Validation method to check if the type of the base feature is of is numerical based. Will throw a
        FeatureDefinitionException if the type of the base feature is NOT numerical.

        @return: None
        """
        if not FeatureHelper.is_feature_of_type(self.base_feature, FeatureTypeNumerical):
            raise FeatureDefinitionException(
                f'The base feature parameter of a {self.__class__.__name__} must be a numerical type. ' +
                f'Got [{type(self.base_feature.type)}]'
            )

    def get_base_and_base_embedded_features(self) -> List[Feature]:
        """
        Returns the base_feature + all features embedded in the base_feature.

        @return: (List[Feature]) A list of features.
        """
        return list(set([self.base_feature] + self.base_feature.embedded_features))


class FeatureCategorical(Feature, ABC):
    """
    Placeholder for features that are categorical in nature. They implement an additional __len__ method which
    will be used in embedding layers.
    """
    @abstractmethod
    def __len__(self):
        """Return the cardinality of the categorical feature.

        @return: Integer value, the cardinality of the categorical feature.
        """
        return not_implemented(self)

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_CATEGORICAL


# class methods seem to get lost in data classes. This class bundles some class methods which the original FeatureType
# and Feature classes had, but were no longer visible after changing them to a dataclass.
class FeatureHelper:
    @classmethod
    def is_feature_of_type(cls, feature: Feature, feature_type: Type[FeatureType]) -> bool:
        """
        Determine if a certain feature is an instance of a specific FeatureType

        @param feature: (Feature) An instance of 'Feature'
        @param feature_type (Type) A type of feature to check.
        @return: A boolean value, True if the input class is an instance of the feature_type
        """
        return isinstance(feature.type, feature_type)

    @classmethod
    def is_feature(cls, feature: Feature, feature_class: Type[Feature]) -> bool:
        """
        Determine if a certain feature is an instance of the current class

        @param feature: (Feature) An instance of 'Feature'
        @param feature_class (Type) A type of feature to check.
        @return: A boolean value, True if the input class is an instance of the feature class
        """
        return isinstance(feature, feature_class)

    @classmethod
    def filter_feature(cls, feature_class: Type[T], features: List[Feature]) -> List[T]:
        """
        Class method to filter a list of features. The method will return the features from the input
        that match the current class

        @param feature_class (Type) A feature class to filter.
        @param features: (List of Feature) A list of features.
        @return: A list of features. Contains the input features that are of the class of feature_class.
        """
        return [f for f in features if isinstance(f, feature_class)]

    @classmethod
    def filter_not_feature(cls, feature_class: Type[Feature], features: List[T]) -> List[T]:
        """
        Class method to filter a list of features. The method will return the features from the input
        that do NOT match the current class

        @param feature_class (Type) A feature class to filter.
        @param features: A list of features.
        @return: A list of features. Contains the input features that are not of the class feature_class
        """
        return [f for f in features if not isinstance(f, feature_class)]

    @classmethod
    def filter_feature_type(cls, feature_type: Type[FeatureType], features: List[Feature]) -> List[Feature]:
        """
        Class method to filter a list of features. The method will return the features from the input
        that with type that match the requested feature_type

        @param feature_type (Type) A type of feature to check.
        @param features: (List of Feature) A list of features.
        @return: A list of features. Contains the input features that are of the type of the current class
        """
        return [f for f in features if isinstance(f.type, feature_type)]
