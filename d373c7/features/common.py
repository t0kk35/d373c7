"""
Common classes for all features
(c) 2020 d373c7
"""
from typing import List


def not_implemented(class_):
    raise NotImplementedError(f'Feature problem. Not defined for class {class_.__class__.name}')


def check_attribute_type(parameter, expected_type: type):
    if not isinstance(parameter, expected_type):
        raise AttributeError(
            f'Feature Attribute Error. Expected <{expected_type}>. Got <{type(parameter)}> '
        )


class FeatureDefinitionException(Exception):
    """ Exception thrown when a the Definition of a feature fails
    Args:
        message: A free form text message describing the error
    """
    def __init__(self, message: str):
        super().__init__("Error Defining Feature: " + message)


class FeatureType:
    """Defines the data type of a particular Feature. See below for the specific implementations"""
    def __init__(self, key: int, name: str):
        self._key = key
        self._name = name

    @property
    def key(self) -> int:
        return self._key

    @property
    def name(self) -> str:
        return self._name

    @property
    def precision(self) -> int:
        return not_implemented(self)

    @staticmethod
    def max_precision(ft1: 'FeatureType', ft2: 'FeatureType') -> 'FeatureType':
        if ft1.precision > ft2.precision:
            return ft1
        else:
            return ft2


class FeatureTypeString(FeatureType):
    pass


class FeatureTypeNumerical(FeatureType):
    def __init__(self, key: int, name: str, precision: int):
        FeatureType.__init__(self, key, name)
        self._precision = precision

    @property
    def precision(self) -> int:
        return self._precision


class FeatureTypeInteger(FeatureTypeNumerical):
    pass


class FeatureTypeFloat(FeatureTypeNumerical):
    pass


class FeatureTypeTimeBased(FeatureType):
    pass


class FeatureTypeBool(FeatureTypeNumerical):
    pass


FEATURE_TYPE_STRING: FeatureType = FeatureTypeString(1, 'STRING')
FEATURE_TYPE_CATEGORICAL: FeatureType = FeatureTypeString(2, 'CATEGORICAL')
FEATURE_TYPE_FLOAT: FeatureType = FeatureTypeFloat(3, 'FLOAT', 64)
FEATURE_TYPE_FLOAT_32: FeatureType = FeatureTypeFloat(4, 'FLOAT_32', 32)
FEATURE_TYPE_FLOAT_64: FeatureType = FEATURE_TYPE_FLOAT
FEATURE_TYPE_DATE: FeatureType = FeatureTypeTimeBased(5, 'DATE')
FEATURE_TYPE_DATE_TIME: FeatureType = FeatureTypeTimeBased(6, 'DATETIME')
FEATURE_TYPE_INTEGER: FeatureType = FeatureTypeInteger(7, 'INTEGER', 32)
FEATURE_TYPE_INT_8: FeatureType = FeatureTypeInteger(8, 'INT_8', 8)
FEATURE_TYPE_INT_16: FeatureType = FeatureTypeInteger(9, 'INT_16', 16)
FEATURE_TYPE_INT_32: FeatureType = FEATURE_TYPE_INTEGER
FEATURE_TYPE_INT_64: FeatureType = FeatureTypeInteger(10, 'INT_64', 64)
FEATURE_TYPE_BOOL: FeatureType = FeatureTypeBool(11, 'INT_8', 8)


class LearningCategory:
    """Describes the feature category. The category will be used to drive what sort of layers and learning models
    can used on a specific feature.
    """
    def __init__(self, key: int, name: str):
        self._key = key
        self._name = name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and self.key == other.key
        else:
            return False

    def __hash__(self):
        return hash(self.key) + hash(self.name)

    def __repr__(self):
        return f'Learning Category: {self.name}'

    @property
    def key(self) -> int:
        return self._key

    @property
    def name(self) -> str:
        return self._name


LEARNING_CATEGORY_BINARY: LearningCategory = LearningCategory(0, 'Binary')
LEARNING_CATEGORY_CATEGORICAL: LearningCategory = LearningCategory(1, 'Categorical')
LEARNING_CATEGORY_CONTINUOUS: LearningCategory = LearningCategory(2, 'Continuous')
LEARNING_CATEGORY_NONE: LearningCategory = LearningCategory(3, 'None')
LEARNING_CATEGORY_LABEL: LearningCategory = LearningCategory(4, 'Label')


class Feature:
    """Base Feature class. All features will inherit from this class.

    :argument name: A name for the feature
    :argument f_type: The type of the feature. Must be a FeatureType class instance
    """
    def __init__(self, name: str, f_type: FeatureType):
        check_attribute_type(name, str)
        check_attribute_type(f_type, FeatureType)
        self.__name = name
        self.__type = f_type

    @property
    def name(self) -> str:
        return self.__name

    @property
    def type(self) -> FeatureType:
        return self.__type

    @property
    def embedded_features(self) -> List['Feature']:
        """Return the feature that are embedded in this feature. This can loosely be interpreted as the feature on which
        this feature depends. Feature that need to be known to build this feature.

        :return: List of embedded features. And empty list if there are no embedded features.
        """
        return not_implemented(self)

    @property
    def learning_category(self) -> LearningCategory:
        """Get the learning category of this feature. Will drive the sort of learning operations that are available

        :return: The Learning Category. An instance of 'LearningCategory'
        """
        return not_implemented(self)


class FeatureInferenceAttributes(Feature):
    """Place holder class for features that have specific inference attributes. Inference attributes are attributes
    that are set during training. During inference, the attributes which have been set at training will be used.
    This is needed to have consistency between training and inference. For instance for all sorts of normalisers.
    """
    @property
    def inference_ready(self) -> bool:
        """Returns a bool indicating if the feature is ready for inference. Some features need to have been trained
        first or loaded so the know some of the inference attributes they will need to build the feature.

        :return: True is the feature is ready for inference
        """
        return not_implemented(self)


class FeatureCategorical(FeatureInferenceAttributes):
    """Place holder for features that are categorical in nature. They implement an additional __len__ method which
    will be used in embedding layers.
    """
    def __len__(self):
        """Return the cardinality of the categorical feature.

        :return: Integer value, the cardinality of the categorical feature.
        """
        return not_implemented(self)
