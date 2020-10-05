"""
Common classes for all features
(c) 2020 d373c7
"""
from typing import List


def not_implemented(class_):
    raise NotImplementedError(f'Feature problem. Not defined for class {class_.__class__.name}')


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


class FeatureCategory:
    """Describes the feature category. For instance if the feature is nominal, binary, ordinal, interval etc...
    """
    def __init__(self, key: int, name: str):
        self._key = key
        self._name = name

    @property
    def key(self) -> int:
        return self._key

    @property
    def name(self) -> str:
        return self._name


FEATURE_CATEGORY_BINARY: FeatureCategory = FeatureCategory(0, 'BINARY')
FEATURE_CATEGORY_NOMINAL: FeatureCategory = FeatureCategory(1, 'NOMINAL')
FEATURE_CATEGORY_INTERVAL: FeatureCategory = FeatureCategory(2, 'INTERVAL')


class Feature:
    """Base Feature class. All features will inherit from this class.

    Args:
        name: A name for the feature
        f_type: The type of the feature. Must be a FeatureType class instance

    Attributes:
        name: The name for the feature
        f_type: The type of the feature. Must be a FeatureType class instance
    """
    def __init__(self, name: str, f_type: FeatureType):
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
        return not_implemented(self)


class FeatureInferenceAttributes(Feature):
    """Place holder class for features that have specific inference attributes. Inference attributes are attributes
    that are set during training. During inference, the attributes which have been set at training will be used.
    This is needed to have consistency between training and inference. For instance for all sorts of normalisers.
    """
    @property
    def inference_ready(self) -> bool:
        return not_implemented(self)
