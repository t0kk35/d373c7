"""
Definition of some fairly straight forward Features
(c) 2020 d373c7
"""
import logging
from typing import List
from ..features.common import Feature, FeatureType, FeatureDefinitionException


logger = logging.getLogger(__name__)


def not_implemented(feature: Feature):
    raise NotImplementedError(f'Feature problem. Not defined for class {feature.__class__.name}')


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
