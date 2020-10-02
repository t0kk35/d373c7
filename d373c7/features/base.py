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
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f'Source Feature {self.name}'

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
    pass
