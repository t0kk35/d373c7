"""
Definition of expander features.
(c) 2020 d373c7
"""
# import logging
from typing import List
from ..features.common import Feature, not_implemented
from ..features.base import FeatureVirtual


class FeatureExpander(Feature):
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
