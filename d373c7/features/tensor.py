"""
Tensor Definition classes. For grouping features.
(c) 2020 d373c7
"""
from .base import Feature
from typing import List


class TensorDefinition:
    """ A TensorDefinition is a container of features. A set of features can be bundled in a tensor definition. That
    tensor definition can then be constructed by the engines and used in modelling.
    """
    def __init__(self, name: str, features: List[Feature] = None):
        self.__name = name
        if features is None:
            self.__feature_list = []
        else:
            self.__features_list = features

    @property
    def name(self):
        return self.__name

    @property
    def features(self) -> List[Feature]:
        return self.__features_list

    def embedded_features(self) -> List[Feature]:
        # Return the base features + all features embedded in the base features.
        base_features = self.__features_list
        embedded_features = [features.embedded_features for features in base_features]
        embedded_features_flat = [feature for features in embedded_features for feature in features]
        return list(set(base_features + embedded_features_flat))

    def remove(self, feature: Feature) -> None:
        self.__features_list.remove(feature)

    def __len__(self):
        return len(self.features)
