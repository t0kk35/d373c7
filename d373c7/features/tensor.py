"""
Tensor Definition classes. For grouping features.
(c) 2020 d373c7
"""
from .common import LearningCategory, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CATEGORICAL
from .common import LEARNING_CATEGORY_CONTINUOUS
from .base import Feature
from typing import List


class TensorDefinitionException(Exception):
    """ Exception thrown when a the Definition of a feature fails
    Args:
        message: A free form text message describing the error
    """
    def __init__(self, message: str):
        super().__init__("Error Defining Tensor: " + message)


class TensorDefinition:
    """ A TensorDefinition is a container of features. A set of features can be bundled in a tensor definition. That
    tensor definition can then be constructed by the engines and used in modelling.
    """
    @staticmethod
    def _val_rank_set(tensor_def: 'TensorDefinition'):
        if tensor_def.rank is None:
            raise TensorDefinitionException(
                f'The Rank of Tensor Definition <{tensor_def.name}> has not been set. Can not retrieve it'
            )

    def __init__(self, name: str, features: List[Feature] = None):
        self._name = name
        self._rank = None
        if features is None:
            self._feature_list = []
        else:
            self._features_list = features

    def __len__(self):
        return len(self.features)

    def __repr__(self):
        return f'Tensor Definition : {self.name}'

    @property
    def name(self):
        """Name of the Tensor Definition
        :return: String representation of the name
        """
        return self._name

    @property
    def rank(self) -> int:
        """Returns the rank of this TensorDefinition. The Rank is only known after the Tensor Definition has been used
        to actually build a tensor.

        :return: Rank of the Tensor as int
        """
        self._val_rank_set(self)
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        """Rank setter.

        :param rank: The rank of this Tensor Definition
        :return: None
        """
        self._rank = rank

    @property
    def features(self) -> List[Feature]:
        """Property that lists all features of this tensor definition

        :return: A list of features of the tensor definition
        """
        return self._features_list

    @property
    def embedded_features(self) -> List[Feature]:
        """Function which returns all features embedded in the base features + the base features themselves

        :return: A list of features embedded in the base features + the base features
        """
        base_features = self._features_list
        embedded_features = [features.embedded_features for features in base_features]
        embedded_features_flat = [feature for features in embedded_features for feature in features]
        return list(set(embedded_features_flat + base_features))

    def remove(self, feature: Feature) -> None:
        self._features_list.remove(feature)

    def filter_features(self, category: LearningCategory) -> List[Feature]:
        """Filter features in this Tensor Definition according to a Learning category.

        :param category: The LearningCategory to filter out.
        :return: List of features of the specified 'LearningCategory'
        """
        return [f for f in self.features if f.learning_category == category]

    def categorical_features(self) -> List[Feature]:
        """Return the categorical features in this Tensor Definition.

        :return: List of categorical features in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_CATEGORICAL)

    def binary_features(self) -> List[Feature]:
        """Return the binary features in this Tensor Definition

        :return: List of binary features in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_BINARY)

    def continuous_features(self) -> List[Feature]:
        """Return the continuous feature in this Tensor Definition

        :return:
        """
        return self.filter_features(LEARNING_CATEGORY_CONTINUOUS)

    @property
    def learning_categories(self) -> List[LearningCategory]:
        lcs = [LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_CATEGORICAL]
        return [lc for lc in lcs if len(self.filter_features(lc)) > 0]
