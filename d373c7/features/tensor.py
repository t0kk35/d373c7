"""
Tensor Definition classes. For grouping features.
(c) 2020 d373c7
"""
from .common import LearningCategory, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_LABEL
from .common import LEARNING_CATEGORY_CONTINUOUS
from .common import FeatureTypeNumerical
from .base import Feature, FeatureIndex, FeatureInferenceAttributes
from .expanders import FeatureOneHot, FeatureExpander
from typing import List

LEARNING_CATEGORIES = [
    LEARNING_CATEGORY_BINARY,
    LEARNING_CATEGORY_CONTINUOUS,
    LEARNING_CATEGORY_CATEGORICAL,
    LEARNING_CATEGORY_LABEL
]


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
    def _val_rank_set(self):
        if self._rank is None:
            raise TensorDefinitionException(
                f'The Rank of Tensor Definition <{self.name}> has not been set. Can not retrieve it'
            )

    def _val_labels_defined(self, labels: List[Feature]):
        for lb in labels:
            if lb not in self.features:
                raise TensorDefinitionException(
                    f'Label <{lb.name}> does not exist in tensor definition <{self.name}>'
                )

    def _val_duplicate_entries(self):
        if len(list(set(self.features))) != len(self.features):
            raise TensorDefinitionException(
                f'Tensor definition has duplicate entries <{[f for f in self.features if self.features.count(f) > 1]}>'
            )

    def _val_not_empty(self):
        if len(self.features) == 0:
            raise TensorDefinitionException(
                f'Tensor definition <{self.name} has no features. It is empty. Can not perform action'
            )

    def _val_has_numerical_features(self):
        f = [f for f in self.features if isinstance(f.type, FeatureTypeNumerical)]
        if len(f) == 0:
            raise TensorDefinitionException(
                f'Tensor definition <{self.name} has no numerical features. It is empty. Can not perform action'
            )

    def _val_base_feature_overlap(self):
        fi = set([f.base_feature for f in self.features if isinstance(f, FeatureIndex)])
        fo = set([f.base_feature for f in self.features if isinstance(f, FeatureOneHot)])
        s = fi.intersection(fo)
        if len(s) != 0:
            raise TensorDefinitionException(
                f'FeatureIndex and FeatureOneHot should not have the same base features. Overlap <{s}>'
            )

    def _val_inference_ready(self, operation: str):
        if not self.inference_ready:
            raise TensorDefinitionException(
                f'Tensor is not ready for inference. Can not perform operation <{operation}>'
            )

    def __init__(self, name: str, features: List[Feature] = None):
        self._name = name
        self._rank = None
        if features is None:
            self._feature_list = []
        else:
            self._features_list = features
        self._val_duplicate_entries()
        self._val_base_feature_overlap()
        self._labels = []

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
        self._val_rank_set()
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

    @property
    def inference_ready(self) -> bool:
        """Method that return True if the Tensor is ready for inference. It means it knows it's own rank and all
        embedded features have their inference attributes set.

        :return: Bool. True or False Indicating if the tensor is ready or not for inference.
        """
        if self._rank is None:
            return False
        else:
            for f in self.embedded_features:
                if isinstance(f, FeatureInferenceAttributes):
                    if not f.inference_ready:
                        return False
        return True

    @property
    def learning_categories(self) -> List[LearningCategory]:
        res = [lc for lc in LEARNING_CATEGORIES if len(self.filter_features(lc)) > 0]
        return res

    @staticmethod
    def _expand_features(features: List[Feature]) -> List[Feature]:
        r = []
        for f in features:
            if isinstance(f, FeatureExpander):
                r.extend(f.expand())
            else:
                r.append(f)
        return r

    def set_label(self, label: Feature) -> 'TensorDefinition':
        """Define which feature in this Tensor Definition will be used as label for training

        :param: The feature that should be use a training target.
        """
        self._val_labels_defined([label])
        self._labels.append(label)
        return self

    @property
    def highest_precision_feature(self) -> Feature:
        """Return the highest precision (numerical) feature in this Tensor Definition.

        :return: The feature with the highest precision
        """
        self._val_has_numerical_features()
        t = [f for f in self.features if isinstance(f.type, FeatureTypeNumerical)]
        t.sort(key=lambda x: x.type.precision)
        # Last has biggest precision
        return t[-1]

    def set_labels(self, labels: List[Feature]):
        """Define which of the features in the Tensor Definition will be used as label for training.

        :param labels: List of feature that need to treated as labels during training.
        """
        self._val_labels_defined(labels)
        self._labels.extend(labels)

    def remove(self, feature: Feature) -> None:
        self._features_list.remove(feature)

    def filter_features(self, category: LearningCategory, expand=False) -> List[Feature]:
        """Filter features in this Tensor Definition according to a Learning category.
        NOTE that with expand 'True', the Tensor Definition must be ready for inference.

        :param category: The LearningCategory to filter out.
        :param expand: Bool value. True or False indicating if the names of expander features will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False. NOTE that with expand 'True', the Tensor Definition must be ready for inference.
        :return: List of features of the specified 'LearningCategory'
        """
        if expand:
            self._val_inference_ready('filter ' + category.name)
        if category == LEARNING_CATEGORY_LABEL:
            return self._labels
        else:
            r = [f for f in self.features if f.learning_category == category and f not in self._labels]
            if expand:
                r = TensorDefinition._expand_features(r)
            return r

    def categorical_features(self, expand=False) -> List[Feature]:
        """Return the categorical features in this Tensor Definition.

        :param expand: Bool value. True or False indicating if the names of expander features will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False
        :return: List of categorical features in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_CATEGORICAL, expand)

    def binary_features(self, expand=False) -> List[Feature]:
        """Return the binary features in this Tensor Definition

        :param expand: Bool value. True or False indicating if the names of expander features will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False
        :return: List of binary features in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_BINARY, expand)

    def continuous_features(self, expand=False) -> List[Feature]:
        """Return the continuous feature in this Tensor Definition

        :param expand: Bool value. True or False indicating if the names of expander features will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False
        :return: List of continuous features in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_CONTINUOUS, expand)

    def label_features(self, expand=False) -> List[Feature]:
        """Return the label feature in this Tensor Definition

        :param expand: Bool value. True or False indicating if the names of expander features will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False
        :return: List of label features in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_LABEL, expand)
