"""
Tensor Definition classes. For grouping features.
(c) 2020 d373c7
"""
from .common import LearningCategory, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_LABEL
from .common import LEARNING_CATEGORIES_MODEL
from .common import LEARNING_CATEGORY_CONTINUOUS
from .common import FeatureTypeNumerical, FeatureHelper
from .base import Feature, FeatureIndex
from .expanders import FeatureOneHot, FeatureExpander
from typing import List, Union, Tuple


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

    Args:
        name: A name for this tensor definition
        features: A list of features to group in this feature definition
    """
    def _val_rank_set(self):
        if self._rank is None:
            raise TensorDefinitionException(
                f'The Rank of Tensor Definition <{self.name}> has not been set. Can not retrieve it. Maybe ' +
                f'it needs to be used in a run with no "Inference"'
            )

    def _val_shapes_set(self):
        if self._shapes is None:
            raise TensorDefinitionException(
                f'The Shape of Tensor Definition <{self.name}> has not been set. Can not retrieve it. Maybe ' +
                f'it needs to be used in a run with no "Inference"'
            )

    def _val_shapes_match_lcs(self, shapes: List[Tuple[int, ...]]):
        if len(shapes) != len(self.learning_categories):
            raise TensorDefinitionException(
                f'The number of shapes <{len(shapes)}> is not the same as the number of learning categories for ' +
                f'this TensorDefinition <{self.learning_categories}>'
            )

    def _val_duplicate_entries(self):
        names = [f.name for f in self.features]
        if len(list(set(names))) != len(names):
            raise TensorDefinitionException(
                f'Tensor definition has duplicate entries ' +
                f' <{[n for n in names if names.count(n) > 1]}>'
            )

    def _val_not_empty(self):
        if len(self.features) == 0:
            raise TensorDefinitionException(
                f'Tensor definition <{self.name} has no features. It is empty. Can not perform action'
            )

    def _val_has_numerical_features(self):
        if len(FeatureHelper.filter_feature_type(FeatureTypeNumerical, self.features)) == 0:
            raise TensorDefinitionException(
                f'Tensor definition <{self.name} has no numerical features. Can not perform action'
            )

    def _val_base_feature_overlap(self):
        fi = set([f.base_feature for f in FeatureHelper.filter_feature(FeatureIndex, self.features)])
        fo = set([f.base_feature for f in FeatureHelper.filter_feature(FeatureOneHot, self.features)])
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
        self._shapes = None
        if features is None:
            self._feature_list = []
        else:
            self._features_list = features
        self._val_duplicate_entries()
        self._val_base_feature_overlap()

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
    def shapes(self) -> List[Tuple[int, ...]]:
        """Returns the 'expected' shape of this TensorDefinition. The Shape is only known after the Tensor Definition
        has been used to build a Numpy List.
        Because a Tensor definition can not know the batch size, the first dimension is *hard-coded to -1*.

        :return: A List of int Tuples. There will be a tuple per each learning category. The tuples will contain an int
        for each dimension, each int is the size along that dimension.
        """
        self._val_shapes_set()
        return self._shapes

    @shapes.setter
    def shapes(self, shapes: List[Tuple[int, ...]]):
        """Shape setter.

        :param shapes: A Tuple of ints describing the length along the various dimensions.
        :return: None
        """
        self._val_shapes_match_lcs(shapes)
        self._shapes = shapes

    @property
    def features(self) -> List[Feature]:
        """Property that lists all features of this tensor definition

        :return: A list of features of the tensor definition
        """
        return self._features_list

    @property
    def embedded_features(self) -> List[Feature]:
        """
        Function which returns all features embedded in the base features + the base features themselves. It effectively
        returns all features referenced in this Tensor Definition.

        @return: A list of features embedded in the base features + the base features
        """
        base_features = self._features_list
        embedded_features = [features.embedded_features for features in base_features]
        embedded_features_flat = [feature for features in embedded_features for feature in features]
        return list(set(embedded_features_flat + base_features))

    @property
    def inference_ready(self) -> bool:
        """Method that return True if the Tensor is ready for inference. It means  all embedded features are ready fpr
        inference, they either have no inference attributes or their inference attributes are set.

        :return: Bool. True or False Indicating if the tensor is ready or not for inference.
        """
        return all([f.inference_ready for f in self.embedded_features])

    @property
    def learning_categories(self) -> List[LearningCategory]:
        res = [lc for lc in LEARNING_CATEGORIES_MODEL if len(self.filter_features(lc)) > 0]
        return res

    @staticmethod
    def _expand_features(features: List[Feature]) -> List[Feature]:
        r = []
        for f in features:
            if FeatureHelper.is_feature(f, FeatureExpander):
                r.extend(f.expand())
            else:
                r.append(f)
        return r

    @property
    def highest_precision_feature(self) -> Feature:
        """
        Return the highest precision (numerical) feature in this Tensor Definition.

        :return: The feature with the highest precision
        """
        self._val_has_numerical_features()
        t = FeatureHelper.filter_feature_type(FeatureTypeNumerical, self.features)
        t.sort(key=lambda x: x.type.precision)
        # Last one has the biggest precision
        return t[-1]

    def remove(self, feature: Feature) -> None:
        self._features_list.remove(feature)

    def filter_features(self, category: LearningCategory, expand=False) -> List[Feature]:
        """
        Filter features in this Tensor Definition according to a Learning category.
        NOTE that with expand 'True', the Tensor Definition must be ready for inference.

        :param category: The LearningCategory to filter out.
        :param expand: Bool value. True or False indicating if the names of expander features will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False. NOTE that with expand 'True', the Tensor Definition must be ready for inference.
        :return: List of features of the specified 'LearningCategory'
        """
        if expand:
            self._val_inference_ready('filter ' + category.name)

        r = [f for f in self.features if f.learning_category == category]
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

    def features_not_inference_ready(self) -> List[Feature]:
        """List features of this TensorDefinition which are not ready for inference

        :return: A list of features that returned False to the inference_ready call
        """
        return [f for f in self.embedded_features if f.inference_ready]


class TensorDefinitionMulti:
    """Class for Multi-Head TensorDefinitions. They basically hold multiple Tensor Definitions.

    Args:
        tensor_def: List of TensorDefinitions to be bundled in the Multi-Head Tensor Definition.
    """
    @staticmethod
    def _val_max_one_def_with_label(tensor_def: List[TensorDefinition]):
        tds = [td for td in tensor_def if LEARNING_CATEGORY_LABEL in td.learning_categories]
        if len(tds) > 1:
            raise TensorDefinitionException(
                f'A TensorDefinitionMulti should only have one embedded TensorDefinition with Labels. ' +
                f'These TensorDefinitions had a label {[td.name for td in tds]}'
            )

    def __init__(self, tensor_def: List[TensorDefinition]):
        TensorDefinitionMulti._val_max_one_def_with_label(tensor_def)
        self._tensor_def = tensor_def

    @property
    def tensor_definitions(self) -> List[TensorDefinition]:
        return self._tensor_def

    @property
    def label_tensor_definition(self) -> Union[TensorDefinition, None]:
        """Returns the TensorDefinition out of the various TensorDefinition that has the labels.

        :return: The TensorDefinition has contains the labels
        """
        tds = [td for td in self.tensor_definitions if LEARNING_CATEGORY_LABEL in td.learning_categories]
        if len(tds) == 0:
            return None
        else:
            return tds[0]

    @property
    def label_index(self) -> int:
        """Return the 'Learning Category' index of the label in the TensorDefinitionMulti

        :return: The index at which the Label should normally be found in for instance a NumpyList
        """
        label_td = self.label_tensor_definition
        if label_td is None:
            raise TensorDefinitionException(
                f'Could not find a TensorDefinition containing a {LEARNING_CATEGORY_LABEL}. Can not determine index'
            )
        off_sets: List[int] = [0]
        for td in self.tensor_definitions:
            off_sets.append(len(td.learning_categories) + off_sets[-1])
        off_sets = off_sets[:-1]
        label_index = label_td.learning_categories.index(LEARNING_CATEGORY_LABEL)
        label_index += off_sets[self.tensor_definitions.index(label_td)]
        return label_index
