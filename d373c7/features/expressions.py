"""
Definition of expression style features.
(c) 2020 d373c7
"""
import logging
from ..features.common import Feature, FeatureDefinitionException, FeatureType
from ..features.common import FeatureTypeInteger, FeatureTypeBool, FeatureTypeFloat
from ..features.common import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_NONE, LEARNING_CATEGORY_CATEGORICAL
from ..features.common import LEARNING_CATEGORY_BINARY, LearningCategory
from typing import List, Callable
from inspect import signature, isfunction


logger = logging.getLogger(__name__)


class FeatureExpression(Feature):
    """Derived Feature. This is a Feature that will be built off of other features using a function. It can be used
    to perform all sorts of custom operations on other features, such as adding, formatting, calculating ratio's etc...

    The function passed to as expression must be available in the main Python Context

    Args:
        name: A name for the feature
        f_type: The type of the feature. This must be an instance of the FeatureType class
        expression: A Python Function or Lambda that creates the feature values. (Don't include the brackets)
        param_features: A list of Feature that are the input to the 'expression'
    """
    @staticmethod
    def _val_parameters_is_features_list(param_features: List[Feature]):
        if not isinstance(param_features, List):
            raise FeatureDefinitionException(
                f'Param_features argument to <{FeatureExpression.__name__}> must be a list. ' +
                f'Got <{type(param_features)}>'
            )
        if not len(param_features) == 0 and not isinstance(param_features[0], Feature):
            raise FeatureDefinitionException(
                f'The elements in the param_feature list to <{FeatureExpression.__name__}> must be Feature Objects. ' +
                f'Got <{type(param_features[0])}>'
            )

    @staticmethod
    def _val_function_is_callable(expression: Callable, param_features: List[Feature]):
        if not isfunction(expression):
            raise FeatureDefinitionException(f' Expression parameter must be function')

        expression_signature = signature(expression)
        if len(expression_signature.parameters) != len(param_features):
            raise FeatureDefinitionException(
                f'Number of arguments of function and features do not match '
                f'[{len(expression_signature.parameters)}]  [{len(param_features)}]'
            )

    def __init__(self, name: str, f_type: FeatureType, expression: Callable, param_features: List[Feature]):
        Feature.__init__(self, name, f_type)
        self._val_parameters_is_features_list(param_features)
        self._val_function_is_callable(expression, param_features)
        self._expression = expression
        self._features = param_features

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and self.type == other.type
        else:
            return False

    def __hash__(self):
        return hash(self.type) + hash(self.name)

    def __repr__(self):
        return f'Expression Feature: {self.name}'

    @property
    def expression(self) -> Callable:
        """Returns the expression that is used to build the feature

        :return: A callable (Function or Lambda)
        """
        return self._expression

    @property
    def is_lambda(self) -> bool:
        """Flag indicating if the expression that is used to build the feature is a Lambda style callable

        :return: Boolean. True if the expression is a Lambda.
        """
        return self._expression.__name__ == '<lambda>'

    @property
    def param_features(self) -> List[Feature]:
        """Returns a list of features that need to be fed as parameters to the expression.

        :return: List of 'Feature' objects.
        """
        return self._features

    @property
    def embedded_features(self) -> List[Feature]:
        return self._features

    @property
    def learning_category(self) -> LearningCategory:
        if isinstance(self.type, FeatureTypeBool):
            return LEARNING_CATEGORY_BINARY
        elif isinstance(self.type, FeatureTypeFloat):
            return LEARNING_CATEGORY_CONTINUOUS
        elif isinstance(self.type, FeatureTypeInteger):
            return LEARNING_CATEGORY_CATEGORICAL
        else:
            return LEARNING_CATEGORY_NONE


class FeatureExpressionSeries(FeatureExpression):
    def _val_expression_not_lambda(self):
        if self.is_lambda:
            raise FeatureDefinitionException(
                f'The expression for series expression feature <{self.name}> can not be a lambda expression. ' +
                f'Lambdas are not serializable during multi-processing. ' +
                f'The "expression" parameter for series expressions must be a function'
            )

    def __init__(self, name: str, f_type: FeatureType, expression: Callable, features: List[Feature]):
        super(FeatureExpressionSeries, self).__init__(name, f_type, expression, features)
        self._val_expression_not_lambda()
