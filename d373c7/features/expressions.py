"""
Definition of expression style features.
(c) 2020 d373c7
"""
import logging
from dataclasses import dataclass, field
from ..common import enforce_types
from ..features.common import Feature, FeatureDefinitionException
from ..features.common import LearningCategory
from typing import List, Callable
from inspect import signature, isfunction


logger = logging.getLogger(__name__)


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureExpression(Feature):
    """
    Derived Feature. This is a Feature that will be built off of other features using a function. It can be used
    to perform all sorts of custom operations on other features, such as adding, formatting, calculating ratio's etc...

    The function passed to as expression must be available in the main Python Context
    """
    expression: Callable = field(hash=False)
    param_features: List[Feature] = field(default_factory=list, hash=False)

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

    def _val_expression_not_lambda(self):
        if self.is_lambda:
            raise FeatureDefinitionException(
                f'The expression for series expression feature <{self.name}> can not be a lambda expression. ' +
                f'Lambdas are not serializable during multi-processing. ' +
                f'The "expression" parameter for series expressions must be a function'
            )

    def __post_init__(self):
        # Run post init validation
        self._val_parameters_is_features_list(self.param_features)
        self._val_function_is_callable(self.expression, self.param_features)
        # The parameters needed to run the function are the embedded features
        self.embedded_features.extend(self.param_features)
        for pf in self.param_features:
            self.embedded_features.extend(pf.embedded_features)
        self.embedded_features = list(set(self.embedded_features))

    @property
    def inference_ready(self) -> bool:
        # An expression feature has no inference attributes
        return True

    @property
    def is_lambda(self) -> bool:
        """Flag indicating if the expression that is used to build the feature is a Lambda style callable

        :return: Boolean. True if the expression is a Lambda.
        """
        return self.expression.__name__ == '<lambda>'

    @property
    def learning_category(self) -> LearningCategory:
        # Should be the Learning category of the type of the Expression Feature.
        return self.type.learning_category


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureFilter(FeatureExpression):
    """
    Is a specialisation of the FeatureExpression. It can only output a true or false, so must have a boolean type
    and the function must return a 'bool' type. It can be used to filter certain rows.
    """
    def __post_init__(self):
        # Run post init validation.
        self.val_bool_type()
        self._val_parameters_is_features_list(self.param_features)
        self._val_expression_not_lambda()
        self._val_function_is_callable(self.expression, self.param_features)
        # The parameters needed to run the function are the embedded features
        self.embedded_features.extend(self.param_features)
        for pf in self.param_features:
            self.embedded_features.extend(pf.embedded_features)
        self.embedded_features = list(set(self.embedded_features))


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureExpressionSeries(FeatureExpression):
    def __post_init__(self):
        # Run post init validation.
        self._val_expression_not_lambda()
        self._val_parameters_is_features_list(self.param_features)
        self._val_function_is_callable(self.expression, self.param_features)
        # The parameters needed to run the function are the embedded features
        self.embedded_features.extend(self.param_features)
        for pf in self.param_features:
            self.embedded_features.extend(pf.embedded_features)
        self.embedded_features = list(set(self.embedded_features))
