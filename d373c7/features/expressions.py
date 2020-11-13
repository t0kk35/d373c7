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
    @staticmethod
    def _val_function_is_callable(expression: Callable, arguments: List[Feature]):
        if not isfunction(expression):
            raise FeatureDefinitionException(f' Expression parameter must be function')

        expression_signature = signature(expression)
        if len(expression_signature.parameters) != len(arguments):
            raise FeatureDefinitionException(
                f'Number of arguments of function and features do not match '
                f'[{len(expression_signature.parameters)}]  [{len(arguments)}]'
            )

    def __init__(self, name: str, f_type: FeatureType, expression: Callable, features: List[Feature]):
        Feature.__init__(self, name, f_type)
        self._val_function_is_callable(expression, features)
        self._expression = expression
        self._features = features

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
        return self._expression

    @property
    def is_lambda(self) -> bool:
        return self._expression.__name__ == '<lambda>'

    @property
    def param_features(self) -> List[Feature]:
        return self._features

    @property
    def embedded_features(self) -> List[Feature]:
        return self._features

    @property
    def learning_category(self) -> LearningCategory:
        if self.type == FeatureTypeBool:
            return LEARNING_CATEGORY_BINARY
        elif self.type == FeatureTypeFloat:
            return LEARNING_CATEGORY_CONTINUOUS
        elif self.type == FeatureTypeInteger:
            return LEARNING_CATEGORY_CATEGORICAL
        else:
            return LEARNING_CATEGORY_NONE
