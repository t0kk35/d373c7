"""
Module for encoder Models
(c) 2020 d373c7
"""
import logging
import torch
from .common import _Model, _LossBase, PyTorchModelException
from ..optimizer import _Optimizer, AdamWOptimizer
from ...features import TensorDefinition, LEARNING_CATEGORY_LABEL, LEARNING_CATEGORY_BINARY
from ...features import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_CATEGORICAL
from typing import List


class EncoderModel(_Model):
    def __init__(self, tensor_def: TensorDefinition):
        super(EncoderModel, self).__init__(tensor_def)

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # With encoders the assume all lists a input
        return ds

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # With encoders the assume all lists a output (also)
        return ds

    @property
    def default_metrics(self) -> List[str]:
        return ['loss']
