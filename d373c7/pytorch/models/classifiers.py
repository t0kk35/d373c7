"""
Module for classifier Models
(c) 2020 d373c7
"""
import logging
import torch
from .common import _Model, _LossBase, PyTorchModelException
from ..layers import SingleClassBinaryOutput, TensorDefinitionHead, LinDropAct
from ..optimizer import _Optimizer, AdamWOptimizer
from ..loss import SingleLabelBCELoss
from ...features import TensorDefinition, LEARNING_CATEGORY_LABEL
from typing import List


logger = logging.getLogger(__name__)


class _ClassifierModel(_Model):
    @staticmethod
    def _val_has_lc_label(tensor_def: TensorDefinition):
        if LEARNING_CATEGORY_LABEL not in tensor_def.learning_categories:
            raise PyTorchModelException(
                f'Tensor Definition <{tensor_def.name}> does not have a label learning category. '
                f'Can not build a classifier without a label. Please the .set_label(xyz) on the tensor definition'
            )

    def __init__(self, tensor_def: TensorDefinition):
        super(_ClassifierModel, self).__init__(tensor_def)
        _ClassifierModel._val_has_lc_label(tensor_def)
        self._label_index = tensor_def.learning_categories.index(LEARNING_CATEGORY_LABEL)

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # Return the label with Learning Category 'LEARNING_CATEGORY_LABEL'
        return ds[self._label_index: self._label_index+1]

    def optimizer(self, lr=None, wd=None) -> _Optimizer:
        return AdamWOptimizer(self, lr, wd)

    @property
    def default_metrics(self) -> List[str]:
        return ['acc', 'loss']


class _TensorHeadModel(_Model):
    def __init__(self, tensor_def: TensorDefinition):
        super(_TensorHeadModel, self).__init__(tensor_def)
        self.head = TensorDefinitionHead(tensor_def)
        
    def forward(self, x):
        x = self.head(x)
        return x

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        x = [ds[x] for x in self.head.x_indexes]
        return x


class FeedForwardFraudClassifier(_ClassifierModel, _TensorHeadModel):
    def __init__(self, tensor_def: TensorDefinition):
        super(FeedForwardFraudClassifier, self).__init__(tensor_def)
        self.linear = LinDropAct(self.head.output_size, [(16, 0.0)])
        self.out = SingleClassBinaryOutput(self.linear.output_size)
        self._loss_fn = SingleLabelBCELoss()

    def forward(self, x):
        x = _TensorHeadModel.forward(self, x)
        x = self.linear(x)
        x = self.out(x)
        return x

    def loss_fn(self) -> _LossBase:
        return self._loss_fn
