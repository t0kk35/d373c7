"""
Module for classifier Models
(c) 2020 d373c7
"""
import logging
import torch
from .common import _Model, _LossBase, PyTorchModelException, ModelDefaults, _TensorHeadModel
from ..layers import SingleClassBinaryOutput, LinDropAct
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

    def __init__(self, tensor_def: TensorDefinition, defaults: ModelDefaults):
        super(_ClassifierModel, self).__init__(tensor_def, defaults)
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


class ClassifierDefaults(ModelDefaults):
    def __init__(self):
        super(ClassifierDefaults, self).__init__()
        self.emb_dim(4, 100, 0.2)
        self.set('lin_interlayer_drop_out', 0.1)

    def emb_dim(self, minimum: int, maximum: int, dropout: float):
        self.set('emb_min_dim', minimum)
        self.set('emb_max_dim', maximum)
        self.set('emb_dropout', dropout)


class FeedForwardFraudClassifier(_ClassifierModel, _TensorHeadModel):
    def __init__(self, tensor_def: TensorDefinition, defaults=ClassifierDefaults()):
        super(FeedForwardFraudClassifier, self).__init__(tensor_def, defaults)
        do = self.defaults.get_float('lin_interlayer_drop_out')
        self.linear = LinDropAct(self.head.output_size, [(16, do)])
        self.out = SingleClassBinaryOutput(self.linear.output_size)
        self._loss_fn = SingleLabelBCELoss()

    def forward(self, x):
        x = _TensorHeadModel.forward(self, x)
        x = self.linear(x)
        x = self.out(x)
        return x

    def loss_fn(self) -> _LossBase:
        return self._loss_fn
