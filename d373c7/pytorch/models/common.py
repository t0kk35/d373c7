"""
Common classes for all Pytorch Models
(c) 2020 d373c7
"""

import os
import torch
import torch.nn as nn
from ..common import PyTorchTrainException
from ..loss import _LossBase
from ..optimizer import _Optimizer
from ..layers import Embedding, SingleClassBinaryOutput
from ...features.tensor import TensorDefinition
from ...features.common import LearningCategory, LEARNING_CATEGORY_LABEL, LEARNING_CATEGORY_BINARY
from ...features import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_CATEGORICAL
from typing import List, Any


class PyTorchModelException(Exception):
    """Standard exception raised during training"""
    def __init__(self, message: str):
        super().__init__('Error in Model: ' + message)


class _Model(nn.Module):
    def __init__(self, tensor_def: TensorDefinition):
        nn.Module.__init__(self)
        self._tensor_def = tensor_def

    def _forward_unimplemented(self, *inp: Any) -> None:
        raise NotImplemented('Abstract method _forward_unimplemented not implemented')

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        """Get the values that are considered the x values. I.e the independent variables, I.e. NOT the label.

        :param ds: A list of tensors as read from a DataLoader object.
        :return: A list of tensors to be used as input to a neural net.
        """
        pass

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        """Get the values that are considered the y values. I.e. the dependent variable, I.e. the label

        :param ds: A list of tensors as read from a DataLoader object.
        :return: A list of tensors to be use as label for the neural net.
        """
        pass

    @property
    def tensor_definition(self):
        return self._tensor_def

    @property
    def loss_fn(self) -> _LossBase:
        raise NotImplemented(f'Loss Function getter not implemented in base _Model Class. '
                             f'Needs to be implemented by the child classes')

    @property
    def default_metrics(self) -> List[str]:
        raise NotImplemented()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def optimizer(self, lr=None, wd=None) -> _Optimizer:
        pass


class _ModelManager:
    def __init__(self, model: _Model, device: torch.device):
        self._model = model
        self._device = device

    @staticmethod
    def _get_x(model: _Model, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        return model.get_x(ds)

    @staticmethod
    def _get_y(model: _Model, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        return model.get_y(ds)

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def save(self, path: str):
        if os.path.exists(path):
            raise PyTorchTrainException(f'File {path} already exists. Not overriding model')
        torch.save(self._model.state_dict(), path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise PyTorchTrainException(f'File {path} does not exist. Not loading model')
        self._model.load_state_dict(torch.load(path))
