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
from typing import List, Any


class _Model(nn.Module):
    def __init__(self, loss: _LossBase):
        nn.Module.__init__(self)
        self._loss_fn = loss

    def _forward_unimplemented(self, *inp: Any) -> None:
        raise NotImplemented('Abstract method _forward_unimplemented not implemented')

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        pass

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        pass

    def get_optimizer(self, lr=None, wd=None) -> _Optimizer:
        pass

    @property
    def loss_fn(self) -> _LossBase:
        return self._loss_fn

    @property
    def default_metrics(self) -> List[str]:
        raise NotImplemented()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
