"""
Common classes for all Pytorch Models
(c) 2020 d373c7
"""

import torch
import torch.nn as nn
from ..loss import _LossBase
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

    def get_optimizer(self) -> torch.optim.optimizer.Optimizer:
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
