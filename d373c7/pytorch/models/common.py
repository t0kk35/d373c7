"""
Common classes for all Pytorch Models
(c) 2020 d373c7
"""
from abc import ABC

import torch
import torch.nn as nn
from typing import List


class _LossFunction:
    """Loss function object"""
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented('Call not implemented for loss function')

    def score(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented('Call not implemented for loss function')


class _TrainableModel(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        pass

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        pass

    def get_optimizer(self) -> torch.optim.optimizer.Optimizer:
        pass

    def get_loss(self) -> _LossFunction:
        pass
