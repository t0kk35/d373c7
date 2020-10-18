"""
Module for custom optimizers
(c) 2020 d373c7
"""
import torch.optim as opt
import torch.nn as nn


class _Optimizer:
    def zero_grad(self):
        pass

    def step(self):
        pass

    @property
    def optimizer(self) -> opt.Optimizer:
        raise NotImplemented('To be implemented by child classes')

    @property
    def lr(self) -> float:
        raise NotImplemented('To be implemented by child classes')


class AdamWOptimizer(_Optimizer):
    def __init__(self, model: nn.Module, lr: float, wd: float = 1e-2):
        lr = lr if lr is not None else 1e-3
        wd = wd if wd is not None else 1e-2
        self._opt = opt.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def zero_grad(self):
        self._opt.zero_grad()

    def step(self):
        self._opt.step()

    @property
    def optimizer(self) -> opt.Optimizer:
        return self._opt

    @property
    def lr(self) -> float:
        return self._opt.param_groups[0]['lr']
