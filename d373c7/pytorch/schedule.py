"""
Module for custom schedulers
(c) 2020 d373c7
"""
import logging
import torch
import torch.utils.data as data
from .optimizer import _Optimizer
from .common import _History
from typing import List


logger = logging.getLogger(__name__)


# noinspection PyProtectedMember,PyUnresolvedReferences
class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    # inspection PyProtectedMember,PyUnresolvedReferences
    @staticmethod
    def _val_more_than_1_iter(num_iter: int):
        if num_iter <= 1:
            raise PyTorchTrainException(f'Number if iterations must be > 1. Got {num_iter}')

    def __init__(self, end_lr: float, num_iter: int, optimizer: _Optimizer, last_epoch=-1):
        LinearLR._val_more_than_1_iter(num_iter)
        self._end_lr = end_lr
        self._num_iter = num_iter
        torch.optim.lr_scheduler._LRScheduler.__init__(self, optimizer.optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        r = self.last_epoch / (self._num_iter - 1)
        return [base_lr + r * (self._end_lr - base_lr) for base_lr in self.base_lrs]


class LRHistory(_History):
    lr_key = 'lr'
    loss_key = 'loss'

    def __init__(self, dl: data.DataLoader, scheduler: LinearLR, diverge: int, smooth: float, max_steps: int):
        h = {LRHistory.lr_key: [], LRHistory.loss_key: []}
        _History.__init__(self, dl, h)
        self._step_count = 0
        self._scheduler = scheduler
        self._diverge = diverge
        self._smooth = smooth
        self._max_steps = max_steps
        self._best_loss = float('inf')

    def end_step(self, o: torch.Tensor, y: List[torch.Tensor], loss: torch.Tensor):
        if self._step_count < 1:
            ls = loss.item()
        else:
            ls = self._smooth * loss.item() + (1-self._smooth) * self._history[LRHistory.loss_key][-1]
        self._history[LRHistory.lr_key].append(self._scheduler.get_lr()[0])
        self._history[LRHistory.loss_key].append(ls)
        self._best_loss = ls if ls < self._best_loss else self._best_loss
        self._step_count += 1

    def end_epoch(self):
        raise NotImplemented('end_epoch not implemented for LR-History')

    def early_break(self) -> bool:
        loss_diverged = True if self._history[LRHistory.loss_key][-1] > self._best_loss * self._diverge else False
        max_iter_reached = True if self._step_count >= self._max_steps else False
        return loss_diverged or max_iter_reached
