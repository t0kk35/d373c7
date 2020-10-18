"""
History objects. Mainly used during training
(c) 2020 d373c7
"""
import torch
import torch.utils.data as data
from .common import _History
from typing import List, Dict


class TrainHistory(_History):
    loss_key = 'loss'
    acc_key = 'acc'

    def __init__(self, dl: data.DataLoader, metrics: List[str]):
        _History.__init__(self, dl)
        if metrics is None:
            metrics = [TrainHistory.acc_key]
        self._metrics = metrics
        self._history = {m: [] for m in metrics}
        self._running_loss = 0
        self._running_correct_cnt = 0
        self._running_count = 0

    def end_step(self, out: torch.Tensor, y: List[torch.Tensor], loss: torch.Tensor):
        if TrainHistory.loss_key in self._metrics:
            self._running_loss += loss.item()
        if TrainHistory.acc_key in self._metrics:
            p = torch.ge(out, 0.5).view(-1)
            # Assume that the first list contains the labels. This counts how many we got correct
            self._running_correct_cnt += torch.sum(torch.eq(p, y[0])).item()
            self._running_count += out.shape[0]
        super(TrainHistory, self).end_step(out, y, loss)

    def end_epoch(self):
        if TrainHistory.loss_key in self._metrics:
            self._history[TrainHistory.loss_key].append(round(self._running_loss/self.steps, 4))
        if TrainHistory.acc_key in self._metrics:
            self._history[TrainHistory.acc_key].append(round(self._running_correct_cnt/self.samples, 4))
        self._running_correct_cnt = 0
        self._running_count = 0
        self._running_loss = 0
        super(TrainHistory, self).end_epoch()

    @property
    def history(self) -> Dict:
        return self._history

    def step_stats(self) -> Dict:
        r = {
            TrainHistory.loss_key: round(self._running_loss/self.step, 4),
            TrainHistory.acc_key: round(self._running_correct_cnt/self._running_count, 4)
        }
        return r

    def early_break(self) -> bool:
        return False
