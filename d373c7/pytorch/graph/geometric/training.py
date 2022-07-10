"""
Module for generic network training stuff.
(c) 2022 d373c7
"""
import logging
import torch

from ...common import _History, PyTorchTrainException

from typing import Dict

logger = logging.getLogger(__name__)


class GraphBinaryClassifierHistory:
    loss_key = 'loss'
    acc_key = 'acc'

    def __init__(self):
        self._history = {m: [] for m in [GraphBinaryClassifierHistory.loss_key, GraphBinaryClassifierHistory.acc_key]}
        self._running_loss = 0
        self._running_correct_cnt = 0
        self._running_count = 0
        self.steps = 1
        self.step = 1
        self._epoch = 0

    @staticmethod
    def _reshape_label(pr: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        if pr.shape == lb.shape:
            return lb
        elif len(pr.shape)-1 == len(lb.shape) and pr.shape[-1] == 1:
            return torch.unsqueeze(lb, dim=len(pr.shape)-1)
        else:
            raise PyTorchTrainException(
                f'Incompatible shapes for prediction and label. Got {pr.shape} and {lb.shape}. Can not safely compare'
            )

    def end_step(self, *args):
        # GraphBinaryClassifierHistory._val_is_tensor(args[0])
        # GraphBinaryClassifierHistory._val_is_tensor_list(args[1])
        # GraphBinaryClassifierHistory._val_is_tensor(args[2])
        pr, lb, loss, mask = args[0], args[1], args[2], args[3]
        self._running_loss += loss.item()
        self._running_correct_cnt += torch.sum(torch.eq(torch.ge(pr[mask], 0.5), lb[0][mask])).item()
        self._running_count += lb[0][mask].shape[0]
        # super(GraphBinaryClassifierHistory, self).end_step(pr, lb, loss)

    def end_epoch(self):
        self._history[GraphBinaryClassifierHistory.loss_key].append(round(self._running_loss/self.steps, 4))
        self._history[GraphBinaryClassifierHistory.acc_key].append(
            round(self._running_correct_cnt/self._running_count, 4)
        )
        self._running_correct_cnt = 0
        self._running_count = 0
        self._running_loss = 0
        self._epoch += 1

    def step_stats(self) -> Dict:
        r = {
            GraphBinaryClassifierHistory.loss_key: round(self._running_loss/self.step, 4),
            GraphBinaryClassifierHistory.acc_key: round(self._running_correct_cnt/self._running_count, 4)
        }
        return r

    @property
    def epoch(self):
        return self._epoch

    @property
    def history(self) -> Dict:
        return self._history

    @staticmethod
    def early_break() -> bool:
        return False
