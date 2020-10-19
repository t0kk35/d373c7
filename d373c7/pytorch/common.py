"""
Imports for Pytorch Training functions
(c) 2020 d373c7
"""
import torch
import torch.utils.data as data
# noinspection PyProtectedMember
from typing import List, Dict


class PyTorchTrainException(Exception):
    """Standard exception raised during training"""
    def __init__(self, message: str):
        super().__init__('Error in PyTorch Training: ' + message)


class _History:
    """Object that keeps track of metrics during training and testing

    :argument dl: A Data loader that will be iterated over in the training or validation loop.
    """
    def __init__(self, dl: data.DataLoader):
        self._batch_size = dl.batch_size
        self._samples = len(dl.dataset)
        self._step = 0
        self._steps = len(dl)
        self._epoch = 0

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def history(self) -> Dict:
        raise NotImplemented('history property not implemented')

    @property
    def step(self) -> int:
        return self._step

    def start_step(self):
        self._step += 1

    def end_step(self, o: torch.Tensor, y: List[torch.Tensor], loss: torch.Tensor):
        pass

    def early_break(self) -> bool:
        pass

    def start_epoch(self):
        self._step = 0
        self._epoch += 1

    def end_epoch(self):
        pass
