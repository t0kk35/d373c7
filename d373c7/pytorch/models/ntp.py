"""
Module for Next Transaction Prediction Models
(c) 2020 d373c7
"""
import logging
import numpy as np
import torch
from .common import _Model, ModelDefaults, PyTorchModelException, TensorDefinitionHead
from ..common import _History


logger = logging.getLogger(__name__)


class NtpEncoderHistory(_History):
    loss_key = 'loss'

    def __init__(self, *args):
        dl = self._val_argument(args)
        h = {m: [] for m in [NtpEncoderHistory.loss_key]}
        _History.__init__(self, dl, h)
        self._running_loss = 0

    def end_step(self, *args):
        NtpEncoderHistory._val_is_tensor_list(args[1])
        NtpEncoderHistory._val_is_tensor(args[2])
        loss = args[2]
        self._running_loss += loss.item()
        super(NtpEncoderHistory, self).end_step(loss)

    def end_epoch(self):
        self._history[NtpEncoderHistory.loss_key].append(round(self._running_loss/self.steps, 4))
        self._running_loss = 0
        super(NtpEncoderHistory, self).end_epoch()


class NtpDefaults(ModelDefaults):
    def __init__(self):
        super(NtpDefaults, self).__init__()
