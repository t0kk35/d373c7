"""
Module for Layers that can be used as body in the _Module objects.
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from .common import _Layer, PyTorchLayerException
from typing import List, Union


class BodyMulti(_Layer):
    def __init__(self, head: _Layer, layers: List[Union[_Layer, None]]):
        super(BodyMulti, self).__init__()
        self.layers = nn.ModuleList(layers)
        self._output_size = sum([h.output_size if ly is None else ly.output_size for ly, h in zip(layers, head.heads)])

    def forward(self, x):
        # Run head through Layers if the layer is NOT None (i.e. it's a series) else just return head
        x = [xi if ly is None else ly(xi) for ly, xi in zip(self.layers, x)]
        # Concatenate the whole bunch into one big rank 2 tensor.
        x = torch.cat(x, dim=1)
        return x

    @property
    def output_size(self) -> int:
        return self._output_size


class _RecurrentBody(_Layer):
    def __init__(self, in_size: int, recurrent_features: int, recurrent_layers: int, dense: bool, batch_norm: bool):
        super(_RecurrentBody, self).__init__()
        self._in_size = in_size
        self._recurrent_features = recurrent_features
        self._recurrent_layers = recurrent_layers
        self._dense = dense
        self.rnn = self.init_rnn()
        self.bn = nn.BatchNorm1d(self.output_size) if batch_norm else None

    def init_rnn(self):
        raise NotImplemented('Should be implemented by children')

    def forward(self, x):
        y, _ = self.rnn(x)
        # Take last output of recurrent layer
        y = y[:, -1, :]
        if self._dense:
            # Add full last original input entry
            y = torch.cat([y, x[:, -1, :]], dim=1)
        if self.bn is not None:
            y = self.bn(y)
        return y

    @property
    def output_size(self) -> int:
        if self._dense:
            return self._recurrent_features + self._in_size
        else:
            return self._recurrent_features


class LSTMBody(_RecurrentBody):
    def __init__(self, in_size: int, recurrent_features: int, recurrent_layers: int, dense: bool, batch_norm: bool):
        super(LSTMBody, self).__init__(in_size, recurrent_features, recurrent_layers, dense, batch_norm)

    def init_rnn(self):
        rnn = nn.LSTM(self._in_size, self._recurrent_features, self._recurrent_layers, batch_first=True)
        return rnn


class GRUBody(_RecurrentBody):
    def __init__(self, in_size, recurrent_features: int, recurrent_layers: int, dense: bool, batch_norm: bool):
        super(GRUBody, self).__init__(in_size, recurrent_features, recurrent_layers, dense, batch_norm)

    def init_rnn(self):
        rnn = nn.GRU(self._in_size, self._recurrent_features, self._recurrent_layers, batch_first=True)
        return rnn
