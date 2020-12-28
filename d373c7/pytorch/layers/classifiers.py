"""
Module for Layers that can be used as body in the _Module objects.
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from .common import _Layer, PyTorchLayerException
from .base import PositionalEncoding, PositionalEmbedding, ConvolutionalBodyBase1d
from collections import OrderedDict
from typing import List, Union, Tuple


class BodyMulti(_Layer):
    def __init__(self, head: _Layer, layers: List[Union[_Layer, None]]):
        super(BodyMulti, self).__init__()
        self.layers = nn.ModuleList(layers)
        self._output_size = sum([h.output_size if ly is None else ly.output_size for ly, h in zip(layers, head.heads)])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # Run head through Layers if the layer is NOT None (i.e. it's a series) else just return head
        x = [xi if ly is None else ly(xi) for ly, xi in zip(self.layers, x)]
        # Concatenate the whole bunch into one big rank 2 tensor.
        x = torch.cat(x, dim=1)
        return x

    @property
    def output_size(self) -> int:
        return self._output_size


class BodySequential(_Layer):
    def __init__(self, in_size: int, layers: List[_Layer]):
        super(BodySequential, self).__init__()
        self._in_size = in_size
        self._out_size = layers[-1].output_size
        self.layers = nn.Sequential(OrderedDict({f'seq_{i:02d}': l for i, l in enumerate(layers)}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    @property
    def output_size(self) -> int:
        return self._out_size


class BodyWithAttention(BodySequential):
    def __init__(self, in_size, attention: _Layer, layers: List[_Layer]):
        super(BodyWithAttention, self).__init__(in_size, [attention]+layers)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        _, w = self.layers[0](x, return_weights=True)
        return w


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ConvolutionalBody1d(ConvolutionalBodyBase1d):
    def __init__(self, in_size: int, series_size: int, conv_layers: [List[Tuple[int, int, int]]],
                 drop_out: float, dense: bool):
        super(ConvolutionalBody1d, self).__init__(in_size, series_size, conv_layers, drop_out)
        self._dense = dense
        if self._dense:
            self.output_size += self._in_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Switch series and feature dim, Conv layers want the channel/feature dim as second, the series as third.
        y = x.transpose(1, 2)
        # Do convolutions + Potential Dropout
        y = self.conv_layers(y)
        # Flatten out to Rank-2 tensor
        y = torch.flatten(y, start_dim=1)
        if self._dense:
            # Add full last original input entry
            y = torch.cat([y, x[:, -1, :]], dim=1)
        return y


class TransformerBody(_Layer):
    def __init__(self, in_size: int, series_size: int, positional_size: int, heads: int, feedforward_size: int,
                 drop_out: float, pos_logic: str):
        super(TransformerBody, self).__init__()
        if pos_logic == 'encoding':
            self.pos = PositionalEncoding(in_size, series_size, positional_size)
        else:
            self.pos = PositionalEmbedding(in_size, series_size, positional_size)
        self.trans = nn.TransformerEncoderLayer(self.pos.output_size, heads, feedforward_size, drop_out, 'relu')
        self._output_size = self.pos.output_size * series_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(x)
        x = self.trans(x)
        x = torch.flatten(x, start_dim=1)
        return x

    @property
    def output_size(self) -> int:
        return self._output_size
