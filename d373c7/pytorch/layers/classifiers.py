"""
Module for Layers that can be used as body in the _Module objects.
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from .common import Layer
from .base import PositionalEncoding, PositionalEmbedding, ConvolutionalBodyBase1d
from collections import OrderedDict
from typing import List, Union, Tuple


# New Generator Classes
class TailBinary(Layer):
    """Layer that runs a sequence of Linear/Drop-out/Activation operations. The definition will determine how many
    layers there are.
    For instance definition = [(128,0.0),(64,0.0),(32.0.1) will create 3 Linear Layers of 128, 64 and 32 features
    respectively. A dropout of 0.1 will be applied to the last layer.
    After the linear layers it runs a final linear layer of output size to and a sigmoid, to come to a binary output.

    :argument input_size: The size of the first layer. This must be the same as the output size of the previous layer
    :argument definition: A List of Tuples. Each entry in the list will be turned into a layer. The Tuples must be
    of type [int, float]. The int is the number of features in that specific layer, the float is the dropout rate at
    that layer. If the dropout is 0.0 no dropout will be performed.
    """
    def __init__(self, input_size: int, definition: List[Tuple[int, float]], add_bn=True):
        super(TailBinary, self).__init__()
        ls = OrderedDict()
        prev_size = input_size
        for i, (o_size, dropout) in enumerate(definition):
            ls.update({f'tail_lin_{i+1:02d}': nn.Linear(prev_size, o_size)})
            ls.update({f'tail_act_{i+1:02d}': nn.ReLU()})
            if dropout != 0:
                ls.update({f'tail_dropout_{i+1:02d}': nn.Dropout(dropout)})
            prev_size = o_size
        # Add Last Binary layer
        if add_bn:
            ls.update({f'tail_batch_norm': nn.BatchNorm1d(prev_size)})
        ls.update({f'tail_binary': nn.Linear(prev_size, 1)})
        ls.update({f'tail_bin_act': nn.Sigmoid()})
        self.layers = nn.Sequential(ls)

    @property
    def output_size(self) -> int:
        return 1

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # If we received multiple Tensors, there were multiple streams, which we will concatenate before applying linear
        if len(x) > 1:
            x = torch.cat(x, dim=1)
        else:
            x = x[0]
        return self.layers(x)


class _RecurrentBody(Layer):
    def __init__(self, in_size: int, recurrent_features: int, recurrent_layers: int, dense: bool, batch_norm: bool):
        super(_RecurrentBody, self).__init__()
        self._out_size = recurrent_features + in_size if dense else recurrent_features
        self._dense = dense
        self.rnn = self.init_rnn(in_size, recurrent_features, recurrent_layers)
        self.bn = nn.BatchNorm1d(self.output_size) if batch_norm else None

    def init_rnn(self, in_size: int, recurrent_features: int, recurrent_layers: int):
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
        return self._out_size


class LSTMBody(_RecurrentBody):
    def __init__(self, in_size: int, recurrent_features: int, recurrent_layers: int, dense: bool, batch_norm: bool):
        super(LSTMBody, self).__init__(in_size, recurrent_features, recurrent_layers, dense, batch_norm)

    def init_rnn(self, in_size: int, recurrent_features: int, recurrent_layers: int):
        rnn = nn.LSTM(in_size, recurrent_features, recurrent_layers, batch_first=True)
        return rnn


class GRUBody(_RecurrentBody):
    def __init__(self, in_size, recurrent_features: int, recurrent_layers: int, dense: bool, batch_norm: bool):
        super(GRUBody, self).__init__(in_size, recurrent_features, recurrent_layers, dense, batch_norm)

    def init_rnn(self, in_size: int, recurrent_features: int, recurrent_layers: int):
        rnn = nn.GRU(in_size, recurrent_features, recurrent_layers, batch_first=True)
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

# New Generator Classes


class BodyMulti(Layer):
    def __init__(self, head: Layer, layers: List[Union[Layer, None]]):
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


class BodySequential(Layer):
    def __init__(self, in_size: int, layers: List[Layer]):
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
    def __init__(self, in_size, attention: Layer, layers: List[Layer]):
        super(BodyWithAttention, self).__init__(in_size, [attention]+layers)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        _, w = self.layers[0](x, return_weights=True)
        return w


class TransformerBody(Layer):
    def __init__(self, in_size: int, series_size: int, positional_size: int,  positional_logic: str,
                 heads: int, feedforward_size: int, drop_out: float):
        super(TransformerBody, self).__init__()
        if positional_logic == 'encoding':
            self.pos = PositionalEncoding(in_size, series_size, positional_size)
        else:
            self.pos = PositionalEmbedding(in_size, series_size, positional_size)
        self.trans = nn.TransformerEncoderLayer(self.pos.output_size, heads, feedforward_size, drop_out, 'relu')
        self._output_size = self.pos.output_size * series_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(x)
        # Transformers want (S, B, F) and the input is (B, S, F).
        x = self.trans(x.transpose(0, 1))
        x = torch.flatten(x.transpose(1, 0), start_dim=1)
        return x

    @property
    def output_size(self) -> int:
        return self._output_size
