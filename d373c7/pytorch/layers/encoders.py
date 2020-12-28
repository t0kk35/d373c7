"""
Module for layers that are used in auto-encoders
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from .common import _Layer
from .base import ConvolutionalBodyBase1d, ConvolutionalBodyBaseTranspose1d
from typing import List, Tuple


class VAELinearToLatent(_Layer):
    """Layer to encoder a linear layer to a VAE style latent dimension. Ie. having 2 vectors. One containing the
    averages, the other containing the standard deviations.

    Args:
        input_size: Integer value with the size of the input.
        latent_size: Integer value specifying the size of the desired latent dimension
    """
    def __init__(self, input_size: int, latent_size: int):
        super(VAELinearToLatent, self).__init__()
        self._out_size = latent_size
        self.mu = nn.Linear(input_size, latent_size)
        self.s = nn.Linear(input_size, latent_size)

    def forward(self, x):
        mu = self.mu(x)
        s = self.s(x)
        return mu, s

    @property
    def output_size(self) -> int:
        return self._out_size

    def extra_repr(self) -> str:
        return f'output_size={self._out_size}'


class VAELatentToLinear(_Layer):
    def __int__(self):
        super(VAELatentToLinear, self).__init__()

    @staticmethod
    def forward(mu: torch.Tensor, s: torch.Tensor):
        s = torch.exp(0.5*s)
        eps = torch.randn_like(s)
        return mu + eps * s

    def extra_repr(self) -> str:
        return f're-parameterization layer'


class _RecurrentEncoder(_Layer):
    """Recurrent encoder. This layer will encode series and narrow down the number of features. It uses multiple layers
    instead of the faster 'recurrent_layers' because the number of features needs to shrink.

    Args:
        in_size: The size of the previous layer.
        recurrent_features: A list of integers, specifying how many layers and how many feature we want in each layer
        the size of the list will be the number of layer, the integers themselves the number of features per layer.
        reverse: Flag indicating whether or not to reverse the recurrent feature. Not reversing will encode (compress),
        reversing will decode (de-compress)
    """
    def __init__(self, in_size: int, recurrent_features: List[int], reverse: bool):
        super(_RecurrentEncoder, self).__init__()
        self._in_size = in_size
        self._recurrent_features = recurrent_features
        layers = []
        prev_size = in_size
        if reverse:
            features = recurrent_features[::-1]
        else:
            features = recurrent_features
        for rf in features:
            layers.append(self.init_rnn(prev_size, rf))
            prev_size = rf
        self.rnn_layers = nn.ModuleList(layers)
        self.drop_out = nn.Dropout(p=0.2)

    def init_rnn(self, in_size: int, feature_size: int) -> nn.Module:
        raise NotImplemented('Should be implemented by children')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Iterate over the recurrent layers
        for rl in self.rnn_layers:
            x, _ = rl(x)
            x = self.drop_out(x)
        return x

    @property
    def output_size(self) -> int:
        return self._recurrent_features[-1]


class _LSTM(_RecurrentEncoder):
    def __init__(self, in_size: int, recurrent_features: List[int], reverse: bool):
        super(_LSTM, self).__init__(in_size, recurrent_features, reverse)

    def init_rnn(self, in_size: int, feature_size: int) -> nn.Module:
        return nn.LSTM(in_size, feature_size, 1, batch_first=True)


class _GRU(_RecurrentEncoder):
    def __init__(self, in_size: int, recurrent_features: List[int], reverse: bool):
        super(_GRU, self).__init__(in_size, recurrent_features, reverse)

    def init_rnn(self, in_size: int, feature_size: int) -> nn.Module:
        return nn.GRU(in_size, feature_size, 1, batch_first=True)


class LSTMEncoder(_LSTM):
    def __init__(self, in_size: int, recurrent_features: List[int]):
        super(LSTMEncoder, self).__init__(in_size, recurrent_features, False)


class LSTMDecoder(_LSTM):
    def __init__(self, in_size: int, recurrent_features: List[int]):
        super(LSTMDecoder, self).__init__(in_size, recurrent_features, True)


class GRUEncoder(_GRU):
    def __init__(self, in_size: int, recurrent_features: List[int]):
        super(GRUEncoder, self).__init__(in_size, recurrent_features, False)


class GRUDecoder(_GRU):
    def __init__(self, in_size: int, recurrent_features: List[int]):
        super(GRUDecoder, self).__init__(in_size, recurrent_features, True)


class ConvolutionalEncoder(ConvolutionalBodyBase1d):
    def __init__(self, in_size: int, series_size: int, conv_layers: List[Tuple[int, int]], drop_out: float,
                 batch_norm_interval: int):
        # Default Stride to 1
        layers = [(ch, fl, 1) for ch, fl in conv_layers]
        super(ConvolutionalEncoder, self).__init__(
            in_size, series_size, layers, drop_out, batch_norm_interval=batch_norm_interval, activate_last=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Switch series and feature dim, Conv layers want the channel/feature dim as second, the series as third.
        x = x.transpose(1, 2)
        # Do convolutions + Potential Dropout
        x = self.conv_layers(x)
        return x


class ConvolutionalDecoder(ConvolutionalBodyBaseTranspose1d):
    def __init__(self, in_size: int, series_size: int, conv_layers: List[Tuple[int, int]], drop_out: float,
                 batch_norm_interval: int):
        # Default Stride to 1
        layers = [(ch, fl, 1) for ch, fl in conv_layers]
        super(ConvolutionalDecoder, self).__init__(
            in_size, series_size, layers, drop_out, batch_norm_interval=batch_norm_interval, activate_last=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        # Switch back the series and feature dim.
        x = x.transpose(1, 2)
        return x
