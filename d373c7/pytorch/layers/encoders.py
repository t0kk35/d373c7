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
