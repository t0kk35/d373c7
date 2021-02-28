"""
Module for layers that are used in Next Transaction Predictions
(c) 2021 d373c7
"""
import torch
import torch.nn as nn
from .common import Layer
from .base import ConvolutionalBodyBase1d
from typing import List, Tuple


class ConvolutionalNtpBody(ConvolutionalBodyBase1d):
    def __init__(self, in_size: int, series_size: int, conv_layers: List[Tuple[int, int]], drop_out: float,
                 batch_norm_interval: int):
        """NTp layer which will create a body using standard Convolutional Layers.
        """
        # Default Stride to 1
        layers = [(ch, fl, 1) for ch, fl in conv_layers]
        super(ConvolutionalNtpBody, self).__init__(
            in_size, series_size, layers, drop_out, batch_norm_interval=batch_norm_interval, activate_last=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Switch series and feature dim, Conv layers want the channel/feature dim as second, the series as third.
        # Shape needs to be (B, F, S)
        x = x.transpose(1, 2)
        # Do convolutions + Potential Dropout
        x = self.conv_layers(x)
        # Flatten so we can follow with a linear layer if needed.
        # This changes the tensor shape from (B, S, F) to (B, S*F)
        x = torch.flatten(x, start_dim=1)
        return x


class LSTMNtpBody(Layer):
    def __init__(self, in_size: int, recurrent_features: int, recurrent_layers: int):
        super(LSTMNtpBody, self).__init__()
        self.rnn = nn.LSTM(in_size, recurrent_features, recurrent_layers, batch_first=True)
        self._output_size = recurrent_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(x)
        # Take last output of the RNN.
        x = x[:, -1, :]
        return x

    @property
    def output_size(self) -> int:
        return self._output_size
