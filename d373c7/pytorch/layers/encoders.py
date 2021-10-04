"""
Module for layers that are used in auto-encoders
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from .common import Layer
from .base import ConvolutionalBodyBase1d, ConvolutionalBodyBaseTranspose1d
from collections import OrderedDict
from typing import List, Tuple


# TODO Update Doc. Change Argument and review text
class LinearEncoder(Layer):
    """Layer that runs a sequence of Linear/Drop-out/Activation operations. The definition will determine how many
    layers there are.
    For instance definition = [(128,0.0),(64,0.0),(32.0.1) will create 3 Linear Layers of 128, 64 and 32 features
    respectively. A dropout of 0.1 will be applied to the last layer.

    The last layer does not get activated as this is an encoder layer

    :argument input_size: The size of the first layer. This must be the same as the output size of the previous layer
    :argument definition: A List of Tuples. Each entry in the list will be turned into a layer. The Tuples must be
    of type [int, float]. The int is the number of features in that specific layer, the float is the dropout rate at
    that layer. If the dropout is 0.0 no dropout will be performed.
    """
    def __init__(self, input_size: int, definition: List[Tuple[int, float]], latent_features: int, add_bn=True):
        super(LinearEncoder, self).__init__()
        ls = OrderedDict()
        prev_size = input_size
        for i in range(len(definition)):
            (o_size, dropout) = definition[i]
            ls.update({f'enc_lin_{i+1:02d}': nn.Linear(prev_size, o_size)})
            ls.update({f'enc_act_{i+1:02d}': nn.ReLU()})
            if dropout != 0:
                ls.update({f'enc_dropout_{i+1:02d}': nn.Dropout(dropout)})
            prev_size = o_size
        # Add Last Binary layer. The Latent Layer
        if add_bn:
            ls.update({f'enc_batch_norm': nn.BatchNorm1d(prev_size)})
        self._layer_definition = definition
        self._output_size = latent_features
        ls.update({f'enc_latent': nn.Linear(prev_size, self._output_size)})
        self.layers = nn.Sequential(ls)

    @property
    def layer_definition(self) -> List[Tuple[int, float]]:
        return self._layer_definition

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LinearDecoder(Layer):
    """Layer that runs a sequence of Linear/Drop-out/Activation operations. The definition will determine how many
    layers there are.
    For instance definition = [(128,0.0),(64,0.0),(32.0.1) will create 3 Linear Layers of 128, 64 and 32 features
    respectively. A dropout of 0.1 will be applied to the last layer.

    The last layer does not get activated as this is an encoder layer

    :argument input_size: The size of the first layer. This must be the same as the output size of the previous layer
    :argument definition: A List of Tuples. Each entry in the list will be turned into a layer. The Tuples must be
    of type [int, float]. The int is the number of features in that specific layer, the float is the dropout rate at
    that layer. If the dropout is 0.0 no dropout will be performed.
    """
    def __init__(self, input_size: int, definition: List[Tuple[int, float]], add_bn=True):
        super(LinearDecoder, self).__init__()
        ls = OrderedDict()
        prev_size = input_size
        for i in reversed(range(len(definition))):
            (o_size, dropout) = definition[i]
            ls.update({f'dec_lin_{i+1:02d}': nn.Linear(prev_size, o_size)})
            ls.update({f'dec_act_{i+1:02d}': nn.ReLU()})
            if dropout != 0:
                ls.update({f'dec_dropout_{i+1:02d}': nn.Dropout(dropout)})
            prev_size = o_size
        if add_bn:
            ls.update({f'dec_batch_norm': nn.BatchNorm1d(prev_size)})
        # We do not add the final output, that is done later.
        self._output_size = prev_size
        self.layers = nn.Sequential(ls)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LinearVAEEncoder(Layer):
    """Layer that runs a sequence of Linear/Drop-out/Activation operations. The definition will determine how many
    layers there are.
    For instance definition = [(128,0.0),(64,0.0),(32.0.1) will create 3 Linear Layers of 128, 64 and 32 features
    respectively. A dropout of 0.1 will be applied to the last layer.

    The last layer does not get activated as this is an encoder layer

    :argument input_size: The size of the first layer. This must be the same as the output size of the previous layer
    :argument definition: A List of Tuples. Each entry in the list will be turned into a layer. The Tuples must be
    of type [int, float]. The int is the number of features in that specific layer, the float is the dropout rate at
    that layer. If the dropout is 0.0 no dropout will be performed.
    """
    def __init__(self, input_size: int, definition: List[Tuple[int, float]], latent_features: int, add_bn=True):
        super(LinearVAEEncoder, self).__init__()
        ls = OrderedDict()
        prev_size = input_size
        for i in range(len(definition)):
            (o_size, dropout) = definition[i]
            ls.update({f'enc_lin_{i+1:02d}': nn.Linear(prev_size, o_size)})
            ls.update({f'enc_act_{i+1:02d}': nn.ReLU()})
            if dropout != 0:
                ls.update({f'enc_dropout_{i+1:02d}': nn.Dropout(dropout)})
            prev_size = o_size
        if add_bn:
            ls.update({f'enc_batch_norm': nn.BatchNorm1d(prev_size)})
        self.layers = nn.Sequential(ls)
        # Add Last Layer. This is specific to VAE. We'll have a layer for the mu and a layer for the sigma's.
        self._output_size = latent_features
        self.mu = nn.Linear(prev_size,  self._output_size)
        self.s = nn.Linear(prev_size,  self._output_size)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layers(x)
        mu = self.mu(x)
        s = self.s(x)
        return mu, s


class LinearVAEDecoder(Layer):
    """Layer that runs a sequence of Linear/Drop-out/Activation operations. The definition will determine how many
    layers there are.
    For instance definition = [(128,0.0),(64,0.0),(32.0.1) will create 3 Linear Layers of 128, 64 and 32 features
    respectively. A dropout of 0.1 will be applied to the last layer.

    The last layer does not get activated as this is an encoder layer

    :argument input_size: The size of the first layer. This must be the same as the output size of the previous layer
    :argument definition: A List of Tuples. Each entry in the list will be turned into a layer. The Tuples must be
    of type [int, float]. The int is the number of features in that specific layer, the float is the dropout rate at
    that layer. If the dropout is 0.0 no dropout will be performed.
    """
    def __init__(self, input_size: int, definition: List[Tuple[int, float]], add_bn=True):
        super(LinearVAEDecoder, self).__init__()
        ls = OrderedDict()
        prev_size = input_size
        for i in reversed(range(len(definition))):
            (o_size, dropout) = definition[i]
            ls.update({f'dec_lin_{i+1:02d}': nn.Linear(prev_size, o_size)})
            ls.update({f'dec_act_{i+1:02d}': nn.ReLU()})
            if dropout != 0:
                ls.update({f'dec_dropout_{i+1:02d}': nn.Dropout(dropout)})
            prev_size = o_size
        if add_bn:
            ls.update({f'dec_batch_norm': nn.BatchNorm1d(prev_size)})
        # We do not add the final output, that is done later.
        self._output_size = prev_size
        self.layers = nn.Sequential(ls)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The input is not just a single Tensor, but a Tuple. That is what the LinearVAEEncoder has set.
        mu, s = x
        # first re-parameterize the latent mu and s.
        s = torch.exp(0.5*s)
        eps = torch.randn_like(s)
        x = mu + eps * s
        # Now run through the linear layers.
        x = self.layers(x)
        # Attention! We need to also return the mu and s.
        return x, mu, s


class LinearVAEOut(Layer):
    """Specialised output layer for VAE. This needs to be able to handle a tuple if x, mu, s as set by the
    LinearVAEDecoder. The mu and s are forwarded to the loss as-in, the x goes through a final linear layer.

    :argument input_size: Int value. The size of the previous layer
    :argument output_size: Int value. The target output size of the linear linear layer
    """
    def __init__(self, input_size: int, output_size: int):
        super(LinearVAEOut, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self._output_size = output_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor])\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The input is not just a single Tensor, but a Tuple. That is what the LinearVAEDecoder has set.
        x, mu, s = x
        x = self.linear(x)
        x = self.sigmoid(x)
        # Attention! We need to also return the mu and s. This will go to the loss function
        return x, mu, s


class ConvolutionalEncoder(ConvolutionalBodyBase1d):
    def __init__(self, in_size: int, series_size: int, conv_layers: List[Tuple[int, int]], drop_out: float,
                 batch_norm_interval: int):
        """Encoder layer which will encode using standard Convolutional Layers.
        """
        # Default Stride to 1
        layers = [(ch, fl, 1) for ch, fl in conv_layers]
        super(ConvolutionalEncoder, self).__init__(
            in_size, series_size, layers, drop_out, batch_norm_interval=batch_norm_interval, activate_last=False
        )
        self.output_size = layers[-1][0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Switch series and feature dim, Conv layers want the channel/feature dim as second, the series as third.
        # Shape needs to be (B, F, S)
        x = x.transpose(1, 2)
        # Do convolutions + Potential Dropout
        x = self.conv_layers(x)
        return x


class ConvolutionalDecoder(ConvolutionalBodyBaseTranspose1d):
    def __init__(self, in_size: int, out_size: int, series_size: int, conv_layers: List[Tuple[int, int]],
                 drop_out: float, batch_norm_interval: int):
        """Decoder layer which will encode using standard Convolutional Layers.
        """
        # Shift the kernel_sizes by one, add a layer w/input_size and reverse the layer list
        layers = [(conv_layers[i][0], conv_layers[i+1][1]) for i in range(len(conv_layers)-1)]
        layers = [(out_size, conv_layers[0][1])] + layers
        layers = layers[::-1]
        # Default Stride to 1
        layers = [(ch, fl, 1) for ch, fl in layers]
        super(ConvolutionalDecoder, self).__init__(
            in_size, series_size, layers, drop_out, batch_norm_interval=batch_norm_interval, activate_last=False
        )
        self.output_size = layers[-1][0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        # Switch back the series and feature dim. Shape becomes (B, S, F)
        x = x.transpose(1, 2)
        return x
