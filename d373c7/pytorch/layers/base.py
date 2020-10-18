"""
Module for common layers
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from .common import _Layer, PyTorchLayerException
from ...features.tensor import TensorDefinition
from ...features.base import FeatureIndex
from typing import List, Tuple


class LinDropAct(_Layer):
    """Layer that runs a sequence of Linear/Drop-out/Activation operations. Attention: The last layer in the sequence
    will NOT have a activation layer. The definition will determine how many layers there are. For instance
    definition = [(128,0.0),(64,0.0),(32.0.1)  will create 3 Linear Layers of 128, 64 and 32 features respectively.
    A dropout of 0.1 will be applied to the last layer.

    :argument input_size: The size of the first layer. This must be the same as the output size of the previous layer
    :argument definition: A List of Tuples. Each entry in the list will be turned into a layer. The Tuples must be
        of type [int, float]. The int is the number of features in that specific layer, the float is the dropout rate at
        that layer. If the dropout is 0.0 no dropout will be performed.
    """
    def __init__(self, input_size: int, definition: List[Tuple[int, float]]):
        super(LinDropAct, self).__init__()
        ls = []
        prev_size = input_size
        for o_size, dropout in definition:
            ls.append(nn.Linear(prev_size, o_size))
            if dropout != 0:
                ls.append(nn.Dropout(dropout))
            ls.append(nn.ReLU(inplace=True))
            prev_size = o_size
        # Remove last activation layer
        ls.pop()
        self.layers = nn.Sequential(*ls)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Embedding(_Layer):
    """Layer that creates a set of torch embedding layers. One for each 'FeatureIndex' more specifically. The embeddings
    will be concatenated in the forward operation. So this will take a tensor of torch.long, run each through a torch
    embedding layer, concatenate the output, apply dropout and return.

    :argument tensor_def: A Tensor Definition describing the input. Each FeatureIndex in this definition will be turned
        into an embedding layer
    :argument dropout: A float number that determines the dropout amount to apply. The dropout will be applied to the
        concatenated output layer
    """
    def __init__(self, tensor_def: TensorDefinition, dropout: float, min_dims=4, max_dims=50):
        super(Embedding, self).__init__()
        i_feature = [f for f in tensor_def.categorical_features() if isinstance(f, FeatureIndex)]
        emb_dim = [(len(f)+1, min(max(int(len(f)/2), min_dims), max_dims)) for f in i_feature]
        self._out_size = sum([y for _, y in emb_dim])
        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dim])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        s = x.shape
        if len(s) == 2:
            x = torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        elif len(s) == 3:
            x = torch.cat([emb(x[:, :, i]) for i, emb in enumerate(self.embeddings)], dim=2)
        else:
            raise PyTorchLayerException(f'Don\'t know how to handle embedding with input tensor of rank {len(s)}')
        x = self.dropout(x)
        return x
