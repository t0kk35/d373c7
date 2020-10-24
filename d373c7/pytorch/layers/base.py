"""
Module for common layers
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from .common import _Layer, PyTorchLayerException
from ...features.tensor import TensorDefinition
from ...features.base import FeatureIndex
from ...features.common import LEARNING_CATEGORY_BINARY
from ...features import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_CATEGORICAL
from typing import List, Tuple


class LinDropAct(_Layer):
    """Layer that runs a sequence of Linear/Drop-out/Activation operations. The definition will determine how many
    layers there are.
    For instance definition = [(128,0.0),(64,0.0),(32.0.1) will create 3 Linear Layers of 128, 64 and 32 features
    respectively. A dropout of 0.1 will be applied to the last layer.

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
        self._out_size = prev_size
        self.layers = nn.Sequential(*ls)

    @property
    def output_size(self) -> int:
        return self._out_size

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

    @property
    def output_size(self) -> int:
        return self._out_size

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


class BinaryOutput(_Layer):
    """Layer which can be used as output layer for binary classifications. It consists of a linear layer mapping from
    'input_size' to 'output_size' followed by a Sigmoid Layer

    :argument in_size: The input size of the layer
    :argument out_size: The output size of the layer
    """
    def __init__(self, in_size: int, out_size: int):
        super(BinaryOutput, self).__init__()
        self._out_size = out_size
        self.out_block = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.Sigmoid()
        )

    @property
    def output_size(self) -> int:
        return self._out_size

    def forward(self, x):
        x = self.out_block(x)
        return x


class SingleClassBinaryOutput(BinaryOutput):
    """Specialisation of the BinaryOutput Layer. It consists of a linear layer mapping from
    'input_size' to size 1 followed by a Sigmoid Layer

    :argument in_size: The input size of the layer
    """

    def __init__(self, in_size: int):
        super(SingleClassBinaryOutput, self).__init__(in_size, 1)

    @property
    def output_size(self) -> int:
        return 1


class TensorDefinitionHead(_Layer):
    @staticmethod
    def _val_has_bin_or_con_or_cat_features(tensor_def: TensorDefinition):
        if not (LEARNING_CATEGORY_BINARY in tensor_def.learning_categories
                or LEARNING_CATEGORY_CONTINUOUS in tensor_def.learning_categories
                or LEARNING_CATEGORY_CATEGORICAL in tensor_def.learning_categories):
            raise PyTorchLayerException(
                f'_StandardHead needs features of Learning category "Binary" or "Continuous" or "Categorical. '
                f'Tensor definition <{tensor_def.name} has none of these.'
            )

    def __init__(self, tensor_def: TensorDefinition):
        TensorDefinitionHead._val_has_bin_or_con_or_cat_features(tensor_def)
        super(TensorDefinitionHead, self).__init__()
        self._rank = tensor_def.rank
        lcs = (LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_CATEGORICAL)
        self._used_learning_categories = {
            lc: tensor_def.learning_categories.index(lc)
            for lc in lcs if lc in tensor_def.learning_categories
        }
        lcs = (LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CONTINUOUS)
        self._output_size = sum(
            len(tensor_def.filter_features(lc, True)) for lc in lcs if lc in self._used_learning_categories.keys()
        )
        if LEARNING_CATEGORY_CATEGORICAL in self._used_learning_categories.keys():
            self.embedding = Embedding(tensor_def, 0.1)
            self._output_size += self.embedding.output_size
        else:
            self.embedding = None

    @property
    def x_indexes(self) -> List[int]:
        return list(self._used_learning_categories.values())

    @property
    def output_size(self) -> int:
        return self._output_size

    def extra_repr(self) -> str:
        return f'lcs={[e.name for e in self._used_learning_categories.keys()]}'

    def forward(self, x):
        # Concatenate the binary and continuous categories and run the categorical through an embedding.
        cat_list = []
        for lc in (LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_CATEGORICAL):
            if lc in self._used_learning_categories.keys():
                i = self._used_learning_categories[lc]
                if lc == LEARNING_CATEGORY_CATEGORICAL:
                    e = self.embedding(x[i])
                    cat_list.append(e)
                else:
                    if len(x[i].shape) < self._rank:
                        p = torch.unsqueeze(x[i], dim=self._rank-1)
                    else:
                        p = x[i]
                    cat_list.append(p)
        x = torch.cat(cat_list, dim=1)
        return x
