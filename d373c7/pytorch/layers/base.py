"""
Module for common layers
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from .common import _Layer, PyTorchLayerException
from ...features.tensor import TensorDefinition, TensorDefinitionMulti
from ...features.common import LEARNING_CATEGORY_BINARY, FeatureCategorical
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
    """Layer that creates a set of torch embedding layers. One for each 'FeatureIndex' more specifically.
    The embeddings will be concatenated in the forward operation. So this will take a tensor of torch.long, run each
    through a torch embedding layer, concatenate the output, apply dropout and return.

    :argument tensor_def: A Tensor Definition describing the input. Each FeatureIndex in this definition will be turned
    into an embedding layer
    :argument dropout: A float number that determines the dropout amount to apply. The dropout will be applied to the
    concatenated output layer
    :argument min_dims: The minimum dimension of an embedding.
    :argument max_dims: The maximum dimension of an embedding.
    """
    def _val_feature_in_embedding(self, feature: FeatureCategorical):
        if not isinstance(feature, FeatureCategorical):
            raise PyTorchLayerException(
                f'Feature <{feature.name}> is not of type {FeatureCategorical.__class__}. Embedding only work with ' +
                f'Index Features'
            )
        if feature not in self._i_features:
            raise PyTorchLayerException(
                f'Feature <{feature.name}> is not known to this embedding layer. Please check the model was created ' +
                f'with a Tensor Definition than contains this feature'
            )

    def __init__(self, tensor_def: TensorDefinition, dropout: float, min_dims: int, max_dims: int):
        super(Embedding, self).__init__()
        self._i_features = [f for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)]
        emb_dim = [(len(f)+1, min(max(int(len(f)/2), min_dims), max_dims)) for f in self._i_features]
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

    def embedding_weight(self, feature: FeatureCategorical) -> torch.Tensor:
        self._val_feature_in_embedding(feature)
        i = self._i_features.index(feature)
        w = self.embeddings[i].weight
        return w


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

    def __init__(self, tensor_def: TensorDefinition, emb_dropout: float, emb_min_dim: int, emb_max_dim: int):
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
            self.embedding = Embedding(tensor_def, emb_dropout, emb_min_dim, emb_max_dim)
            self._output_size += self.embedding.output_size
        else:
            self.embedding = None
        self._indexes = list(self._used_learning_categories.values())

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def x_indexes(self) -> List[int]:
        return self._indexes

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
        x = torch.cat(cat_list, dim=self._rank-1)
        return x

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        x = [ds[x] for x in self._indexes]
        return x

    def embedding_weight(self, feature: FeatureCategorical) -> torch.Tensor:
        return self.embedding.embedding_weight(feature)


class TensorDefinitionHeadMulti(_Layer):
    def __init__(self, tensor_def: TensorDefinitionMulti, emb_dropout: float, emb_min_dim: int, emb_max_dim: int):
        super(TensorDefinitionHeadMulti, self).__init__()
        self.tensor_definition = tensor_def
        self.heads = nn.ModuleList(
            [TensorDefinitionHead(td, emb_dropout, emb_min_dim, emb_max_dim) for td in tensor_def.tensor_definitions]
        )
        self._output_size = sum([h.output_size for h in self.heads])
        lcs = [0] + [len(td.learning_categories) for td in tensor_def.tensor_definitions]
        self._indexes = [[ind + lcs[i] for ind in h.x_indexes] for i, h in enumerate(self.heads)]

    @property
    def x_indexes(self) -> List[int]:
        return self._x_indexes

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x) -> List[torch.Tensor]:
        x = [h([x[ind] for ind in self._indexes[i]]) for i, h in enumerate(self.heads)]
        return x

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        x = [ds[i] for ind in self._indexes for i in ind]
        return x

    def embedding_weight(self, feature: FeatureCategorical) -> torch.Tensor:
        h = [i for i, td in enumerate(self.tensor_definition.tensor_definitions) if feature in td.features]
        return self.heads[h[0]].embedding.embedding_weight(feature)

    def extra_repr(self) -> str:
        return f'Embedded TDs={[td.name for td in self.tensor_definition.tensor_definitions]}'
