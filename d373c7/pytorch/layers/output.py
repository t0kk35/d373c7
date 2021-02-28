"""
Module for layers that are typically used as output
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from math import sqrt
from .common import Layer
from ...features.common import FeatureCategorical
from ...features.tensor import TensorDefinition
from collections import OrderedDict
from typing import List, Tuple


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
        for i, (o_size, dropout) in enumerate(definition[:-1]):
            ls.update({f'tail_lin_{i+1:02d}': nn.Linear(prev_size, o_size)})
            ls.update({f'tail_act_{i+1:02d}': nn.ReLU()})
            if dropout != 0:
                ls.update({f'tail_dropout_{i+1:02d}': nn.Dropout(dropout)})
            prev_size = o_size
        # Add batch norm is requested and there is more than 1 layer.
        if add_bn and len(definition) > 1:
            ls.update({f'tail_batch_norm': nn.BatchNorm1d(prev_size)})
        # Add Last Binary layer
        ls.update({f'tail_binary': nn.Linear(prev_size, definition[-1][0])})
        ls.update({f'tail_bin_act': nn.Sigmoid()})
        self.layers = nn.Sequential(ls)

    @property
    def output_size(self) -> int:
        return 1

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # If we received multiple Tensors, there were multiple streams, which we will concatenate before applying linear
        if isinstance(x, List) and len(x) > 1:
            x = torch.cat(x, dim=1)
        else:
            x = x[0]
        return self.layers(x)


# TODO, Think I'd like to make this a single class that uses the TensorDef to see if we have a 2 or 3 D tensor.
# TODO 2. We will want to make this consistent with the Tail binary class and feed in a list of tensors.
class _CategoricalLogSoftmax(Layer):
    def __init__(self, tensor_def: TensorDefinition, input_rank: int, es_expr: str, use_mask=False):
        super(_CategoricalLogSoftmax, self).__init__()
        self.use_mask = use_mask
        self.input_rank = input_rank
        self.ein_sum_expression = es_expr
        i_features = [f for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)]
        self.sizes = [len(f) + 1 for f in i_features]
        self.hidden_dim = max(self.sizes)
        self.class_dim = len(self.sizes)
        self.lsm = nn.LogSoftmax(dim=self.input_rank-1)

    @property
    def output_size(self) -> int:
        return self.class_dim

    @staticmethod
    def calculate_fan_in_and_fan_out(x: torch.Tensor):
        dimensions = x.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        num_input_maps = x.size()[1]
        num_output_maps = x.size()[0]
        receptive_field_size = 1
        if x.dim() > 2:
            receptive_field_size = x[0][0].numel()
        fan_in = num_input_maps * receptive_field_size
        fan_out = num_output_maps * receptive_field_size
        return fan_in, fan_out

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.f_weight, a=sqrt(5))
        fan_in, _ = self.calculate_fan_in_and_fan_out(self.f_weight)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.f_bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        # If we received multiple Tensors, there were multiple streams, which we will concatenate before applying linear
        if isinstance(x, List) and len(x) > 1:
            x = torch.cat(x, dim=1)
        elif isinstance(x, List):
            x = x[0]
        x = torch.einsum(self.ein_sum_expression, x, self.f_weight)
        x = x + self.f_bias
        if self.mask is not None:
            x = x * self.mask
        x = self.lsm(x)
        return x


class CategoricalLogSoftmax1d(_CategoricalLogSoftmax):
    def __init__(self, tensor_def: TensorDefinition, input_size: int, use_mask=False):
        super(CategoricalLogSoftmax1d, self).__init__(tensor_def, 2, 'bi,ilc->blc', use_mask)
        self.f_weight = nn.parameter.Parameter(torch.zeros(input_size, self.hidden_dim, self.class_dim))
        self.f_bias = nn.parameter.Parameter(torch.zeros(self.hidden_dim, self.class_dim))
        mask = torch.zeros(self.hidden_dim, self.class_dim)
        for i, s in enumerate(self.sizes):
            mask[:s+1, i] = 1.0
        self.register_buffer('mask', mask if use_mask else None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f'max_dim={self.hidden_dim}, classes={self.class_dim}, use_mask={self.use_mask}'


class CategoricalLogSoftmax2d(_CategoricalLogSoftmax):
    def __init__(self, tensor_def: TensorDefinition, input_size: int, use_mask=False):
        super(CategoricalLogSoftmax2d, self).__init__(tensor_def, 3, 'bsi,ilc->bslc', use_mask)
        self.f_weight = nn.parameter.Parameter(torch.zeros(input_size, self.hidden_dim, self.class_dim))
        self.f_bias = nn.parameter.Parameter(torch.zeros(self.hidden_dim, self.class_dim))
        mask = torch.zeros(self.hidden_dim, self.class_dim)
        for i, s in enumerate(self.sizes):
            mask[:s+1, i] = 1.0
        self.register_buffer('mask', mask if use_mask else None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f'max_dim={self.hidden_dim}, classes={self.class_dim}, use_mask={self.use_mask}'
