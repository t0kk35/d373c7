"""
Module for layers that are typically used as output
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from math import sqrt
from .common import _Layer, PyTorchLayerException
from ...features.base import FeatureIndex
from ...features.tensor import TensorDefinition


class _CategoricalLogSoftmax(_Layer):
    def __init__(self, tensor_def: TensorDefinition, input_rank: int, es_expr: str, dropout=0.0, use_mask=False):
        super(_CategoricalLogSoftmax, self).__init__()
        self.use_mask = use_mask
        self.dr = dropout
        self.input_rank = input_rank
        self.ein_sum_expression = es_expr
        self._i_features = [f for f in tensor_def.categorical_features() if isinstance(f, FeatureIndex)]
        self.sizes = [len(f) + 1 for f in self._i_features]
        self.hidden_dim = max(self.sizes)
        self.class_dim = len(self.sizes)
        self.do = nn.Dropout(0.0) if dropout != 0.0 else None
        self.lsm = nn.LogSoftmax(dim=self.input_rank)

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
        x = torch.einsum(self.ein_sum_expression, x, self.f_weight)
        x = x + self.f_bias
        if self.do is not None:
            x = self.do(x)
        if self.mask is not None:
            x = x * self.mask
        x = self.lsm(x)
        return x


class CategoricalLogSoftmax1d(_CategoricalLogSoftmax):
    def __init__(self, tensor_def: TensorDefinition, dropout=0.0, use_mask=False):
        super(CategoricalLogSoftmax1d, self).__init__(tensor_def, 2, 'bh,lc->blc', dropout, use_mask)
        self.f_weight = nn.parameter.Parameter(torch.zeros(self.hidden_dim, self.class_dim))
        self.f_bias = nn.parameter.Parameter(torch.zeros(self.hidden_dim, self.class_dim))
        mask = torch.zeros(self.hidden_dim, self.class_dim)
        for i, s in enumerate(self.sizes):
            mask[:s, i] = 1.0
        self.register_buffer('mask', mask if use_mask else None)
        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        x = _CategoricalLogSoftmax.forward(self, x)
        return x

    def extra_repr(self) -> str:
        return f'max_dim={self.hidden_dim}, classes={self.class_dim}, dropout={self.dr}, use_mask={self.use_mask}'
