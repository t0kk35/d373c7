"""
Module for layers that are used in variational auto-encoders
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
from .common import _Layer


class VAELinearToLatent(_Layer):
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
        # if self.training:
        #     s = s.mul(0.5).exp_()
        #     eps = torch.empty_like(s).normal_()
        #     return eps.mul(s).add_(mu)
        # else:
        #     return mu

    def extra_repr(self) -> str:
        return f're-parameterization layer'
