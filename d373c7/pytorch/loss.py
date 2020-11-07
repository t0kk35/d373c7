"""
Imports for Pytorch data
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss as TorchLoss
from typing import Type, List, Any


class _LossBase:
    """Loss function object fed to the training """
    def __init__(self, loss_fn: Type[TorchLoss], reduction: str):
        self._training_loss = loss_fn(reduction=reduction)
        self._score_loss = torch.nn.BCELoss(reduction='none')
        self._aggregator = torch.sum if reduction == 'sum' else torch.mean

    def __call__(self, *args, **kwargs) -> List[torch.Tensor]:
        raise NotImplemented('Call not implemented for loss function')

    def score(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented('Call not implemented for loss function')

    @property
    def train_loss(self) -> TorchLoss:
        return self._training_loss

    @property
    def score_loss(self) -> TorchLoss:
        return self._score_loss

    @property
    def score_aggregator(self) -> Any:
        return self._aggregator


class SingleLabelBCELoss(_LossBase):
    def __init__(self, reduction='mean'):
        _LossBase.__init__(self, nn.BCELoss, reduction)

    def __call__(self, *args, **kwargs) -> List[torch.Tensor]:
        pr = torch.squeeze(args[0])
        lb = torch.squeeze(args[1][0])
        loss = self.train_loss(pr, lb)
        return loss

    def score(self, *args, **kwargs) -> torch.Tensor:
        pr = torch.squeeze(args[0])
        lb = torch.squeeze(args[1][0])
        loss = self.score_loss(pr, lb)
        return self.score_aggregator(loss, dim=1)


class NLLLoss(_LossBase):
    """ SingleLabel Negative Log Likely-hood Loss"""
    def __init__(self, reduction='mean'):
        _LossBase.__init__(self, nn.NLLLoss, reduction)

    def __call__(self, *args, **kwargs):
        pr = torch.squeeze(args[0])
        lb = args[1][0]
        loss = self.train_loss(pr, lb)
        return loss

    def score(self, *args, **kwargs) -> torch.Tensor:
        pr = torch.squeeze(args[0])
        lb = args[1][0]
        score = self.score_loss(pr, lb)
        score = self.score_aggregator(score, dim=list(range(1, len(pr.shape))))
        return score


class BinaryVAELoss(_LossBase):
    """ Binary VAE Loss

    Args:
        reduction : The reduction to use. One of 'mean', 'sum'. Do not use 'none'.
    """
    def __init__(self, reduction='mean'):
        _LossBase.__init__(self, nn.BCELoss, reduction)

    def __call__(self, *args, **kwargs):
        # With a VAE the first argument is the latent dim, the second is the mu, the third the sigma.
        pr = torch.squeeze(args[0][0])
        mu = args[0][1]
        s = args[0][2]
        lb = args[1][0]
        recon_loss = self.train_loss(pr, lb)
        kl_divergence = -0.5 * torch.sum(1 + s - mu.pow(2) - s.exp())
        return recon_loss + 1 * kl_divergence

    def score(self, *args, **kwargs) -> torch.Tensor:
        pr = torch.squeeze(args[0][0])
        lb = args[1][0]
        loss = self.score_loss(pr, lb)
        score = self.score_aggregator(loss, dim=1)
        return score
