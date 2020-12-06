"""
Imports for Pytorch data
(c) 2020 d373c7
"""
import torch
import torch.nn as nn
import torch.nn.functional as nnf
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss as TorchLoss
from ..features import TensorDefinition, FeatureCategorical
from typing import Type, Any


class _LossBase:
    """Loss function object fed to the training """
    def __init__(self, loss_fn: Type[TorchLoss], reduction: str):
        self._training_loss = loss_fn(reduction=reduction)
        self._score_loss = loss_fn(reduction='none')
        self._aggregator = torch.sum if reduction == 'sum' else torch.mean

    def __call__(self, *args, **kwargs) -> torch.Tensor:
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

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pr = torch.squeeze(args[0])
        lb = torch.squeeze(args[1][0])
        loss = self.train_loss(pr, lb)
        return loss

    def score(self, *args, **kwargs) -> torch.Tensor:
        pr = torch.squeeze(args[0])
        lb = torch.squeeze(args[1][0])
        loss = self.score_loss(pr, lb)
        return self.score_aggregator(loss, dim=1)


class MultiLabelBCELoss(_LossBase):
    def __init__(self, tensor_def: TensorDefinition, reduction='mean'):
        _LossBase.__init__(self, nn.BCELoss, reduction)
        cat_features = [f for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)]
        self._sizes = [len(f)+1 for f in cat_features]

    def __call__(self, *args, **kwargs):
        pr = torch.squeeze(args[0])
        lb = [nnf.one_hot(args[1][0][:, i], num_classes=s).type(pr.type()) for i, s in enumerate(self._sizes)]
        loss = self.train_loss(pr, torch.cat(lb, dim=1))
        return loss

    def score(self, *args, **kwargs) -> torch.Tensor:
        pr = torch.squeeze(args[0])
        lb = [nnf.one_hot(args[1][0][:, i], num_classes=s).type(pr.type()) for i, s in enumerate(self._sizes)]
        lb = torch.cat(lb, dim=1)
        score = self.score_loss(pr, lb)
        score = self.score_aggregator(score, dim=1)
        return score


class MultiLabelNLLLoss(_LossBase):
    """ MultiLabelNLLLoss Negative Log Likely-hood Loss"""
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
        score = self.score_aggregator(score, dim=list(range(1, len(pr.shape)-1)))
        return score


class BinaryVAELoss(_LossBase):
    """ Binary VAE Loss

    Args:
        reduction : The reduction to use. One of 'mean', 'sum'. Do not use 'none'.
        kl_weight : A weight to apply to the kl divergence. The kl_divergence will be multiplied by the weight
        before adding to the BCE Loss. Default is 1. That means the full kl_divergence is added. The kl_divergence
        can be given a lower importance with values < 0.
    """
    def __init__(self, reduction='mean', kl_weight=1.0):
        _LossBase.__init__(self, nn.BCELoss, reduction)
        self._kl_weight = kl_weight

    def __call__(self, *args, **kwargs):
        # With a VAE the first argument is the latent dim, the second is the mu, the third the sigma.
        pr = torch.squeeze(args[0][0])
        mu = args[0][1]
        s = args[0][2]
        lb = args[1][0]
        recon_loss = self.train_loss(pr, lb)
        kl_divergence = -0.5 * torch.sum(1 + s - mu.pow(2) - s.exp())
        return recon_loss + (self._kl_weight * kl_divergence)

    def score(self, *args, **kwargs) -> torch.Tensor:
        pr = torch.squeeze(args[0][0])
        lb = args[1][0]
        mu = args[0][1]
        s = args[0][2]
        # BCE Loss
        loss = self.score_loss(pr, lb)
        recon_loss = self.score_aggregator(loss, dim=1)
        # KL Divergence. Do not run over the batch dimension
        kl_divergence = -0.5 * torch.sum(1 + s - mu.pow(2) - s.exp(), tuple(range(1, len(mu.shape))))
        return recon_loss + (self._kl_weight * kl_divergence)
