"""
Imports for Pytorch data
(c) 2020 d373c7
"""
from .common import PyTorchTrainException
from ..features.tensor import TensorDefinition
import torch


class _Loss:
    """Base Loss function object off of which all loss functions will be created.
    """
    @staticmethod
    def _val_categorical_or_binary_features(tensor_def: TensorDefinition):
        if len(tensor_def.categorical_features()) == 0 and len(tensor_def.binary_features()) == 0:
            raise PyTorchTrainException(
                f'The tensor definition should contain categorical feature or binary feature or both. '
                f'Got {len(tensor_def.categorical_features())} categorical features '
                f'and {len(tensor_def.binary_features())} binary features.'
            )

    def __init__(self, tensor_def: TensorDefinition, reduction='mean'):
        self._val_categorical_or_binary_features(tensor_def)
        self._train_loss = None
        self._score_loss = None
        self._aggregator = torch.sum if reduction == 'sum' else torch.mean
        self._bin_features = tensor_def.binary_features()
        self._cat_features = tensor_def.categorical_features()
        self._bin_size = len(self._bin_features)
        self._cat_sizes = [len(f) + 1 for f in self._cat_features]
        self._cat_offset = 1 if len(self._bin_features) > 0 else 0

    def __call__(self, *args, **kwargs):
        # If there are no categorical features there must be binary features.
        pr = torch.squeeze(args[0])
        lb = [torch.squeeze(args[1][0])] if len(self._bin_features) > 0 else []
        lbc = [torch.nn.functional.one_hot(args[1][self._cat_offset][:, i], num_classes=s).type(pr.type())
               for i, s in enumerate(self._cat_sizes)]
        lb.extend(lbc)
        lb = torch.cat(lb, dim=1)
        return self._train_loss(pr, lb)

    def score(self, *args) -> torch.Tensor:
        pr = torch.squeeze(args[0])
        lb = [torch.squeeze(args[1][0])] if len(self._bin_features) > 0 else []
        lbc = [torch.nn.functional.one_hot(args[1][self._cat_offset][:, i], num_classes=s).type(pr.type())
               for i, s in enumerate(self._cat_sizes)]
        lb.extend(lbc)
        lb = torch.cat(lb, dim=1)
        score = self._score_loss(pr, lb)
        score = self._aggregator(score, dim=1)
        return score
