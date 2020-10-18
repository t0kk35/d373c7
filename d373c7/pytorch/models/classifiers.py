"""
Module for classifier Models
(c) 2020 d373c7
"""
import torch
from .common import _Model
from typing import List


class ClassifierModel(_Model):
    def __init__(self):
        _Model.__init__(self)

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # Assume everything except the last list is the input
        return ds[0:len(ds) - 1]

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # Assume the last list are the labels
        return ds[len(ds) - 1:len(ds)]

    @property
    def default_metrics(self) -> List[str]:
        return ['acc', 'loss']


class BinaryClassifier(ClassifierModel):
    pass
