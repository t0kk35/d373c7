"""
Common classes for all Pytorch Models
(c) 2020 d373c7
"""
import torch.nn as nn
from typing import Any


class PyTorchLayerException(Exception):
    """Standard exception raised in _Layer"""
    def __init__(self, message: str):
        super().__init__('Error in Layer: ' + message)


class _Layer(nn.Module):
    def __init__(self):
        super(_Layer, self).__init__()

    def _forward_unimplemented(self, *inp: Any) -> None:
        raise NotImplemented(f'Unimplemented <_forward_unimplemented>  in <{self.__class__.__name__}>')
