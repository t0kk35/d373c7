"""
Common classes for all Pytorch Models
(c) 2020 d373c7
"""

import os
import torch
import torch.nn as nn
from ..common import PyTorchTrainException, _History
from ..layers.base import TensorDefinitionHead
from ..loss import _LossBase
from ..optimizer import _Optimizer
from ...features.base import FeatureIndex
from ...features.tensor import TensorDefinition
from typing import List, Any, Optional, Type


class PyTorchModelException(Exception):
    """Standard exception raised during training"""
    def __init__(self, message: str):
        super().__init__('Error in Model: ' + message)


class ModelDefaults:
    """Object where model defaults can be stored. This will avoid having too many parameters in each model creation.
    """
    @staticmethod
    def _val_not_none(value: Optional[Any], key: str):
        if value is None:
            raise PyTorchModelException(
                f'Could not find default for key <{key}>'
            )

    @staticmethod
    def _val_type(value: Optional[Any], e_type: Type, key: str):
        if not isinstance(value, e_type):
            raise PyTorchModelException(
                f'Value for key <{key}> not of correct type. Got <{type(value)}, expected {e_type}>'
            )

    def __init__(self):
        self._defaults = {}

    def get_bool(self, key: str) -> bool:
        v = self._defaults.get(key, None)
        ModelDefaults._val_not_none(v, key)
        ModelDefaults._val_type(v, bool, key)
        return v

    def get_str(self, key: str) -> str:
        v = self._defaults.get(key, None)
        ModelDefaults._val_not_none(v, key)
        ModelDefaults._val_type(v, str, key)
        return v

    def get_int(self, key: str) -> int:
        v = self._defaults.get(key, None)
        ModelDefaults._val_not_none(v, key)
        ModelDefaults._val_type(v, int, key)
        return v

    def get_float(self, key: str) -> float:
        v = self._defaults.get(key, None)
        ModelDefaults._val_not_none(v, key)
        ModelDefaults._val_type(v, float, key)
        return v

    def set(self, key: str, value: any):
        self._defaults[key] = value


class _Model(nn.Module):
    def __init__(self, defaults: ModelDefaults):
        nn.Module.__init__(self)
        self._model_defaults = defaults

    def _forward_unimplemented(self, *inp: Any) -> None:
        raise NotImplemented('Abstract method _forward_unimplemented not implemented')

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        """Get the values that are considered the x values. I.e the independent variables, I.e. NOT the label.

        :param ds: A list of tensors as read from a DataLoader object.
        :return: A list of tensors to be used as input to a neural net.
        """
        raise NotImplemented(
            f'get_x method not unimplemented in {self.__class__.name}. Children of _Model should implement this method'
        )

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        """Get the values that are considered the y values. I.e. the dependent variable, I.e. the label

        :param ds: A list of tensors as read from a DataLoader object.
        :return: A list of tensors to be use as label for the neural net.
        """
        raise NotImplemented(
            f'get_y method not unimplemented in {self.__class__.name}. Children of _Model should implement this method'
        )

    @property
    def label_index(self) -> int:
        raise NotImplemented(f'Class label index should be implemented by children')

    @property
    def loss_fn(self) -> _LossBase:
        raise NotImplemented(f'Loss Function getter not implemented in base _Model Class. '
                             f'Needs to be implemented by the child classes')

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def defaults(self) -> ModelDefaults:
        return self._model_defaults

    def optimizer(self, lr=None, wd=None) -> _Optimizer:
        raise NotImplemented(
            f'optimizer method not unimplemented in {self.__class__.name}. '
            + f'Children of _Model should implement this method'
        )

    def history(self, *args) -> _History:
        raise NotImplemented(f'History getter not implemented in base _History class. Must be implemented by children')

    def extra_repr(self) -> str:
        return f'Number of parameters : {self.num_parameters}'


class _ModelManager:
    def __init__(self, model: _Model, device: torch.device):
        self._model = model
        self._device = device

    @staticmethod
    def _get_x(model: _Model, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        return model.get_x(ds)

    @staticmethod
    def _get_y(model: _Model, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        return model.get_y(ds)

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def save(self, path: str):
        if os.path.exists(path):
            raise PyTorchTrainException(f'File {path} already exists. Not overriding model')
        torch.save(self._model.state_dict(), path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise PyTorchTrainException(f'File {path} does not exist. Not loading model')
        self._model.load_state_dict(torch.load(path))


class TensorHeadModel(_Model):
    def __init__(self, tensor_def: TensorDefinition, defaults: ModelDefaults):
        super(TensorHeadModel, self).__init__(defaults)
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        self._tensor_def = tensor_def
        self.head = TensorDefinitionHead(tensor_def, do, mn, mx)

    @property
    def tensor_definition(self):
        return self._tensor_def

    def forward(self, x):
        x = self.head(x)
        return x

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        x = [ds[x] for x in self.head.x_indexes]
        return x

    def embedding_weights(self, feature: FeatureIndex, as_numpy: bool = False):
        w = self.head.embedding_weight(feature)
        if as_numpy:
            w = w.cpu().detach().numpy()
        return w

    @property
    def output_size(self) -> int:
        return self.head.output_size
