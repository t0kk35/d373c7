"""
Common classes for all Pytorch Models
(c) 2020 d373c7
"""

import os
import torch
import torch.nn as nn
from ..common import PyTorchTrainException, _History
from ..layers.common import Layer
from ..layers.base import TensorDefinitionHead
from ..loss import _LossBase
from ..optimizer import _Optimizer, AdamWOptimizer
from ...features.tensor import TensorDefinition, TensorDefinitionMulti
from collections import OrderedDict
from typing import List, Any, Optional, Type, Union, Tuple
from math import log, ceil, floor


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


class _ModelStream:
    """Class used in Generator models to set-up a set of layers to be executed sequentially. This is not an nn.Module.
    It's just a place-holder class to bundle the layers. By calling the create, an nn.Sequential will be returned which
    can be used in models.

    Args:
        name: A name for the Stream.
        layer: Optional. This can be used to create a stream and immediately add a layer. Default it is None.
    """
    def __init__(self, name: str, layer: Layer = None):
        self.name = name
        self.layers = OrderedDict()
        if layer is not None:
            self.layers.update({name: layer})
            self._out_size = layer.output_size
        else:
            self._out_size = -1

    def add(self, name: str, layer: Union[Layer, nn.Module], new_size: int):
        self.layers.update({name: layer})
        self._out_size = new_size

    def create(self) -> Union[nn.Sequential, Layer]:
        if len(self.layers) == 1:
            # There is just one layer, return the first item from the Dict.
            return next(iter(self.layers.values()))
        else:
            # There is more than one layer. Build a nn.Sequential.
            return nn.Sequential(self.layers)

    @property
    def out_size(self) -> int:
        if self._out_size == -1:
            raise PyTorchModelException(
                f'Outsize has not been set on stream {self.name}. Can not get the out_size'
            )
        return self._out_size


class _ModelGenerated(_Model):
    """ Base class for Generator Models. It stores the loss function and sets the default Adam Optimizer.
    It also has some common helper functions.

    Args:
        defaults: A ModelDefaults object containing the defaults to be used.
        loss_fn: The loss function to use.
    """
    def __init__(self, defaults: ModelDefaults, loss_fn: _LossBase):
        super(_ModelGenerated, self).__init__(defaults)
        self._loss_fn = loss_fn

    def loss_fn(self) -> _LossBase:
        return self._loss_fn

    def optimizer(self, lr=None, wd=None) -> _Optimizer:
        return AdamWOptimizer(self, lr, wd)

    @staticmethod
    def is_param_defined(key: str, kwargs: dict) -> bool:
        param = kwargs.get(key, None)
        if param is None:
            return False
        else:
            return True

    @staticmethod
    def get_list_parameter(key: str, p_type: type, kwargs: dict, default=None):
        param = kwargs.get(key, None)
        if param is None and default is None:
            raise PyTorchModelException(
                f'Could not find mandatory parameter with name {key}. Please provide a named parameter with this name'
            )
        if param is None:
            return default
        else:
            if not isinstance(param, List):
                raise PyTorchModelException(
                    f'Expected parameter with name {key} to be of type {List.__name__}'
                )
            for p in param:
                if not isinstance(p, p_type):
                    raise PyTorchModelException(
                        f'Expected parameter with name {key} to have all {p_type.__name__}. Found {p}'
                    )
            return param

    @staticmethod
    def get_list_of_tuples_parameter(key: str, p_type: type, kwargs: dict, default=None) -> List[Tuple]:
        param = kwargs.get(key, None)
        if param is None and default is None:
            raise PyTorchModelException(
                f'Could not find mandatory parameter with name {key}. Please provide a named parameter with this name'
            )
        if param is None:
            return default
        else:
            if not isinstance(param, List):
                raise PyTorchModelException(
                    f'Expected parameter with name {key} to be of type {List.__name__}'
                )
            for p in param:
                if not isinstance(p, Tuple):
                    raise PyTorchModelException(
                        f'Expected the list with name {key} to contain Tuples. It contains {p}'
                    )
                else:
                    for t in p:
                        if not isinstance(t, p_type):
                            raise PyTorchModelException(
                                f'Expected parameter with name {key} to have all {p_type.__name__}. Found {t}'
                            )
            return param

    @staticmethod
    def get_str_parameter(key: str, kwargs: dict, default=None):
        param = kwargs.get(key, None)
        if param is None and default is None:
            raise PyTorchModelException(
                f'Could not find mandatory parameter with name {key}. Please provide a named parameter with this name'
            )
        if param is None:
            return default
        else:
            if not isinstance(param, str):
                raise PyTorchModelException(
                    f'Expected parameter with name {key} to be of type {str.__name__}'
                )
            return param

    @staticmethod
    def get_int_parameter(key: str, kwargs: dict, default=None):
        param = kwargs.get(key, None)
        if param is None and default is None:
            raise PyTorchModelException(
                f'Could not find mandatory parameter with name {key}. Please provide a named parameter with this name'
            )
        if param is None:
            return default
        else:
            if not isinstance(param, int):
                raise PyTorchModelException(
                    f'Expected parameter with name {key} to be of type {str.__name__}'
                )
            return param

    @staticmethod
    def closest_power_of_2(inp: int):
        res = 2**floor(log(inp, 2)), 2**ceil(log(inp, 2))
        return min(res, key=lambda x: abs(inp-x))

    @staticmethod
    def create_head(tensor_def: TensorDefinition, defaults: ModelDefaults) -> TensorDefinitionHead:
        mn = defaults.get_int('emb_min_dim')
        mx = defaults.get_int('emb_max_dim')
        do = defaults.get_float('emb_dropout')
        return TensorDefinitionHead(tensor_def, do, mn, mx)

    @staticmethod
    def val_is_td_multi(tensor_def: Union[TensorDefinition, TensorDefinitionMulti]) -> TensorDefinitionMulti:
        if not isinstance(tensor_def, TensorDefinitionMulti):
            raise PyTorchModelException(
                f'This Model Generator only must be given a {TensorDefinitionMulti.__name__} as input'
            )
        return tensor_def

    @staticmethod
    def val_td_is_inference_ready(tensor_def: TensorDefinitionMulti):
        for td in tensor_def.tensor_definitions:
            if not td.inference_ready:
                raise PyTorchModelException(
                    f'Tensor Definition {td.name} is not read for inference. A Tensor Definition can be made ready ' +
                    f'for inference by using an engine method with the "inference=False" set'
                )


# TODO can be removed
# class TensorHeadModel(_Model):
#     def __init__(self, tensor_def: TensorDefinition, defaults: ModelDefaults):
#         super(TensorHeadModel, self).__init__(defaults)
#         mn = self.defaults.get_int('emb_min_dim')
#         mx = self.defaults.get_int('emb_max_dim')
#         do = self.defaults.get_float('emb_dropout')
#         self._tensor_def = tensor_def
#         self.head = TensorDefinitionHead(tensor_def, do, mn, mx)
#
#     @property
#     def tensor_definition(self):
#         return self._tensor_def
#
#     def forward(self, x):
#         x = self.head(x)
#         return x
#
#     def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
#         x = [ds[x] for x in self.head.x_indexes]
#         return x
#
#     def embedding_weights(self, feature: FeatureIndex, as_numpy: bool = False):
#         w = self.head.embedding_weight(feature)
#         if as_numpy:
#             w = w.cpu().detach().numpy()
#         return w
#
#     @property
#     def output_size(self) -> int:
#         return self.head.output_size
