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
from ...features import FeatureCategorical, LEARNING_CATEGORY_LABEL
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
        return f'Number of parameters : {self.num_parameters}. Loss : {self.loss_fn}'


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
        tensor_def: A TensorDefinitionMulti object. Needed to know how the set up and use the heads of the model.
        defaults: A ModelDefaults object containing the defaults to be used.
    """
    def __init__(self, tensor_def: TensorDefinitionMulti, defaults: ModelDefaults):
        super(_ModelGenerated, self).__init__(defaults)
        self.val_td_is_inference_ready(tensor_def)
        self._loss_fn = None
        self._tensor_def = self.val_is_td_multi(tensor_def)
        self.val_td_is_inference_ready(self._tensor_def)
        self._x_indexes = []
        self._head_indexes = []

    @property
    def loss_fn(self) -> _LossBase:
        if self._loss_fn is None:
            raise PyTorchModelException(
                f'Can not get loss function. The child model has not set a loss function. This is bad....'
            )
        return self._loss_fn

    def set_loss_fn(self, loss_fn: _LossBase):
        self._loss_fn = loss_fn

    def optimizer(self, lr=None, wd=None) -> _Optimizer:
        return AdamWOptimizer(self, lr, wd)

    def set_up_heads(self, defaults: ModelDefaults, tds: List[TensorDefinition], streams: List[_ModelStream]):
        x_offset = 0
        # Add a head layer to each stream.
        for td, s in zip(tds, streams):
            head = self.create_head(td, defaults)
            self._x_indexes.extend([x+x_offset for x in head.x_indexes])
            self._head_indexes.append([x+x_offset for x in head.x_indexes])
            x_offset = self._x_indexes[-1] + 1
            s.add(td.name, head, head.output_size)

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        ds = [ds[x] for x in self._x_indexes]
        return ds

    @property
    def head_indexes(self) -> List[Tuple[int]]:
        """Method that returns the x-indexes for each of the heads in this model. The x-indexes are the indexes of the
        input dataset that correspond to a specific head.

        :return: A list of int Tuples containing the x-indexes for each of the heads. There is a list entry per
            stream/tensor_definition. Within the list there is a tuple of int contain indexes.
        """
        return self._head_indexes

    @property
    def tensor_definition(self) -> TensorDefinitionMulti:
        """Property Method that returns the TensorDefinitionMulti object used to create this Generated Model

        :return: A TensorDefinitionMulti object which was used to create this model.
        """
        return self._tensor_def

    def embedding_weight(self, feature: FeatureCategorical, as_numpy=False) -> torch.Tensor:
        """Return the 'embedding weights' of a feature. This method returns the tensor containing the weights from the
        'head' layer. It will look for the input feature across all head layers of the model.

        :param: feature: The feature for which to get the embedding weights. It must be a 'FeatureCategorical'
        :param: as_numpy: Boolean flag indicating if the returned value is numpy object. Default is False in which case
        the return is a torch.Tensor
        """
        i = [i for i, td in enumerate(self.tensor_definition.tensor_definitions) if feature in td.features]
        w = self.heads[i[0]].embedding_weight(feature)
        if as_numpy:
            w = w.cpu().detach().numpy()
        return w

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
                    f'Expected parameter with name {key} to be of type {int.__name__}'
                )
            return param

    @staticmethod
    def get_float_parameter(key: str, kwargs: dict, default=None):
        param = kwargs.get(key, None)
        if param is None and default is None:
            raise PyTorchModelException(
                f'Could not find mandatory parameter with name {key}. Please provide a named parameter with this name'
            )
        if param is None:
            return default
        else:
            if not isinstance(param, float):
                raise PyTorchModelException(
                    f'Expected parameter with name {key} to be of type {float.__name__}'
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
    def td_to_multi(tensor_def: Union[TensorDefinition, TensorDefinitionMulti]) -> TensorDefinitionMulti:
        if isinstance(tensor_def, TensorDefinitionMulti):
            return tensor_def
        elif isinstance(tensor_def, TensorDefinition):
            return TensorDefinitionMulti([tensor_def])
        else:
            raise (
                f'Expected an instance of either "TensorDefinition" of "TensorDefinitionMulti". ' +
                f'Got {tensor_def.__class__}'
            )

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

    @staticmethod
    def label_tensor_def(tensor_def: TensorDefinitionMulti) -> List[TensorDefinition]:
        """ Method to find the tensor definition that holds the label

        :param tensor_def: A TensorDefinitionMulti Object.
        :return: The tensor_definition that hold the labels.
        """
        label_td = [td for td in tensor_def.tensor_definitions if LEARNING_CATEGORY_LABEL in td.learning_categories]
        if len(label_td) == 0:
            raise PyTorchModelException(
                f'Could not find the label tensor. A classifier model needs a label. Please make sure there is a ' +
                f'Tensor Definition that has a features of type {LEARNING_CATEGORY_LABEL.name}'
            )
        return label_td

    @property
    def heads(self) -> List[TensorDefinitionHead]:
        """Returns the heads of the model. This is the first layer of each stream which is in the model. This method
        will raise an exception if no head layers were found.

        :return: A layer object which is the 'head' layer of the model
        """
        # A stream can be a single layer or an nn.Sequential.
        hd = [s[0] if hasattr(s, "__getitem__") else s for s in self.streams]
        if len(hd) == 0:
            raise PyTorchModelException(
                f'Did not find any head layer in the streams. Aborting...'
            )
        for h in hd:
            if not isinstance(h, TensorDefinitionHead):
                raise PyTorchModelException(
                    f'Expected the first layer to be an instance of TensorDefinitionHead. But got <{h}>'
                )
        return hd
