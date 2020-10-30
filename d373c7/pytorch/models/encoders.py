"""
Module for encoder Models
(c) 2020 d373c7
"""
import logging
import torch
import torch.nn as nn
from .common import _Model, _TensorHeadModel, ModelDefaults, PyTorchModelException
from ..common import _History
from ..layers import LinDropAct
from ..optimizer import _Optimizer, AdamWOptimizer
from ...features import TensorDefinition, LEARNING_CATEGORY_LABEL, LEARNING_CATEGORY_BINARY
from ...features import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_CATEGORICAL
from typing import List, Any

logger = logging.getLogger(__name__)


class AutoEncoderHistory(_History):
    loss_key = 'loss'

    def __init__(self, *args):
        dl = self._val_argument(args)
        _History.__init__(self, dl)
        self._history = {m: [] for m in [AutoEncoderHistory.loss_key]}
        self._running_loss = 0

    def end_step(self, *args):
        AutoEncoderHistory._val_is_tensor_list(args[1])
        AutoEncoderHistory._val_is_tensor(args[2])
        loss = args[2]
        self._running_loss += loss.item()
        super(AutoEncoderHistory, self).end_step(loss)

    def end_epoch(self):
        self._history[AutoEncoderHistory.loss_key].append(round(self._running_loss/self.steps, 4))
        self._running_loss = 0
        super(AutoEncoderHistory, self).end_epoch()


class _AutoEncoderModel(_Model):
    def __init__(self, tensor_def: TensorDefinition, defaults: ModelDefaults):
        super(_AutoEncoderModel, self).__init__(tensor_def, defaults)

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # With encoders we return the same as the input so we call get_x
        return self.get_x(ds)

    def optimizer(self, lr=None, wd=None) -> _Optimizer:
        return AdamWOptimizer(self, lr, wd)


class AutoEncoderDefaults(ModelDefaults):
    def __init__(self):
        super(AutoEncoderDefaults, self).__init__()
        self.set('lin_interlayer_drop_out', 0.1)


class _LinearEncoder(_TensorHeadModel):
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int], defaults: ModelDefaults):
        super(_LinearEncoder, self).__init__(tensor_def, defaults)
        do = defaults.get_float('lin_interlayer_drop_out')
        ly = [(i, do) for i in layers]
        self.linear = LinDropAct(self.head.output_size, ly)
        self.latent = nn.Linear(self.linear.output_size, latent_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.latent(x)
        return x


class _LinearDecoder(nn.Module):
    def _forward_unimplemented(self, *inp: Any) -> None:
        raise NotImplemented(f'_forward_unimplemented not implemented in {self.__class__.__name__}')

    def __init__(self, latent_dim, layers: List[int], defaults: ModelDefaults):
        super(_LinearDecoder, self).__init__()
        do = defaults.get_float('lin_interlayer_drop_out')
        ly = [(i, do) for i in layers]
        self.linear = LinDropAct(latent_dim, ly)

    def forward(self, x):
        x = self.linear(x)
        return x


class _LinearAutoEncoder(_AutoEncoderModel):
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, defaults):
        super(_LinearAutoEncoder, self).__init__(tensor_def, defaults)
        self.encoder = _LinearEncoder(tensor_def, latent_dim, [16], defaults)
        self.decoder = _LinearDecoder(latent_dim, [16], defaults)


class LinearToBinaryAutoEncoder(_LinearAutoEncoder):
    """Auto-encoder which will only take binary input and will return binary output. It will use BinaryCrossEntropy loss

    Args:
        tensor_def: The Tensor Definition that will be used
        latent_dim : The size of the latent dimension. As integer value
        layers: A list of integers. Drives the number of layers and their size. For instance [64,32,16] would create a
        neural net with 3 layers of size 64, 32 and 16 respectively. Note that the NN will also have an additional
        input layer (depends on the tensor_def) and an output layer.
        defaults: Optional defaults object. If omitted, the ClassifierDefaults will be used.
    """
    def _val_layers(self, layers: List[int]):
        if not isinstance(layers, List):
            raise PyTorchModelException(
                f'Layers parameter for {self.__class__.__name__} should have been a list. It was <{type(layers)}>'
            )
        if len(layers) == 0:
            raise PyTorchModelException(
                f'Layers parameter for {self.__class__.__name__} should was an empty list'
            )
        if not isinstance(layers[0], int):
            raise PyTorchModelException(
                f'Layers parameter for {self.__class__.__name__} should contain ints. It contains <{type(layers[0])}>'
            )

    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int],
                 defaults=AutoEncoderDefaults()):
        super(LinearToBinaryAutoEncoder, self).__init__(tensor_def, latent_dim, defaults)
        self._val_layers(layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
