"""
Module for encoder Models
(c) 2020 d373c7
"""
import logging
import torch
import torch.nn as nn
from .common import _Model, _TensorHeadModel, ModelDefaults, PyTorchModelException
from ..common import _History
from ..layers import LinDropAct, BinaryOutput
from ..optimizer import _Optimizer, AdamWOptimizer
from ..loss import _LossBase, SingleLabelBCELoss
from ...features import TensorDefinition, LEARNING_CATEGORY_LABEL, LEARNING_CATEGORY_BINARY
from typing import List, Any

logger = logging.getLogger(__name__)


class AutoEncoderHistory(_History):
    loss_key = 'loss'

    def __init__(self, *args):
        dl = self._val_argument(args)
        h = {m: [] for m in [AutoEncoderHistory.loss_key]}
        _History.__init__(self, dl, h)
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

    def history(self, *args) -> _History:
        return AutoEncoderHistory(*args)


class AutoEncoderDefaults(ModelDefaults):
    def __init__(self):
        super(AutoEncoderDefaults, self).__init__()
        self.emb_dim(4, 100, 0.2)
        self.set('lin_interlayer_drop_out', 0.1)

    def emb_dim(self, minimum: int, maximum: int, dropout: float):
        self.set('emb_min_dim', minimum)
        self.set('emb_max_dim', maximum)
        self.set('emb_dropout', dropout)


class _LinearEncoder(_TensorHeadModel):
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int], defaults: ModelDefaults):
        super(_LinearEncoder, self).__init__(tensor_def, defaults)
        do = defaults.get_float('lin_interlayer_drop_out')
        ly = [(i, do) for i in layers]
        self.linear = LinDropAct(self.head.output_size, ly)
        self.latent = nn.Linear(self.linear.output_size, latent_dim)

    def forward(self, x):
        x = _TensorHeadModel.forward(self, x)
        x = self.linear(x)
        x = self.latent(x)
        return x


class _LinearDecoder(nn.Module):
    def _forward_unimplemented(self, *inp: Any) -> None:
        raise NotImplemented(f'_forward_unimplemented not implemented in {self.__class__.__name__}')

    def __init__(self, latent_dim, layers: List[int], defaults: ModelDefaults):
        super(_LinearDecoder, self).__init__()
        self._out_size = layers[-1]
        do = defaults.get_float('lin_interlayer_drop_out')
        ly = [(i, do) for i in layers]
        self.linear = LinDropAct(latent_dim, ly)

    @property
    def output_size(self) -> int:
        return self._out_size

    def forward(self, x):
        x = self.linear(x)
        return x


class _LinearAutoEncoder(_AutoEncoderModel):
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, defaults):
        super(_LinearAutoEncoder, self).__init__(tensor_def, defaults)
        self.encoder = _LinearEncoder(tensor_def, latent_dim, [16], defaults)
        self.decoder = _LinearDecoder(latent_dim, [16], defaults)

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # Call the get_x of the _TensorHeadModel
        return self.encoder.get_x(ds)


class LinearToBinaryAutoEncoder(_LinearAutoEncoder):
    """Auto-encoder which will only take binary input, (not categorical or continuous features) and will return binary
    output. The encoder will use linear layers to condense until it reaches the latent dim size. The decoder will start
    from the latent feature and reconstruct until is reaches the original input size.
    It will use BinaryCrossEntropy loss.

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

    def _val_only_bin(self, tensor_def: TensorDefinition):
        fl = len(tensor_def.features)
        bl = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY))
        ll = len(tensor_def.filter_features(LEARNING_CATEGORY_LABEL))
        if fl - ll > bl:
            raise PyTorchModelException(
                f'The tensor definition for  {self.__class__.__name__} should only contain binary learning type ' +
                f'features (and labels those will be ignored)'
            )

    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int],
                 defaults=AutoEncoderDefaults()):
        super(LinearToBinaryAutoEncoder, self).__init__(tensor_def, latent_dim, defaults)
        self._val_layers(layers)
        self._val_only_bin(tensor_def)
        self._input_size = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY, True))
        self.out = BinaryOutput(self.decoder.output_size, self._input_size)
        self._loss_fn = SingleLabelBCELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out(x)
        return x

    def loss_fn(self) -> _LossBase:
        return self._loss_fn
