"""
Module for encoder Models
(c) 2020 d373c7
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from .common import _Model, ModelDefaults, PyTorchModelException, TensorDefinitionHead
from ..common import _History
from ..layers import LinDropAct, BinaryOutput
# noinspection PyProtectedMember
from ..layers.common import _Layer
from ..layers import VAELinearToLatent, VAELatentToLinear
from ..layers import CategoricalLogSoftmax1d, CategoricalLogSoftmax2d, SigmoidOut
from ..layers import LSTMEncoder, LSTMDecoder, GRUEncoder, GRUDecoder
from ..layers import ConvolutionalEncoder, ConvolutionalDecoder
from ..optimizer import _Optimizer, AdamWOptimizer
from ..loss import _LossBase, SingleLabelBCELoss, BinaryVAELoss, MultiLabelNLLLoss, MultiLabelBCELoss
from ..loss import MultiLabelNLLLoss2d
from ...features import TensorDefinition, FeatureCategorical, LEARNING_CATEGORY_LABEL, LEARNING_CATEGORY_BINARY
from ...features import LEARNING_CATEGORY_CATEGORICAL
from typing import List, Union, Tuple

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


class AutoEncoderDefaults(ModelDefaults):
    def __init__(self):
        super(AutoEncoderDefaults, self).__init__()
        self.emb_dim(4, 100, 0.2)
        self.set('lin_interlayer_drop_out', 0.1)
        self.set('vae_loss_kl_weight', 1.0)

    def emb_dim(self, minimum: int, maximum: int, dropout: float):
        self.set('emb_min_dim', minimum)
        self.set('emb_max_dim', maximum)
        self.set('emb_dropout', dropout)

    def set_vae_kl_weight(self, weight: float):
        self.set('vae_loss_kl_weight', weight)


class AutoEncoderConvolutionalDefaults(AutoEncoderDefaults):
    def __init__(self):
        super(AutoEncoderConvolutionalDefaults, self).__init__()
        self.set_dense(False)
        self.set_encoder_conv_dropout(0.0)
        self.set_decoder_conv_dropout(0.1)
        self.set_encoder_conv_batch_norm_interval(1000)
        self.set_decoder_conv_batch_norm_interval(2)

    def set_dense(self, dense: bool):
        self.set('conv_body_dense', dense)

    def set_encoder_conv_dropout(self, dropout: float):
        self.set('enc_conv_dropout', dropout)

    def set_encoder_conv_batch_norm_interval(self, interval: int):
        self.set('enc_conv_bn_interval', interval)

    def set_decoder_conv_dropout(self, dropout: float):
        self.set('dec_conv_dropout', dropout)

    def set_decoder_conv_batch_norm_interval(self, interval: int):
        self.set('dec_conv_bn_interval', interval)


class _Encoder(_Layer):
    """Base Encoder class. Internal. Used to abstract some of the logic of heads.

    Args:
        defaults: An instance of AutoEncoderDefaults, it contains some of the default settings to use.
    """
    def __init__(self, defaults: AutoEncoderDefaults):
        self._defaults = defaults
        super(_Encoder, self).__init__()
        self.head = self.init_head()
        self.body = self.init_body()
        self.latent = self.init_latent()

    def init_head(self) -> _Layer:
        """Method that should be implemented by the children that inherit _Encoder to set the head layer. The head is
        the first layer of our model.
        """
        raise NotImplemented('Should be implemented by Children')

    def init_body(self) -> _Layer:
        """Method that should be implemented by the children that inherit _Encoder to set the body layer. The body is
        the part of the model that compresses the input until it is ready to to be encoded to the latent variables.
        """
        raise NotImplemented('Should be implemented by Children')

    def init_latent(self) -> [_Layer, nn.Module, None]:
        """Method that should be implemented by the children that inherit _Encoder to set the latent layer. The is
        latent layer is the part of the model that encodes the output of the body to a latent representation.
        """
        raise NotImplemented('Should be implemented by Children')

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        """Specific get_x for the encoder. It calls the get_x of the head layer. It is used so the _AutoEncoder can
        call the get_x of the underlying head.

        :argument ds: a List of tensors. As found in the data-loader
        :return: A list of tensors that should be used as input to the model

        """
        # Call the get_x of the head layer.
        return self.head.get_x(ds)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if self.latent is not None:
            x = self.latent(x)
        return x

    @property
    def defaults(self) -> AutoEncoderDefaults:
        """Return the defaults of the auto-encoder

        :return: The defaults of this Auto-encoder. Is an instance of AutoEncoderDefaults
        """
        return self._defaults

    def embedding_weights(self, feature: FeatureCategorical, as_numpy=True) -> Union[torch.Tensor, np.array]:
        """Method that will call get the weights of the embedding layer of the head layer.

        :argument feature: The feature of which the weights need to fetched. Should be an instance of FeatureCategorical
        :argument as_numpy: boolean flag indicating if the weights should be returned as numpy array. If false the
        output will be a tensor
        :return: The embedding weight matrix as numpy array or torch Tensor.
        """
        w = self.head.embedding_weight(feature)
        if as_numpy:
            w = w.cpu().detach().numpy()
        return w


class _Decoder(_Layer):
    """Base Decoder class. Internal. This is used to abstract some of the base logic of decoders.

    Args:
        defaults: An instance of AutoEncoderDefaults, it contains some of the default settings to use.
    """
    def __init__(self, defaults: AutoEncoderDefaults):
        self._defaults = defaults
        super(_Decoder, self).__init__()
        self.latent = self.init_latent()

    def init_latent(self) -> _Layer:
        raise NotImplemented('Should be implemented by Children')

    def init_out(self) -> _Layer:
        raise NotImplemented('Should be implemented by Children')

    def forward(self, x):
        x = self.latent(x)
        return x

    @property
    def defaults(self) -> AutoEncoderDefaults:
        return self._defaults


class _AutoEncoderModel(_Model):
    def _val_layers(self, layers: List[int]):
        if not isinstance(layers, List):
            raise PyTorchModelException(
                f'Layers parameter for {self.__class__.__name__} should have been a list. It was <{type(layers)}>'
            )
        if len(layers) == 0:
            raise PyTorchModelException(
                f'Layers parameter for {self.__class__.__name__}  was an empty list. It should have at least one entry'
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

    def _val_only_cat(self, tensor_def: TensorDefinition):
        fl = len(tensor_def.features)
        cl = len(tensor_def.filter_features(LEARNING_CATEGORY_CATEGORICAL))
        ll = len(tensor_def.filter_features(LEARNING_CATEGORY_LABEL))
        if fl - ll > cl:
            raise PyTorchModelException(
                f'The tensor definition for  {self.__class__.__name__} should only contain categorical learning type ' +
                f'features (and labels will be ignored)'
            )

    def __init__(self, encoder: _Encoder, decoder: _Decoder, out: _Layer, loss_fn: _LossBase,
                 defaults: AutoEncoderDefaults):
        super(_AutoEncoderModel, self).__init__(defaults)
        self.encoder = encoder
        self.decoder = decoder
        self.out = out
        self._loss_fn = loss_fn

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # Call the get_x of the Encoder
        return self.encoder.get_x(ds)

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # With encoders we return the same as the input so we call get_x
        return self.get_x(ds)

    def optimizer(self, lr=None, wd=None) -> _Optimizer:
        return AdamWOptimizer(self, lr, wd)

    def history(self, *args) -> _History:
        return AutoEncoderHistory(*args)

    def loss_fn(self) -> _LossBase:
        return self._loss_fn

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out(x)
        return x

    def embedding_weights(self, feature: FeatureCategorical, as_numpy=True) -> Union[torch.Tensor, np.array]:
        """Method that will call get the weights of the embedding layer in the encoder.

        :argument feature: The feature of which the weights need to fetched. Should be an instance of FeatureCategorical
        :argument as_numpy: boolean flag indicating if the weights should be returned as numpy array. If false the
        output will be a tensor
        :return: The embedding weight matrix as numpy array or torch Tensor.
        """
        return self.encoder.embedding_weights(feature, as_numpy)


class _EncoderSingle(_Encoder):
    """Base Encoder class. Internal. Used to abstract some of the logic of heads for encoders that have a single head.
    I.e. which have one Tensor Definition defining them.

    Args:
        defaults: An instance of AutoEncoderDefaults, it contains some of the default settings to use.
    """
    def __init__(self, tensor_def: TensorDefinition, defaults: AutoEncoderDefaults):
        self._t_def = tensor_def
        super(_EncoderSingle, self).__init__(defaults)

    def init_head(self) -> _Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHead(self._t_def, do, mn, mx)

    @property
    def tensor_def(self) -> TensorDefinition:
        return self._t_def


class _LinearEncoder(_EncoderSingle):
    """Base class for Encoders that encode using linear layers.

    Args:
        tensor_def: A Tensor Definition describing the input.
        layers: List of integers. Each entry will become layer and the int-value the size of the layer.
        defaults: An instance of AutoEncoderDefaults, it contains some of the default settings to use.
    """
    def __init__(self, tensor_def: TensorDefinition, layers: List[int], defaults: AutoEncoderDefaults):
        self._layers = layers
        super(_LinearEncoder, self).__init__(tensor_def, defaults)

    def init_body(self) -> _Layer:
        do = self.defaults.get_float('lin_interlayer_drop_out')
        ly = [(i, do) for i in self._layers]
        return LinDropAct(self.head.output_size, ly)


class LatentLinearEncoder(_LinearEncoder):
    """An encoder that uses linear layers and outputs a linear layer as latent variable.

    Args:
        tensor_def: A Tensor Definition describing the input.
        latent_dim: The size of the latent dimension
        layers: List of integers. Each entry will become layer and the int-value the size of the layer.
        defaults: An instance of AutoEncoderDefaults, it contains some of the default settings to use.
    """
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int], defaults: AutoEncoderDefaults):
        self._latent_dim = latent_dim
        super(LatentLinearEncoder, self).__init__(tensor_def, layers, defaults)

    def init_latent(self) -> [_Layer, nn.Module]:
        return nn.Linear(self.body.output_size, self._latent_dim)


class LatentVAEEncoder(_LinearEncoder):
    """An encoder that uses linear layers and outputs a VAE style latent variable. It outputs a linear layer with the
    averages and a linear layer with the standard deviations.

    Args:
        tensor_def: A Tensor Definition describing the input.
        latent_dim: The size of the latent dimension
        layers: List of integers. Each entry will become layer and the int-value the size of the layer.
        defaults: An instance of AutoEncoderDefaults, it contains some of the default settings to use.
    """
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int], defaults: AutoEncoderDefaults):
        self._latent_dim = latent_dim
        super(LatentVAEEncoder, self).__init__(tensor_def, layers, defaults)

    def init_latent(self) -> [_Layer, nn.Module]:
        return VAELinearToLatent(self.body.output_size, self._latent_dim)


class LatentRNNEncoder(_EncoderSingle):
    def __init__(self, tensor_def: TensorDefinition, node_type: str, layers: List[int],
                 defaults: AutoEncoderDefaults):
        self._node_type = node_type
        self._layers = layers
        super(LatentRNNEncoder, self).__init__(tensor_def, defaults)

    def init_body(self) -> _Layer:
        if self._node_type == 'LSTM':
            rnn = LSTMEncoder(self.head.output_size, self._layers)
        else:
            rnn = GRUEncoder(self.head.output_size, self._layers)
        return rnn

    def init_latent(self) -> [_Layer, nn.Module, None]:
        return None


class LatentCNNEncoder(_EncoderSingle):
    def __init__(self, tensor_def: TensorDefinition, conv_layers: List[Tuple[int, int]],
                 defaults: AutoEncoderConvolutionalDefaults):
        self._conv_layers = conv_layers
        self._latent_series_length = 0
        super(LatentCNNEncoder, self).__init__(tensor_def, defaults)

    def init_body(self) -> _Layer:
        do = self.defaults.get_float('enc_conv_dropout')
        bn_int = self.defaults.get_int('enc_conv_bn_interval')
        s_length = [s[1] for s in self._t_def.shapes if len(s) == 3][0]
        cnn = ConvolutionalEncoder(self.head.output_size, s_length, self._conv_layers, do, batch_norm_interval=bn_int)
        self._latent_series_length = cnn.output_series_length
        return cnn

    def init_latent(self) -> [_Layer, nn.Module, None]:
        return None

    @property
    def latent_series_length(self) -> int:
        return self._latent_series_length


class LatentLinearDecoder(_Decoder):
    """A decoder that will use linear layers to expand the latent dimension out. It assumes the latent dimension is a
    linear layer. I.e each record in the batch is a rank-1 tensor

    Args:
        latent_dim: An integer containing the size of the latent dimension this layer will see as input.
        layers: List of integer values defining how many and the size of the linear layers this layer will use to
        expand the latent dimension.
        defaults: An instance of AutoEncoderDefaults, it contains some of the default settings to use.
    """
    def __init__(self, latent_dim, layers: List[int], defaults: AutoEncoderDefaults):
        self._layers = layers
        self._latent_dim = latent_dim
        self._out_size = layers[-1]
        super(LatentLinearDecoder, self).__init__(defaults)

    def init_latent(self) -> _Layer:
        do = self.defaults.get_float('lin_interlayer_drop_out')
        ly = [(i, do) for i in self._layers]
        return LinDropAct(self._latent_dim, ly)

    @property
    def output_size(self) -> int:
        return self._out_size


class LatentVAEDecoder(LatentLinearDecoder):
    """A decoder that will use linear layers to expand the latent dimension out. It assumes the latent dimension
    consists of 2 latent vectors for each entry, describing the latent distribution. One is the vector with averages,
    the other is the vector with standard deviation.

    Args:
        latent_dim: An integer containing the size of the latent dimension this layer will see as input.
        layers: List of integer values defining how many and the size of the linear layers this layer will use to
        expand the latent dimension.
        defaults: An instance of AutoEncoderDefaults, it contains some of the default settings to use.
    """
    def __init__(self, latent_dim, layers: List[int], defaults: AutoEncoderDefaults):
        super(LatentVAEDecoder, self).__init__(latent_dim, layers, defaults)
        self.to_linear = VAELatentToLinear()

    def forward(self, x):
        # Override the standard forward. Need to first re-parameterize.
        mu, s = x
        x = self.to_linear(mu, s)
        x = super(LatentLinearDecoder, self).forward(x)
        return x


class LatentRNNDecoder(_Decoder):
    def __init__(self, node_type: str, latent_size: int, rnn_layers: [List[int]],
                 defaults: AutoEncoderDefaults):
        self._node_type = node_type
        self._rnn_layers = rnn_layers
        self._latent_size = latent_size
        super(LatentRNNDecoder, self).__init__(defaults)

    def init_latent(self) -> _Layer:
        if self._node_type == 'LSTM':
            rnn = LSTMDecoder(self._latent_size, self._rnn_layers)
        else:
            rnn = GRUDecoder(self._latent_size, self._rnn_layers)
        return rnn


class LatentCNNDecoder(_Decoder):
    def __init__(self, latent_dim: int, latent_series_size: int, conv_layers: List[Tuple[int, int]],
                 defaults: AutoEncoderConvolutionalDefaults):
        self._latent_size = latent_dim
        self._latent_series_size = latent_series_size
        self._conv_layers = conv_layers
        super(LatentCNNDecoder, self).__init__(defaults)

    def init_latent(self) -> _Layer:
        do = self.defaults.get_float('dec_conv_dropout')
        bn_int = self.defaults.get_int('dec_conv_bn_interval')
        cnn = ConvolutionalDecoder(
            self._latent_size, self._latent_series_size, self._conv_layers, do, batch_norm_interval=bn_int
        )
        return cnn


class _AutoEncoderModelSingle(_AutoEncoderModel):
    """Internal class. This is a specialized _AutoEncoderModel

    Args:
        encoder: An instance of _EncoderSingle. This will perform the encoder logic
        decoder: An instance of _Decoder. This will perform the decoder logic
        out: A _Layer instance with the logic to convert to the correct output for this loss
        loss_fn: A instance of _LossBase, used to calculate the loss
        defaults: Instance of AutoEncoderDefaults.
    """
    def __init__(self, encoder: _EncoderSingle, decoder: _Decoder, out: _Layer, loss_fn: _LossBase,
                 defaults: AutoEncoderDefaults):
        self._label_index = encoder.tensor_def.learning_categories.index(LEARNING_CATEGORY_LABEL)
        super(_AutoEncoderModelSingle, self).__init__(encoder, decoder, out, loss_fn, defaults)

    @property
    def label_index(self) -> int:
        return self._label_index


class BinaryToBinaryAutoEncoder(_AutoEncoderModelSingle):
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
        defaults: Optional defaults object. If omitted, the AutoEncoderDefaults will be used.
    """
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int],
                 defaults=AutoEncoderDefaults()):
        self._val_layers(layers)
        self._val_only_bin(tensor_def)
        input_size = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY, True))
        encoder = LatentLinearEncoder(tensor_def, latent_dim, [16], defaults)
        decoder = LatentLinearDecoder(latent_dim, [16], defaults)
        out = BinaryOutput(decoder.output_size, input_size)
        super(BinaryToBinaryAutoEncoder, self).__init__(encoder, decoder, out, SingleLabelBCELoss(), defaults)


class CategoricalToBinaryAutoEncoder(_AutoEncoderModelSingle):
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
        defaults: Optional defaults object. If omitted, the AutoEncoderDefaults will be used.
    """
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int],
                 defaults=AutoEncoderDefaults()):
        self._val_layers(layers)
        self._val_only_cat(tensor_def)
        encoder = LatentLinearEncoder(tensor_def, latent_dim, [16], defaults)
        decoder = LatentLinearDecoder(latent_dim, [16], defaults)
        out_size = sum([len(f) + 1 for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)])
        out = BinaryOutput(decoder.output_size, out_size)
        super(CategoricalToBinaryAutoEncoder, self).__init__(
            encoder, decoder, out, MultiLabelBCELoss(tensor_def), defaults
        )


class CategoricalToCategoricalAutoEncoder(_AutoEncoderModelSingle):
    """Auto-encoder which will only take categorical variables as input. (not binary or continuous features)
    and will return categorical output. It will use Embedding layers and linear layers to condense the input to a
    linear latent dimension. The decoder will use linear layers to reconstruct the latent dim back to the size of
    the original input.

    Args:
        tensor_def: The Tensor Definition that will be used
        latent_dim : The size of the latent dimension. As integer value
        layers: A list of integers. Drives the number of layers and their size. For instance [64,32,16] would create a
        neural net with 3 layers of size 64, 32 and 16 respectively. Note that the NN will also have an additional
        input layer (depends on the tensor_def) and an output layer.
        defaults: Optional defaults object. If omitted, the AutoEncoderDefaults will be used.
    """
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int],
                 defaults=AutoEncoderDefaults()):
        self._val_layers(layers)
        self._val_only_cat(tensor_def)
        # self._input_size = len(tensor_def.filter_features(LEARNING_CATEGORY_CATEGORICAL))
        encoder = LatentLinearEncoder(tensor_def, latent_dim, [16], defaults)
        decoder = LatentLinearDecoder(latent_dim, [16], defaults)
        out = CategoricalLogSoftmax1d(tensor_def, decoder.output_size, False)
        super(CategoricalToCategoricalAutoEncoder, self).__init__(
            encoder, decoder, out, MultiLabelNLLLoss(), defaults
        )


class BinaryToBinaryVariationalAutoEncoder(_AutoEncoderModelSingle):
    """Variational Auto-Encoder which will only take binary variables as input. (not categorical or continuous features)
    and will return binary output

    Args:
        tensor_def: The Tensor Definition that will be used
        latent_dim : The size of the latent dimension. As integer value
        layers: A list of integers. Drives the number of layers and their size. For instance [64,32,16] would create a
        neural net with 3 layers of size 64, 32 and 16 respectively. Note that the NN will also have an additional
        input layer (depends on the tensor_def) and an output layer.
        defaults: Optional defaults object. If omitted, the AutoEncoderDefaults will be used.
    """
    def __init__(self, tensor_def: TensorDefinition, latent_dim: int, layers: List[int],
                 defaults=AutoEncoderDefaults()):
        self._val_layers(layers)
        self._val_only_bin(tensor_def)
        input_size = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY, True))
        encoder = LatentVAEEncoder(tensor_def, latent_dim, layers, defaults)
        decoder = LatentVAEDecoder(latent_dim, layers, defaults)
        out = BinaryOutput(decoder.output_size, input_size)
        kl_weight = defaults.get_float('vae_loss_kl_weight')
        super(BinaryToBinaryVariationalAutoEncoder, self).__init__(
            encoder, decoder, out, BinaryVAELoss(kl_weight=kl_weight), defaults
        )

    def forward(self, x):
        # Override the forward of the standard _AutoEncoderModel, we need to keep the mu and s.
        x = self.encoder(x)
        mu, s = x
        x = self.decoder(x)
        x = self.out(x)
        return x, mu, s


class BinaryToBinaryRecurrentAutoEncoder(_AutoEncoderModelSingle):
    def __init__(self, tensor_def: TensorDefinition, node_type: str, layers: List[int], defaults=AutoEncoderDefaults()):
        self._val_layers(layers)
        self._val_only_bin(tensor_def)
        input_size = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY, True))
        encoder = LatentRNNEncoder(tensor_def, node_type, layers, defaults)
        decoder = LatentRNNDecoder(node_type, layers[-1], [input_size] + layers[:-1], defaults)
        out = SigmoidOut()
        super(BinaryToBinaryRecurrentAutoEncoder, self).__init__(encoder, decoder, out, SingleLabelBCELoss(), defaults)


class CategoricalToBinaryRecurrentAutoEncoder(_AutoEncoderModelSingle):
    def __init__(self, tensor_def: TensorDefinition, node_type: str, layers: List[int], defaults=AutoEncoderDefaults()):
        self._val_layers(layers)
        self._val_only_cat(tensor_def)
        input_size = sum([len(f) + 1 for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)])
        encoder = LatentRNNEncoder(tensor_def, node_type, layers, defaults)
        decoder = LatentRNNDecoder(node_type, layers[-1], [input_size] + layers[:-1], defaults)
        out = SigmoidOut()
        super(CategoricalToBinaryRecurrentAutoEncoder, self).__init__(
            encoder, decoder, out, MultiLabelBCELoss(tensor_def), defaults
        )


class CategoricalToCategoricalRecurrentAutoEncoder(_AutoEncoderModelSingle):
    def __init__(self, tensor_def: TensorDefinition, node_type: str, layers: List[int], defaults=AutoEncoderDefaults()):
        self._val_layers(layers)
        self._val_only_cat(tensor_def)
        input_size = sum([len(f) + 1 for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)])
        encoder = LatentRNNEncoder(tensor_def, node_type, layers, defaults)
        decoder = LatentRNNDecoder(node_type, layers[-1], [input_size] + layers[:-1], defaults)
        out = CategoricalLogSoftmax2d(tensor_def, input_size)
        super(CategoricalToCategoricalRecurrentAutoEncoder, self).__init__(
            encoder, decoder, out, MultiLabelNLLLoss2d(), defaults
        )


class BinaryToBinaryConvolutionalAutoEncoder(_AutoEncoderModelSingle):
    def __init__(self, tensor_def: TensorDefinition, conv_layers: [List[Tuple[int, int]]],
                 i_defaults=AutoEncoderConvolutionalDefaults()):
        input_size = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY, True))
        encoder = LatentCNNEncoder(tensor_def, conv_layers, i_defaults)
        # Shift the kernel_sizes by one, add a layer w/input_size and reverse the layer list
        decoder_layers = [(conv_layers[i][0], conv_layers[i+1][1]) for i in range(len(conv_layers)-1)]
        decoder_layers = [(input_size, conv_layers[0][1])] + decoder_layers
        decoder_layers = decoder_layers[::-1]
        decoder = LatentCNNDecoder(
            conv_layers[-1][0], encoder.latent_series_length, decoder_layers,  i_defaults
        )
        out = SigmoidOut()
        super(BinaryToBinaryConvolutionalAutoEncoder, self).__init__(
            encoder, decoder, out, SingleLabelBCELoss(), i_defaults
        )


class CategoricalToBinaryConvolutionalAutoEncoder(_AutoEncoderModelSingle):
    def __init__(self, tensor_def: TensorDefinition, conv_layers: [List[Tuple[int, int]]],
                 i_defaults=AutoEncoderConvolutionalDefaults()):
        input_size = sum([len(f) + 1 for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)])
        encoder = LatentCNNEncoder(tensor_def, conv_layers, i_defaults)
        # Shift the kernel_sizes by one, add a layer w/input_size and reverse the layer list
        decoder_layers = [(conv_layers[i][0], conv_layers[i+1][1]) for i in range(len(conv_layers)-1)]
        decoder_layers = [(input_size, conv_layers[0][1])] + decoder_layers
        decoder_layers = decoder_layers[::-1]
        decoder = LatentCNNDecoder(
            conv_layers[-1][0], encoder.latent_series_length, decoder_layers,  i_defaults
        )
        out = SigmoidOut()
        super(CategoricalToBinaryConvolutionalAutoEncoder, self).__init__(
            encoder, decoder, out, MultiLabelBCELoss(tensor_def), i_defaults
        )


class CategoricalToCategoricalConvolutionalAutoEncoder(_AutoEncoderModelSingle):
    def __init__(self, tensor_def: TensorDefinition, conv_layers: [List[Tuple[int, int]]],
                 i_defaults=AutoEncoderConvolutionalDefaults()):
        input_size = sum([len(f) + 1 for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)])
        encoder = LatentCNNEncoder(tensor_def, conv_layers, i_defaults)
        # Shift the kernel_sizes by one, add a layer w/input_size and reverse the layer list
        decoder_layers = [(conv_layers[i][0], conv_layers[i+1][1]) for i in range(len(conv_layers)-1)]
        decoder_layers = [(input_size, conv_layers[0][1])] + decoder_layers
        decoder_layers = decoder_layers[::-1]
        decoder = LatentCNNDecoder(
            conv_layers[-1][0], encoder.latent_series_length, decoder_layers,  i_defaults
        )
        out = CategoricalLogSoftmax2d(tensor_def, input_size)
        super(CategoricalToCategoricalConvolutionalAutoEncoder, self).__init__(
            encoder, decoder, out, MultiLabelNLLLoss2d(), i_defaults
        )
