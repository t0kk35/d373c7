"""
Module for encoder Models
(c) 2020 d373c7
"""
import logging
import torch
import torch.nn as nn
from .common import ModelDefaults,  _ModelGenerated, _ModelStream
from ..common import _History
from ..layers import CategoricalLogSoftmax1d, CategoricalLogSoftmax2d
from ..layers import ConvolutionalEncoder, ConvolutionalDecoder, LinearEncoder, LinearDecoder
from ..layers import LinearVAEDecoder, LinearVAEEncoder, LinearVAEOut
from ..loss import SingleLabelBCELoss, BinaryVAELoss, MultiLabelNLLLoss, MultiLabelBCELoss
from ..loss import MultiLabelNLLLoss2d
from ...features import TensorDefinition, TensorDefinitionMulti, FeatureCategorical
from ...features import LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_BINARY
from typing import List, Union, Dict

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
        self.inter_layer_drop_out = 0.1
        self.vae_kl_weight = 1.0
        self.convolutional_encoder_drop_out = 0.0
        self.convolutional_encoder_batch_norm_interval = 3
        self.convolutional_decoder_drop_out = 0.1
        self.convolutional_decoder_batch_norm_interval = 2

    def emb_dim(self, minimum: int, maximum: int, dropout: float):
        self.set('emb_min_dim', minimum)
        self.set('emb_max_dim', maximum)
        self.set('emb_dropout', dropout)

    @property
    def inter_layer_drop_out(self) -> float:
        """Defines a value for the inter layer dropout between linear layers. If set, then dropout will be applied
        between linear layers.

        :return: A float value, the dropout aka p value to apply in the nn.Dropout layers.
        """
        return self.get_float('lin_interlayer_drop_out')

    @inter_layer_drop_out.setter
    def inter_layer_drop_out(self, dropout: float):
        """Define a value for the inter layer dropout between linear layers. If set, then dropout will be applied
        between linear layers.

        :param dropout: The dropout aka p value to apply in the nn.Dropout layers.
        """
        self.set('lin_interlayer_drop_out', dropout)

    @property
    def vae_kl_weight(self) -> float:
        return self.get_float('vae_loss_kl_weight')

    @vae_kl_weight.setter
    def vae_kl_weight(self, weight: float):
        self.set('vae_loss_kl_weight', weight)

    @property
    def convolutional_encoder_drop_out(self) -> float:
        """Define a value for the convolutional dropout in the encoder. If set, then dropout will be applied after the
        convolutional layers

        :return: The dropout aka p value to apply in the nn.Dropout layers.
        """
        return self.get_float('conv_enc_body_dropout')

    @convolutional_encoder_drop_out.setter
    def convolutional_encoder_drop_out(self, dropout: float):
        """Define a value for the convolutional dropout in the encoder. If set, then dropout will be applied after the
        convolutional layers

        :param dropout: The dropout aka p value to apply in the nn.Dropout layers.
        """
        self.set('conv_enc_body_dropout', dropout)

    @property
    def convolutional_decoder_drop_out(self) -> float:
        """Define a value for the convolutional dropout in the decoder. If set, then dropout will be applied after the
        convolutional layers

        :return: The dropout aka p value to apply in the nn.Dropout layers.
        """
        return self.get_float('conv_dec_body_dropout')

    @convolutional_decoder_drop_out.setter
    def convolutional_decoder_drop_out(self, dropout: float):
        """Define a value for the convolutional dropout in the decoder. If set, then dropout will be applied after the
        convolutional layers

        :param dropout: The dropout aka p value to apply in the nn.Dropout layers.
        """
        self.set('conv_dec_body_dropout', dropout)

    @property
    def convolutional_encoder_batch_norm_interval(self) -> int:
        """Get the value for the batch norm layer interval for the encoder. This defines every how many convolutional
        layer a batch-norm layer is inserted.

        :return: The interval at which batch-norm layers will be inserted.
        """
        return self.get_int('conv_enc_bn_interval')

    @convolutional_encoder_batch_norm_interval.setter
    def convolutional_encoder_batch_norm_interval(self, bn_interval: int):
        """Define a value batch norm layer interval for the encoder. This defines every how many convolutional layer a
        batch-norm layer is inserted.

        :return: The interval at which batch-norm layers will be inserted.
        """
        self.set('conv_enc_bn_interval', bn_interval)

    @property
    def convolutional_decoder_batch_norm_interval(self) -> int:
        """Get the value for the batch norm layer interval for the decoder. This defines every how many convolutional
        layer a batch-norm layer is inserted.

        :return: The interval at which batch-norm layers will be inserted.
        """
        return self.get_int('conv_dec_bn_interval')

    @convolutional_decoder_batch_norm_interval.setter
    def convolutional_decoder_batch_norm_interval(self, bn_interval: int):
        """Define a value batch norm layer interval for the decoder. This defines every how many convolutional layer a
        batch-norm layer is inserted.

        :return: The interval at which batch-norm layers will be inserted.
        """
        self.set('conv_dec_bn_interval', bn_interval)


# New Generator Class
class GeneratedAutoEncoder(_ModelGenerated):
    def __init__(self, tensor_def: Union[TensorDefinition, TensorDefinitionMulti],
                 a_defaults=AutoEncoderDefaults(), **kwargs):
        tensor_def_m = self.td_to_multi(tensor_def)
        super(GeneratedAutoEncoder, self).__init__(tensor_def_m, a_defaults)

        # Set-up stream per tensor_definition. Note that we CAN get a label Tensor Definition so we can keep the
        # Test and training data consistent. For testing we might need the label.
        label_td = tensor_def_m.label_tensor_definition
        feature_td = [td for td in self._tensor_def.tensor_definitions if td != label_td]
        streams = [_ModelStream(td.name) for td in feature_td]
        self.set_up_heads(a_defaults, feature_td, streams)
        self._add_encoder(streams[0], feature_td[0], kwargs, a_defaults)
        self._add_decoder(streams[0], feature_td[0], kwargs, a_defaults)
        self._add_out(streams[0], feature_td[0], kwargs, a_defaults)
        self.streams = nn.ModuleList(
            [s.create() for s in streams]
        )
        # Set-up label index
        self._label_index = tensor_def_m.label_index

    def _add_encoder(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: Dict,
                     defaults: AutoEncoderDefaults):

        if tensor_def.rank == 2:
            # We're not dealing with a series.
            latent_features = self.get_int_parameter('latent_features', kwargs)
            if self.is_param_defined('linear_vae_layers', kwargs):
                # Set-up a linear Variational Encoder
                linear_layers = self.get_list_parameter('linear_vae_layers', int, kwargs)
                linear_layers = [(i, defaults.inter_layer_drop_out) for i in linear_layers]
                encoder = LinearVAEEncoder(stream.out_size, linear_layers, latent_features, False)
            else:
                linear_layers = self.get_list_parameter('linear_layers', int, kwargs)
                linear_layers = [(i, defaults.inter_layer_drop_out) for i in linear_layers]
                encoder = LinearEncoder(stream.out_size, linear_layers, latent_features, False)
        else:
            # We have a rank 3 tensor, a series
            s_size = [s[1] for s in tensor_def.shapes if len(s) == 3][0]
            convolutional_layers = self.get_list_of_tuples_parameter('convolutional_layers', int, kwargs, None)
            dropout = defaults.convolutional_encoder_drop_out
            bn_interval = defaults.convolutional_encoder_batch_norm_interval
            encoder = ConvolutionalEncoder(stream.out_size, s_size, convolutional_layers, dropout, bn_interval)

        stream.add('encoder', encoder, encoder.output_size)

    def _add_decoder(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: Dict,
                     defaults: AutoEncoderDefaults):
        if tensor_def.rank == 2:
            # We're not dealing with a series.
            if self.is_param_defined('linear_vae_layers', kwargs):
                # Set-up a linear Variational Decoder
                linear_layers = self.get_list_parameter('linear_vae_layers', int, kwargs)
                linear_layers = [(i, defaults.inter_layer_drop_out) for i in linear_layers]
                decoder = LinearVAEDecoder(stream.out_size, linear_layers, False)
            else:
                linear_layers = self.get_list_parameter('linear_layers', int, kwargs)
                linear_layers = [(i, defaults.inter_layer_drop_out) for i in linear_layers]
                decoder = LinearDecoder(stream.out_size, linear_layers, False)
        else:
            # We have a rank 3 tensor, a series
            convolutional_layers = self.get_list_of_tuples_parameter('convolutional_layers', int, kwargs, None)
            dropout = defaults.convolutional_decoder_drop_out
            bn_interval = defaults.convolutional_decoder_batch_norm_interval
            s_size = [s[1] for s in tensor_def.shapes if len(s) == 3][0]
            if LEARNING_CATEGORY_BINARY in tensor_def.learning_categories:
                output_size = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY, True))
            else:
                output_size = sum(
                    [len(f) + 1 for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)]
                )
            decoder = ConvolutionalDecoder(
                stream.out_size, output_size, s_size, convolutional_layers, dropout, bn_interval
            )

        stream.add('decoder', decoder, decoder.output_size)

    def _add_out(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: Dict, defaults: AutoEncoderDefaults):
        # Can be used to force output to binary
        output = self.get_str_parameter('output', kwargs, 'default')

        if output.lower() == 'binary' and LEARNING_CATEGORY_CATEGORICAL in tensor_def.learning_categories:
            # Output is force to binary even though input in categorical
            output_size = sum(
                [len(f) + 1 for f in tensor_def.categorical_features() if isinstance(f, FeatureCategorical)]
            )
            stream.add('out_lin', nn.Linear(stream.out_size, output_size), output_size)
            stream.add('out_sigmoid', nn.Sigmoid(), output_size)
            self.set_loss_fn(MultiLabelBCELoss(tensor_def))
        elif LEARNING_CATEGORY_CATEGORICAL in tensor_def.learning_categories:
            # Output will be categorical. A multi-label (per FeatureIndex), multi-class problem.
            if tensor_def.rank == 2:
                # No Series
                ls = CategoricalLogSoftmax1d(tensor_def, stream.out_size, False)
                stream.add('softmax', ls, ls.output_size)
                self.set_loss_fn(MultiLabelNLLLoss())
            else:
                # We have a Series
                ls = CategoricalLogSoftmax2d(tensor_def, stream.out_size, False)
                stream.add('softmax2d', ls, ls.output_size)
                self.set_loss_fn(MultiLabelNLLLoss2d())
        elif self.is_param_defined('linear_vae_layers', kwargs):
            # Output is binary, using VAE loss
            output_size = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY, True))
            stream.add('out_lin_vae', LinearVAEOut(stream.out_size, output_size), output_size)
            self.set_loss_fn(BinaryVAELoss(kl_weight=defaults.vae_kl_weight))
        else:
            # Output is binary, using BCELoss.
            output_size = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY, True))
            if tensor_def.rank == 2:
                # Add a last linear layer if not a series. For Conv Auto-encoders, the decoder sets the correct out size
                stream.add('out_lin', nn.Linear(stream.out_size, output_size), output_size)
            stream.add('out_sigmoid', nn.Sigmoid(), output_size)
            self.set_loss_fn(SingleLabelBCELoss())

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # For encoders the in and output are the same
        return self.get_x(ds)

    def history(self, *args) -> _History:
        return AutoEncoderHistory(*args)

    def forward(self, x: List[torch.Tensor]):
        # There should only be one stream.
        x = self.streams[0](([x[i] for i in self.head_indexes[0]]))
        return x

    @property
    def label_index(self) -> int:
        return self._label_index
