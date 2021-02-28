"""
Module for Next Transaction Prediction Models
(c) 2020 d373c7
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from .common import ModelDefaults, PyTorchModelException,  _ModelGenerated, _ModelStream
from ..common import _History
from ..layers import ConvolutionalNtpBody, LSTMNtpBody, TailBinary, CategoricalLogSoftmax1d
from ..loss import SingleLabelBCELoss, BinaryVAELoss, MultiLabelNLLLoss, MultiLabelBCELoss
from ..loss import MultiLabelNLLLoss2d
from ...features import TensorDefinition, TensorDefinitionMulti, FeatureCategorical
from ...features import LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_BINARY
from typing import Union, List, Dict


logger = logging.getLogger(__name__)


class NtpHistory(_History):
    loss_key = 'loss'

    def __init__(self, *args):
        dl = self._val_argument(args)
        h = {m: [] for m in [NtpHistory.loss_key]}
        _History.__init__(self, dl, h)
        self._running_loss = 0

    def end_step(self, *args):
        NtpHistory._val_is_tensor_list(args[1])
        NtpHistory._val_is_tensor(args[2])
        loss = args[2]
        self._running_loss += loss.item()
        super(NtpHistory, self).end_step(loss)

    def end_epoch(self):
        self._history[NtpHistory.loss_key].append(round(self._running_loss/self.steps, 4))
        self._running_loss = 0
        super(NtpHistory, self).end_epoch()


class NtpDefaults(ModelDefaults):
    def __init__(self):
        super(NtpDefaults, self).__init__()
        self.emb_dim(4, 100, 0.2)
        self.convolutional_drop_out = 0.1
        self.convolutional_batch_norm_interval = 2
        self.inter_layer_drop_out = 0.2
        self.linear_batch_norm = False

    def emb_dim(self, minimum: int, maximum: int, dropout: float):
        self.set('emb_min_dim', minimum)
        self.set('emb_max_dim', maximum)
        self.set('emb_dropout', dropout)

    @property
    def convolutional_drop_out(self) -> float:
        """Define a value for the convolutional dropout. If set, then dropout will be applied after the
        convolutional layers

        :return: The dropout aka p value to apply in the nn.Dropout layers.
        """
        return self.get_float('conv_body_dropout')

    @convolutional_drop_out.setter
    def convolutional_drop_out(self, dropout: float):
        """Define a value for the convolutional dropout. If set, then dropout will be applied after the
        convolutional layers

        :param dropout: The dropout aka p value to apply in the nn.Dropout layers.
        """
        self.set('conv_body_dropout', dropout)

    @property
    def convolutional_batch_norm_interval(self) -> int:
        """Get the value for the batch norm layer interval. This defines every how many convolutional layer a
        batch-norm layer is inserted.

        :return: The interval at which batch-norm layers will be inserted.
        """
        return self.get_int('conv_bn_interval')

    @convolutional_batch_norm_interval.setter
    def convolutional_batch_norm_interval(self, bn_interval: int):
        """Define a value batch norm layer interval. This defines every how many convolutional layer a batch-norm layer
        is inserted.

        :return: The interval at which batch-norm layers will be inserted.
        """
        self.set('conv_bn_interval', bn_interval)

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
    def linear_batch_norm(self) -> bool:
        """Define if a batch norm layer will be added before the final hidden layer.

        :return: bool
        """
        return self.get_bool('lin_batch_norm')

    @linear_batch_norm.setter
    def linear_batch_norm(self, flag: bool):
        """Set if a batch norm layer will be added before the final hidden layer.

        :return: bool
        """
        self.set('lin_batch_norm', flag)


class GeneratedNtp(_ModelGenerated):
    def __init__(self, tensor_def: Union[TensorDefinition, TensorDefinitionMulti],
                 n_defaults=NtpDefaults(), **kwargs):
        tensor_def_m = self.td_to_multi(tensor_def)
        super(GeneratedNtp, self).__init__(tensor_def_m, n_defaults)

        # Set-up stream per tensor_definition. Note that we CAN get a label Tensor Definition so we can keep the
        # Test and training data consistent. For testing we might need the label. For training we obviously don't
        label_td = tensor_def_m.label_tensor_definition
        feature_td = [td for td in self._tensor_def.tensor_definitions if td != label_td]
        streams = [_ModelStream(td.name) for td in feature_td]
        self.set_up_heads(n_defaults, feature_td, streams)
        self._add_body(streams[0], feature_td[0], kwargs, n_defaults)
        self.streams = nn.ModuleList(
            [s.create() for s in streams]
        )
        self.tail = None
        self._set_tail(streams, feature_td[0], kwargs, n_defaults)
        # Set-up label index. We only need this for testing. During testing we might want know if it was fraud or not.
        self._label_index = tensor_def_m.label_index

    def _add_body(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: Dict, defaults: NtpDefaults):
        if tensor_def.rank == 3:
            if self.is_param_defined('convolutional_layers', kwargs):
                self._add_convolutional_body(stream, tensor_def, kwargs, defaults)
            elif self.is_param_defined('recurrent_features', kwargs):
                self._add_recurrent_body(stream, tensor_def, kwargs, defaults)
        else:
            raise PyTorchModelException(
                f'Unexpected Error. Rank of tensor needs to be 3 for NTP style bodies. Should have been caught earlier'
            )

    def _add_convolutional_body(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: Dict,
                                defaults: NtpDefaults):
        # Size -1 because the last payment will be the target
        s_size = [s[1] for s in tensor_def.shapes if len(s) == 3][0] - 1
        convolutional_layers = self.get_list_of_tuples_parameter('convolutional_layers', int, kwargs)
        dropout = defaults.convolutional_drop_out
        bn_interval = defaults.convolutional_batch_norm_interval
        body = ConvolutionalNtpBody(stream.out_size, s_size, convolutional_layers, dropout, bn_interval)
        stream.add('ntp_conv_body', body, body.output_size)

    def _add_recurrent_body(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: Dict,
                            defaults: NtpDefaults):
        recurrent_features = self.get_int_parameter('recurrent_features', kwargs)
        recurrent_layers = self.get_int_parameter('recurrent_layers', kwargs, default=1)
        body = LSTMNtpBody(stream.out_size, recurrent_features, recurrent_layers)
        stream.add('ntp_conv_body', body, body.output_size)

    def _set_tail(self, streams: List[_ModelStream], tensor_def: TensorDefinition, kwargs: Dict, defaults: NtpDefaults):
        linear_layers = self.get_list_parameter('linear_layers', int, kwargs, [])
        # Add dropout parameter this will make a list of tuples of (layer_size, dropout)
        linear_layers = [(i, defaults.inter_layer_drop_out) for i in linear_layers]

        if LEARNING_CATEGORY_BINARY in tensor_def.learning_categories:
            output_size = len(tensor_def.filter_features(LEARNING_CATEGORY_BINARY, True))
            # Add Last Linear Layer to the size of the target output which is the same size as the input
            linear_layers.append((output_size, 0.0))
            self.tail = TailBinary(sum(s.out_size for s in streams), linear_layers, defaults.linear_batch_norm)
            self.set_loss_fn(SingleLabelBCELoss())
        elif LEARNING_CATEGORY_CATEGORICAL in tensor_def.learning_categories:
            output_size = sum([s.out_size for s in streams])
            # The output of the tail will be the sum of sizes of the streams
            self.tail = CategoricalLogSoftmax1d(tensor_def, output_size, False)
            self.set_loss_fn(MultiLabelNLLLoss())
        else:
            raise PyTorchModelException(
                f'Can not create out layer, there are no binary or categorical layers.'
            )

    def history(self, *args) -> _History:
        return NtpHistory(*args)

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # Remove the last entry from the series.
        return [d[:, :-1, :] for d in _ModelGenerated.get_x(self, ds)]

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # Return the last entry ONLY of each series. And remove the series dim, it will have size 1.
        x = [torch.squeeze(d[:, -1:, :], dim=1) for d in _ModelGenerated.get_x(self, ds)]
        return x

    def forward(self, x: List[torch.Tensor]):
        # There should only be one stream.
        y = [s([x[i] for i in hi]) for hi, s in zip(self.head_indexes, self.streams)]
        x = self.tail(y)
        return x

    @property
    def label_index(self) -> int:
        return self._label_index
