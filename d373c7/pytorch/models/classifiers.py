"""
Module for classifier Models
(c) 2020 d373c7
"""
import logging
import torch
import torch.nn as nn
from .common import PyTorchModelException, ModelDefaults, _History, _ModelGenerated, _ModelStream
from .encoders import GeneratedAutoEncoder
from ..layers import LSTMBody, ConvolutionalBody1d, AttentionLastEntry, LinearEncoder, TensorDefinitionHead
from ..layers import TransformerBody, TailBinary
from ..loss import SingleLabelBCELoss
from ...features import TensorDefinition, TensorDefinitionMulti
from typing import List, Dict, Union


logger = logging.getLogger(__name__)


class BinaryClassifierHistory(_History):
    loss_key = 'loss'
    acc_key = 'acc'

    def __init__(self, *args):
        dl = self._val_argument(args)
        h = {m: [] for m in [BinaryClassifierHistory.loss_key, BinaryClassifierHistory.acc_key]}
        _History.__init__(self, dl, h)
        self._running_loss = 0
        self._running_correct_cnt = 0
        self._running_count = 0

    @staticmethod
    def _reshape_label(pr: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        if pr.shape == lb.shape:
            return lb
        elif len(pr.shape)-1 == len(lb.shape) and pr.shape[-1] == 1:
            return torch.unsqueeze(lb, dim=len(pr.shape)-1)
        else:
            raise PyTorchModelException(
                f'Incompatible shapes for prediction and label. Got {pr.shape} and {lb.shape}. Can not safely compare'
            )

    def end_step(self, *args):
        BinaryClassifierHistory._val_is_tensor(args[0])
        BinaryClassifierHistory._val_is_tensor_list(args[1])
        BinaryClassifierHistory._val_is_tensor(args[2])
        pr, lb, loss = args[0], args[1][0], args[2]
        lb = BinaryClassifierHistory._reshape_label(pr, lb)
        self._running_loss += loss.item()
        self._running_correct_cnt += torch.sum(torch.eq(torch.ge(pr, 0.5), lb)).item()
        self._running_count += pr.shape[0]
        super(BinaryClassifierHistory, self).end_step(pr, lb, loss)

    def end_epoch(self):
        self._history[BinaryClassifierHistory.loss_key].append(round(self._running_loss/self.steps, 4))
        self._history[BinaryClassifierHistory.acc_key].append(round(self._running_correct_cnt/self.samples, 4))
        self._running_correct_cnt = 0
        self._running_count = 0
        self._running_loss = 0
        super(BinaryClassifierHistory, self).end_epoch()

    def step_stats(self) -> Dict:
        r = {
            BinaryClassifierHistory.loss_key: round(self._running_loss/self.step, 4),
            BinaryClassifierHistory.acc_key: round(self._running_correct_cnt/self._running_count, 4)
        }
        return r

    def early_break(self) -> bool:
        return False


class ClassifierDefaults(ModelDefaults):
    def __init__(self):
        super(ClassifierDefaults, self).__init__()
        self.emb_dim(4, 100, 0.2)
        self.linear_batch_norm = True
        self.inter_layer_drop_out = 0.1
        self.default_series_body = 'recurrent'
        self.attention_drop_out = 0.0
        self.convolutional_dense = True
        self.convolutional_drop_out = 0.1
        self.transformer_positional_logic = 'encoding'
        self.transformer_positional_size = 16
        self.transformer_drop_out = 0.2

    def emb_dim(self, minimum: int, maximum: int, dropout: float):
        self.set('emb_min_dim', minimum)
        self.set('emb_max_dim', maximum)
        self.set('emb_dropout', dropout)

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
    def default_series_body(self) -> str:
        """Defines the default body type for series, which is a tensor of rank 3 (including batch).
         This could be for instance 'recurrent'.

        :return: A string value, the default body type to apply to a rank 3 tensor stream.
        """
        return self.get_str('def_series_body')

    @default_series_body.setter
    def default_series_body(self, def_series_body: str):
        """Defines the default body type for series, which is a tensor of rank 3 (including batch).
         This could be for instance 'recurrent'.

        :param def_series_body: A string value, the default body type to apply to a rank 3 tensor stream.
        """
        self.set('def_series_body', def_series_body)

    @property
    def attention_drop_out(self) -> float:
        """Define a value for the attention dropout. If set, then dropout will be applied after the attention layer.

        :return: The dropout aka p value to apply in the nn.Dropout layers.
        """
        return self.get_float('attn_drop_out')

    @attention_drop_out.setter
    def attention_drop_out(self, dropout: float):
        """Define a value for the attention dropout. If set, then dropout will be applied after the attention layer.

        :param dropout: The dropout aka p value to apply in the nn.Dropout layers.
        """
        self.set('attn_drop_out', dropout)

    @property
    def convolutional_drop_out(self) -> float:
        """Define a value for the attention dropout. If set, then dropout will be applied after the attention layer.

        :return: The dropout aka p value to apply in the nn.Dropout layers.
        """
        return self.get_float('conv_body_dropout')

    @convolutional_drop_out.setter
    def convolutional_drop_out(self, dropout: float):
        """Define a value for the attention dropout. If set, then dropout will be applied after the attention layer.

        :param dropout: The dropout aka p value to apply in the nn.Dropout layers.
        """
        self.set('conv_body_dropout', dropout)

    @property
    def convolutional_dense(self) -> bool:
        """Defines if convolutional bodies are dense. Dense bodies mean that the input to the layer is added to the
        output. It forms a sort of residual connection. The input is concatenated along the features axis. This
        allows the model to work with the input if that turns out to be useful.

        :return: A boolean value, indicating if the input will be added to the output or not.
        """
        return self.get_bool('conv_body_dense')

    @convolutional_dense.setter
    def convolutional_dense(self, dense: bool):
        """Defines if convolutional bodies are dense. Dense bodies mean that the input to the layer is added to the
        output. It forms a sort of residual connection. The input is concatenated along the features axis. This
        allows the model to work with the input if that turns out to be useful.

        :param dense: A boolean value, indicating if the input will be added to the output or not.
        """
        self.set('conv_body_dense', dense)

    @property
    def transformer_positional_logic(self) -> str:
        """Sets which positional logic is used in transformer blocks. 'encoding' : The system will use the encoding,
        'embedding' : The system will use an embedding layer.

        :return: A string value defining which positional logic to use.
        """
        return self.get_str('trans_pos_logic')

    @transformer_positional_logic.setter
    def transformer_positional_logic(self, positional_logic: str):
        """Sets which positional logic is used in transformer blocks. 'encoding' : The system will use the encoding,
         'embedding' : The system will use an embedding layer.

        :param positional_logic: A string value defining which positional logic to use.
        """
        self.set('trans_pos_logic', positional_logic)

    @property
    def transformer_positional_size(self) -> int:
        """Sets the positional size of transformer blocks. The size is the number of elements added to each transaction
        in the series to help the model determine the position of transactions in the series.

        :return: An integer value. The number of elements output by the positional logic
        """
        return self.get_int('trans_pos_size')

    @transformer_positional_size.setter
    def transformer_positional_size(self, positional_size: int):
        """Sets the positional size of transformer blocks. The size is the number of elements added to each transaction
        in the series to help the model determine the position of transactions in the series.

        :param positional_size: An integer value. The number of elements output by the positional logic
        """
        self.set('trans_pos_size', positional_size)

    @property
    def transformer_drop_out(self) -> float:
        """Defines the drop out to apply in the transformer layer

        :return: An float value. The drop out value to apply in transformer layers
        """
        return self.get_float('trans_dropout')

    @transformer_drop_out.setter
    def transformer_drop_out(self, dropout: float):
        """Defines the drop out to apply in the transformer layer

        :param dropout: The drop out value to apply in transformer layers
        """
        self.set('trans_dropout', dropout)


class GeneratedClassifier(_ModelGenerated):
    """Generate a Pytorch classifier model. This class will create a model that fits the input and label definition
    of the TensorDefinition.

    Args:
        tensor_def: A TensorDefinition or TensorDefinitionMulti object describing the various input and output features
        c_defaults: (Optional) ClassifierDefaults object defining the defaults which need to be used.
        kwargs: Various named parameters which can be use to drive the type of classifier and the capacity of the model.
    """
    def __init__(self, tensor_def: Union[TensorDefinition, TensorDefinitionMulti],
                 c_defaults=ClassifierDefaults(), **kwargs):
        tensor_def_m = self.val_is_td_multi(tensor_def)
        super(GeneratedClassifier, self).__init__(tensor_def_m, c_defaults)

        # Set-up stream per tensor_definition
        label_td = self.label_tensor_def(tensor_def_m)
        feature_td = [td for td in self._tensor_def.tensor_definitions if td not in label_td]
        streams = [_ModelStream(td.name) for td in feature_td]

        if self.is_param_defined('transfer_from', kwargs):
            # We're being asked to do transfer learning.
            # TODO we'll need a bunch of validation here.
            om = self.get_gen_model_parameter('transfer_from', kwargs)
            logger.info(f'Transferring from model {om.__class__}')
            # The Source model is an auto-encoder
            if isinstance(om, GeneratedAutoEncoder):
                self.set_up_heads(c_defaults, feature_td, streams)
                # Copy and freeze the TensorDefinitionHead, this should normally be the first item.
                for s, oms in zip(streams, om.streams):
                    for sly in oms:
                        if isinstance(sly, TensorDefinitionHead):
                            src = self.is_tensor_definition_head(sly)
                            trg = self.is_tensor_definition_head(s.layers[0])
                            trg.copy_state_dict(src)
                            trg.freeze()
                            logger.info(f'Transferred and froze TensorDefinitionHead {trg.tensor_definition.name}')
                        elif isinstance(sly, LinearEncoder):
                            # If no linear layers defined then try and copy the encoder linear_layers
                            if not self.is_param_defined('linear_layers', kwargs):
                                linear_layers = sly.layer_definition
                                # Add last layer. Because this is binary, it has to have size of 1.
                                linear_layers.append((1, 0.0))
                                tail = TailBinary(
                                    sum(s.out_size for s in streams), linear_layers, c_defaults.linear_batch_norm
                                )
                                tail_state = tail.state_dict()
                                # Get state of the target layer, remove last item. (popitem)
                                source_state = list(sly.state_dict().values())
                                for i, sk in enumerate(tail_state.keys()):
                                    if i < 2:
                                        tail_state[sk].copy_(source_state[i])
                                # Load target Dict in the target layer.
                                tail.load_state_dict(tail_state)
                                for i, p in enumerate(tail.parameters()):
                                    if i < 2:
                                        p.requires_grad = False
                                logger.info(f'Transferred and froze Linear Encoder layers {sly.layer_definition}')

        else:
            # Set-up a head layer to each stream. This is done in the parent class.
            self.set_up_heads(c_defaults, feature_td, streams)
            # Add Body to each stream.
            for td, s in zip(feature_td, streams):
                self._add_body(s, td, kwargs, c_defaults)
            # Create tail.
            linear_layers = self.get_list_parameter('linear_layers', int, kwargs)
            # Add dropout parameter this will make a list of tuples of (layer_size, dropout)
            linear_layers = [(i, c_defaults.inter_layer_drop_out) for i in linear_layers]
            # Add last layer. Because this is binary, it has to have size of 1.
            linear_layers.append((1, 0.0))
            tail = TailBinary(sum(s.out_size for s in streams), linear_layers, c_defaults.linear_batch_norm)

        # Assume the last entry is the label
        self._y_index = self._x_indexes[-1] + 1
        self.streams = nn.ModuleList(
            [s.create() for s in streams]
        )

        self.tail = tail
        # Last but not least, set-up the loss function
        self.set_loss_fn(SingleLabelBCELoss())

    def _add_body(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: dict, defaults: ClassifierDefaults):
        if tensor_def.rank == 2:
            # No need to add anything to the body, rank goes directly to the tail.
            return
        elif tensor_def.rank == 3:
            # Figure out to which body to use.
            if self.is_param_defined('recurrent_layers', kwargs):
                body_type = 'recurrent'
            elif self.is_param_defined('convolutional_layers', kwargs):
                body_type = 'convolutional'
            elif self.is_param_defined('attention_heads', kwargs):
                body_type = 'transformer'
            else:
                body_type = defaults.default_series_body

            # Set-up the body.
            if body_type.lower() == 'recurrent':
                self._add_recurrent_body(stream, kwargs, defaults)
            elif body_type.lower() == 'convolutional':
                self._add_convolutional_body(stream, tensor_def, kwargs, defaults)
            elif body_type.lower() == 'transformer':
                self._add_transformer_body(stream, tensor_def, kwargs, defaults)
            else:
                raise PyTorchModelException(
                    f'Do not know how to build body of type {body_type}'
                )

    def _add_recurrent_body(self, stream: _ModelStream, kwargs: dict, defaults: ClassifierDefaults):
        attn_heads = self.get_int_parameter('attention_heads', kwargs, 0)
        # attn_do = defaults.attention_drop_out
        rnn_features = self.get_int_parameter(
            'recurrent_features', kwargs, self.closest_power_of_2(int(stream.out_size / 3))
        )
        rnn_layers = self.get_int_parameter('recurrent_layers', kwargs, 1)
        # Add attention if requested
        if attn_heads > 0:
            attn = AttentionLastEntry(stream.out_size, attn_heads, rnn_features)
            stream.add('Attention', attn, attn.output_size)
        # Add main rnn layer
        rnn = LSTMBody(stream.out_size, rnn_features, rnn_layers, True, False)
        stream.add('Recurrent', rnn, rnn.output_size)

    def _add_convolutional_body(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: dict,
                                defaults: ClassifierDefaults):
        s_length = [s[1] for s in tensor_def.shapes if len(s) == 3][0]
        convolutional_layers = self.get_list_of_tuples_parameter('convolutional_layers', int, kwargs, None)
        dropout = defaults.convolutional_drop_out
        dense = defaults.convolutional_dense
        cnn = ConvolutionalBody1d(stream.out_size, s_length, convolutional_layers, dropout, dense)
        stream.add('Convolutional', cnn, cnn.output_size)

    def _add_transformer_body(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: dict,
                              defaults: ClassifierDefaults):
        s_length = [s[1] for s in tensor_def.shapes if len(s) == 3][0]
        attention_head = self.get_int_parameter('attention_heads', kwargs, 1)
        feedforward_size = self.get_int_parameter(
            'feedforward_size', kwargs, self.closest_power_of_2(int(stream.out_size / 3))
        )
        drop_out = defaults.transformer_drop_out
        positional_size = defaults.transformer_positional_size
        positional_logic = defaults.transformer_positional_logic
        trans = TransformerBody(
            stream.out_size,
            s_length,
            positional_size,
            positional_logic,
            attention_head,
            feedforward_size,
            drop_out
        )
        stream.add('Transformer', trans, trans.output_size)

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        return ds[self._y_index: self._y_index+1]

    def history(self, *args) -> _History:
        return BinaryClassifierHistory(*args)

    def forward(self, x: List[torch.Tensor]):
        y = [s([x[i] for i in hi]) for hi, s in zip(self.head_indexes, self.streams)]
        y = self.tail(y)
        return y
