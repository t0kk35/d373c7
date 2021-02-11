"""
Module for classifier Models
(c) 2020 d373c7
"""
import logging
import torch
import torch.nn as nn
from .common import _LossBase, PyTorchModelException, ModelDefaults, _History, _Model, _ModelGenerated, _ModelStream
from ..layers import SingleClassBinaryOutput, LinDropAct, TensorDefinitionHead, TensorDefinitionHeadMulti
from ..layers import LSTMBody, GRUBody, BodyMulti, BodySequential, ConvolutionalBody1d, AttentionLastEntry
from ..layers import TransformerBody, TailBinary
# noinspection PyProtectedMember
from ..layers.common import Layer
from ..optimizer import _Optimizer, AdamWOptimizer
from ..loss import SingleLabelBCELoss
from ..data import NumpyListDataSetMulti
from ...features import TensorDefinition, TensorDefinitionMulti, LEARNING_CATEGORY_LABEL, FeatureLabelBinary
from ...features import FeatureCategorical
from typing import List, Dict, Union, Tuple


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
        self.set_linear_batch_norm(True)
        self.set('lin_interlayer_drop_out', 0.1)
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

    # TODO Remove
    def set_linear_batch_norm(self, flag: bool) -> None:
        """Define if a batch norm layer will be added before the final hidden layer.

        :return: None
        """
        self.set('lin_batch_norm', flag)

    # TODO Remove
    def set_inter_layer_drop_out(self, dropout: float) -> None:
        """Sets the interlayer drop out parameter. Interlayer dropout is the drop out between linear layers.

        :param dropout: Float number. Defined the amount of dropout to apply between linear layers.
        :return: None
        """
        self.set('lin_interlayer_drop_out', dropout)

    def set_embedding_drop_out(self, dropout: float) -> None:
        """Sets the drop out parameter for the embedding layers. That is the drop out applied to the output of
        embedding layers.

        :param dropout: Float number. Defined the amount of dropout to apply between linear layers.
        :return: None
        """
        self.set('emb_dropout', dropout)


# New Generator Class
class GeneratedClassifier(_ModelGenerated):
    def __init__(self, tensor_def: Union[TensorDefinition, TensorDefinitionMulti],
                 c_defaults=ClassifierDefaults(), **kwargs):
        loss_fn = SingleLabelBCELoss()
        super(GeneratedClassifier, self).__init__(c_defaults, loss_fn)
        self._tensor_def = self.val_is_td_multi(tensor_def)
        self.val_td_is_inference_ready(self._tensor_def)
        # Set-up stream per tensor_definition
        label_td = self.label_tensor_def(self._tensor_def)
        feature_td = [td for td in self._tensor_def.tensor_definitions if td not in label_td]
        streams = [_ModelStream(td.name) for td in feature_td]
        self._x_indexes = []
        self._head_indexes = []
        x_offset = 0
        # Add a head layer to each stream.
        for td, s in zip(feature_td, streams):
            head = self.create_head(td, c_defaults)
            self._x_indexes.extend([x+x_offset for x in head.x_indexes])
            self._head_indexes.append([x+x_offset for x in head.x_indexes])
            x_offset = self._x_indexes[-1] + 1
            s.add(td.name, head, head.output_size)
        # Assume the last entry is the label
        self._y_index = self._x_indexes[-1] + 1
        # Add Body to each stream.
        for td, s in zip(feature_td, streams):
            self.add_body(s, td, kwargs, c_defaults)
        self.streams = nn.ModuleList(
            [s.create() for s in streams]
        )
        # Create tail
        linear_layers = self.get_list_parameter('linear_layers', int, kwargs)
        # Add dropout parameter this will make a list of tuples of (layer_size, dropout)
        linear_layers = [(i, c_defaults.inter_layer_drop_out) for i in linear_layers]
        self.tail = TailBinary(sum(s.out_size for s in streams), linear_layers, c_defaults.linear_batch_norm)

    def add_body(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: dict, defaults: ClassifierDefaults):
        if tensor_def.rank == 2:
            # No need to add anything to the body, rank goes directly to the tail.
            return
        elif tensor_def.rank == 3:
            if self.is_param_defined('recurrent_layers', kwargs):
                body_type = 'recurrent'
            elif self.is_param_defined('convolutional_layers', kwargs):
                body_type = 'convolutional'
            elif self.is_param_defined('attention_heads', kwargs):
                body_type = 'transformer'
            else:
                body_type = defaults.default_series_body

            if body_type.lower() == 'recurrent':
                self.add_recurrent_body(stream, kwargs, defaults)
            elif body_type.lower() == 'convolutional':
                self.add_convolutional_body(stream, tensor_def, kwargs, defaults)
            elif body_type.lower() == 'transformer':
                self.add_transformer_body(stream, tensor_def, kwargs, defaults)
            else:
                raise PyTorchModelException(
                    f'Do not know how to build body of type {body_type}'
                )

    def add_recurrent_body(self, stream: _ModelStream, kwargs: dict, defaults: ClassifierDefaults):
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

    def add_convolutional_body(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: dict,
                               defaults: ClassifierDefaults):
        s_length = [s[1] for s in tensor_def.shapes if len(s) == 3][0]
        convolutional_layers = self.get_list_of_tuples_parameter('convolutional_layers', int, kwargs, None)
        dropout = defaults.convolutional_drop_out
        dense = defaults.convolutional_dense
        cnn = ConvolutionalBody1d(stream.out_size, s_length, convolutional_layers, dropout, dense)
        stream.add('Convolutional', cnn, cnn.output_size)

    def add_transformer_body(self, stream: _ModelStream, tensor_def: TensorDefinition, kwargs: dict,
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
    def tensor_definition(self) -> TensorDefinitionMulti:
        """Property Method that returns the TensorDefinitionMulti object used to create this Generated Model

        :return: A TensorDefinitionMulti object which was used to create this model.
        """
        return self._tensor_def

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

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        ds = [ds[x] for x in self._x_indexes]
        return ds

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        return ds[self._y_index: self._y_index+1]

    def history(self, *args) -> _History:
        return BinaryClassifierHistory(*args)

    def forward(self, x: List[torch.Tensor]):
        y = [s([x[i] for i in hi]) for hi, s in zip(self._head_indexes, self.streams)]
        y = self.tail(y)
        return y
# > End new Generator class


class _BinaryClassifier(_Model):
    @staticmethod
    def _val_has_lc_label(tensor_def: TensorDefinition):
        if LEARNING_CATEGORY_LABEL not in tensor_def.learning_categories:
            raise PyTorchModelException(
                f'Tensor Definition <{tensor_def.name}> does not have a label learning category. '
                f'Can not build a classifier without a label. Please the .set_label(xyz) on the tensor definition'
            )

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

    @staticmethod
    def _val_label(tensor_def: TensorDefinition):
        if not len(tensor_def.label_features()) == 1:
            raise PyTorchModelException(
                f'The Tensor Definition of a binary model must contain exactly one LEARNING_CATEGORY_LABEL feature.' +
                f'Got {len(tensor_def.label_features())} of them'
            )
        if not isinstance(tensor_def.label_features()[0], FeatureLabelBinary):
            raise PyTorchModelException(
                f'The LEARNING_CATEGORY_LABEL feature of a Tensor Definition must be of type' +
                f'{FeatureLabelBinary.__class__.__name__} '
            )

    @staticmethod
    def _val_is_multi_head(head) -> TensorDefinitionHeadMulti:
        if not isinstance(head, TensorDefinitionHeadMulti):
            raise PyTorchModelException(
                f'Internal exception. The Head should have been a TensorDefinitionHeadMulti. Got {type(head)}'
            )
        else:
            return head

    def __init__(self, tensor_def: TensorDefinition, layers: List[int], defaults: ClassifierDefaults):
        super(_BinaryClassifier, self).__init__(defaults)
        self._val_layers(layers)
        self._val_has_lc_label(tensor_def)
        self._val_label(tensor_def)
        do = self.defaults.get_float('lin_interlayer_drop_out')
        bn = self.defaults.get_bool('lin_batch_norm')
        ly = [(i, do) for i in layers]
        self.head = self.init_head()
        self.body = self.init_body()
        size_after_body = self.head.output_size if self.body is None else self.body.output_size
        self.linear = LinDropAct(size_after_body, ly)
        self.bn = nn.BatchNorm1d(self.linear.output_size) if bn else None
        self.out = SingleClassBinaryOutput(self.linear.output_size)

    def get_x(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.head.get_x(ds)

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # Return the label with Learning Category 'LEARNING_CATEGORY_LABEL'
        return ds[self.label_index: self.label_index+1]

    def optimizer(self, lr=None, wd=None) -> _Optimizer:
        return AdamWOptimizer(self, lr, wd)

    def loss_fn(self) -> _LossBase:
        return SingleLabelBCELoss()

    def history(self, *args) -> _History:
        return BinaryClassifierHistory(*args)

    def forward(self, x):
        x = self.head(x)
        if self.body is not None:
            x = self.body(x)
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.out(x)
        return x

    def init_head(self) -> Layer:
        raise NotImplemented('Should be implemented by Children')

    def init_body(self) -> Union[Layer, None]:
        raise NotImplemented('Should be implemented by Children')

    def embedding_weights(self, feature: FeatureCategorical, as_numpy: bool = False):
        w = self.head.embedding_weight(feature)
        if as_numpy:
            w = w.cpu().detach().numpy()
        return w


class _BinaryClassifierSingle(_BinaryClassifier):
    """Internal class to bundle some of the logic which is similar for classifiers that with a single head.

    Args:
        defaults: An instance of ClassifierDefaults
    """
    def __init__(self, tensor_def: TensorDefinition, layers: List[int], defaults: ClassifierDefaults):
        self._t_def = tensor_def
        self._label_index = self._t_def.learning_categories.index(LEARNING_CATEGORY_LABEL)
        super(_BinaryClassifierSingle, self).__init__(tensor_def, layers, defaults)

    def init_head(self) -> Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHead(self._t_def, do, mn, mx)

    @property
    def tensor_def(self) -> TensorDefinition:
        return self._t_def

    @property
    def label_index(self) -> int:
        return self._label_index


class _BinaryClassifierMulti(_BinaryClassifier):
    """Internal class to bundle some of the logic which is similar for classifiers that with a multi head.

    Args:
        defaults: An instance of ClassifierDefaults
    """
    def __init__(self, tensor_def: TensorDefinitionMulti, layers: List[int], defaults: ClassifierDefaults):
        self._t_def_m = tensor_def
        self._label_index = NumpyListDataSetMulti.label_index(self._t_def_m)
        super(_BinaryClassifierMulti, self).__init__(tensor_def.label_tensor_definition, layers, defaults)

    def init_head(self) -> Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHeadMulti(self._t_def_m, do, mn, mx)

    @property
    def tensor_def_multi(self) -> TensorDefinitionMulti:
        return self._t_def_m

    @property
    def label_index(self) -> int:
        return self._label_index


class FeedForwardFraudClassifier(_BinaryClassifierSingle):
    """Create a FeedForward Fraud classifier neural net. This model only uses Linear (Feedforward) layers. It is the
    simplest form of Neural Net. The input will be run through a set of Linear layers and ends with a layer of size 1.
    This one number will be an interval between 0-1 and indicate how likely this is fraud. The model uses
    BinaryCrossEntropy loss.

    Args:
        tensor_def: The Tensor Definition that will be used
        layers: A list of integers. Drives the number of layers and their size. For instance [64,32,16] would create a
        neural net with 3 layers of size 64, 32 and 16 respectively. Note that the NN will also have an additional
        input layer (depends on the tensor_def) and an output layer.
        defaults: Optional defaults object. If omitted, the ClassifierDefaults will be used.

    """
    def __init__(self, tensor_def: TensorDefinition, layers: List[int], defaults=ClassifierDefaults()):
        super(FeedForwardFraudClassifier, self).__init__(tensor_def, layers, defaults)

    def init_body(self) -> Union[Layer, None]:
        return None


class FeedForwardFraudClassifierMulti(_BinaryClassifierMulti):
    """Create a FeedForward Fraud classifier neural net. This model only uses Linear (Feedforward) layers. It is the
    simplest form of Neural Net. The input will be run through a set of Linear layers and ends with a layer of size 1.
    This one number will be an interval between 0-1 and indicate how likely this is fraud. The model uses
    BinaryCrossEntropy loss. This version has multi-head support, it can be fed a TensorDefinitionMulti.

    Args:
        tensor_def: The Tensor Definition Multi that will be used
        layers: A list of integers. Drives the number of layers and their size. For instance [64,32,16] would create a
        neural net with 3 layers of size 64, 32 and 16 respectively. Note that the NN will also have an additional
        input layer (depends on the tensor_def) and an output layer.
        defaults: Optional defaults object. If omitted, the ClassifierDefaults will be used.

    """
    def __init__(self, tensor_def: TensorDefinitionMulti, layers: List[int], defaults=ClassifierDefaults()):
        super(FeedForwardFraudClassifierMulti, self).__init__(tensor_def, layers, defaults)

    def init_body(self) -> Union[Layer, None]:
        # Make None layer. This will just concat the layers in the BodyMulti
        lys = [None for _ in self.tensor_def_multi.tensor_definitions]
        ly = BodyMulti(self.head, lys)
        return ly


class RecurrentClassifierDefaults(ClassifierDefaults):
    def __init__(self):
        super(RecurrentClassifierDefaults, self).__init__()
        self.set_dense(True)
        self.set_recurrent_batch_norm(False)
        self.set_attention_heads(0)
        self.set_attention_dropout(0.1)

    def set_dense(self, dense: bool):
        self.set('rec_body_dense', dense)

    def set_recurrent_batch_norm(self, flag: bool) -> None:
        """Define if a batch norm layer will be added before the final hidden layer.

        :return: None
        """
        self.set('rec_batch_norm', flag)

    def set_attention_heads(self, heads: int):
        """Define the number of attention heads to be used in the self-attention layers. 0 means self-attention is
        switched off.

        :return: None
        """
        self.set('rec_attention_heads', heads)

    def set_attention_dropout(self, dropout: float):
        """Define the dropout to be used in the attention layers.

        :return: None
        """
        self.set('rec_attention_dropout', dropout)


class _RecurrentFraudClassifierHelper:
    _node_types = ['LSTM', 'GRU']

    @staticmethod
    def val_node_type(node_type: str):
        if node_type not in _RecurrentFraudClassifierHelper._node_types:
            raise PyTorchModelException(
                f'Node type must be one of <{_RecurrentFraudClassifierHelper._node_types}>. Got <{node_type}>'
            )

    @staticmethod
    def make_rnn_layer(defaults: ModelDefaults, node_type: str, input_size: int, recurrent_features: int,
                       recurrent_layers: int) -> Layer:
        dense = defaults.get_bool('rec_body_dense')
        batch_norm = defaults.get_bool('rec_batch_norm')
        if node_type == 'LSTM':
            rnn = LSTMBody(input_size, recurrent_features, recurrent_layers, dense, batch_norm)
        else:
            rnn = GRUBody(input_size, recurrent_features, recurrent_layers, dense, batch_norm)
        return rnn


class RecurrentFraudClassifier(_BinaryClassifierSingle):
    def __init__(self, tensor_def: TensorDefinition, node_type: str, recurrent_features: int, recurrent_layers: int,
                 linear_layers: List[int], defaults=RecurrentClassifierDefaults()):
        _RecurrentFraudClassifierHelper.val_node_type(node_type)
        self._node_type = node_type
        self._recurrent_features = recurrent_features
        self._recurrent_layers = recurrent_layers
        super(RecurrentFraudClassifier, self).__init__(tensor_def, linear_layers, defaults)

    def init_body(self) -> Union[Layer, None]:
        attn_heads = self.defaults.get_int('rec_attention_heads')
        attn_do = self.defaults.get_float('rec_attention_dropout')
        rnn = _RecurrentFraudClassifierHelper.make_rnn_layer(
            self.defaults, self._node_type, self.head.output_size, self._recurrent_features, self._recurrent_layers
        )
        # Add attention if needed
        if attn_heads != 0:
            # attn = Attention(self.head.output_size, attn_heads, attn_do)
            attn = AttentionLastEntry(self.head.output_size, 1, 32)
            return BodySequential(self.head.output_size, [attn, rnn])
        else:
            return rnn


class RecurrentFraudClassifierMulti(_BinaryClassifierMulti):
    def __init__(self, tensor_def: TensorDefinitionMulti, node_type: str, recurrent_features: int,
                 recurrent_layers: int, linear_layers: List[int], defaults=RecurrentClassifierDefaults()):
        _RecurrentFraudClassifierHelper.val_node_type(node_type)
        self._node_type = node_type
        self._recurrent_features = recurrent_features
        self._recurrent_layers = recurrent_layers
        super(RecurrentFraudClassifierMulti, self).__init__(tensor_def, linear_layers, defaults)

    def init_body(self) -> Layer:
        attn_heads = self.defaults.get_int('rec_attention_heads')
        attn_do = self.defaults.get_float('rec_attention_dropout')
        head = RecurrentFraudClassifierMulti._val_is_multi_head(self.head)
        lys = []
        for td, hs in zip(self._t_def_m.tensor_definitions, [h.output_size for h in head.heads]):
            # Only Rank 3 Tensor Definition are Series
            if td.rank == 3:
                # Add attention if needed
                if attn_heads != 0:
                    # attn = Attention(hs, attn_heads, attn_do)
                    attn = AttentionLastEntry(hs, 1, 32)
                    rnn = _RecurrentFraudClassifierHelper.make_rnn_layer(
                        self.defaults, self._node_type, attn.output_size, self._recurrent_features,
                        self._recurrent_layers
                    )
                    seq = BodySequential(attn.output_size, [attn, rnn])
                    lys.append(seq)
                else:
                    rnn = _RecurrentFraudClassifierHelper.make_rnn_layer(
                        self.defaults, self._node_type, hs, self._recurrent_features,
                        self._recurrent_layers
                    )
                    lys.append(rnn)
            else:
                lys.append(None)
        return BodyMulti(self.head, lys)


class ConvolutionalClassifierDefaults(ClassifierDefaults):
    def __init__(self):
        super(ConvolutionalClassifierDefaults, self).__init__()
        self.set_dense(True)
        self.set_conv_dropout(0.1)

    def set_dense(self, dense: bool):
        self.set('conv_body_dense', dense)

    def set_conv_dropout(self, dropout: float):
        self.set('conv_dropout', dropout)


class ConvolutionalFraudClassifier(_BinaryClassifier):
    def __init__(self, tensor_def: TensorDefinition, conv_layers: List[Tuple[int, int, int]],
                 linear_layers: List[int], defaults=ConvolutionalClassifierDefaults()):
        self._t_def = tensor_def
        self._conv_layers = conv_layers
        self._label_index = self._t_def.learning_categories.index(LEARNING_CATEGORY_LABEL)
        super(ConvolutionalFraudClassifier, self).__init__(tensor_def, linear_layers, defaults)

    def init_head(self) -> Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHead(self._t_def, do, mn, mx)

    def init_body(self) -> Union[Layer, None]:
        do = self.defaults.get_float('conv_dropout')
        dense = self.defaults.get_bool('conv_body_dense')
        # Get the length of the series, it is the second dimension of the shape.
        s_length = [s[1] for s in self._t_def.shapes if len(s) == 3][0]
        return ConvolutionalBody1d(self.head.output_size, s_length, self._conv_layers, do, dense)

    @property
    def label_index(self) -> int:
        return self._label_index


class ConvolutionalFraudClassifierMulti(ConvolutionalFraudClassifier):
    def __init__(self, tensor_def: TensorDefinitionMulti, conv_layers: List[Tuple[int, int, int]],
                 linear_layers: List[int]):
        self._t_def_m = tensor_def
        super(ConvolutionalFraudClassifierMulti, self).__init__(
            tensor_def.label_tensor_definition, conv_layers, linear_layers
        )
        self._label_index = NumpyListDataSetMulti.label_index(tensor_def)

    def init_head(self) -> Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHeadMulti(self._t_def_m, do, mn, mx)

    def init_body(self) -> Union[Layer, None]:
        do = self.defaults.get_float('conv_dropout')
        dense = self.defaults.get_bool('conv_body_dense')
        head = self._val_is_multi_head(self.head)
        lys = []
        for td, hs in zip(self._t_def_m.tensor_definitions, [h.output_size for h in head.heads]):
            # Only Rank 3 Tensor Definition are Series
            if td.rank == 3:
                # Get the length of the series, it is the second dimension of the shape.
                s_length = [s[1] for s in td.shapes if len(s) == 3][0]
                lys.append(ConvolutionalBody1d(hs, s_length, self._conv_layers, do, dense))
            else:
                lys.append(None)
        return BodyMulti(self.head, lys)

    @property
    def label_index(self) -> int:
        return self._label_index


class TransformerClassifierDefaults(ClassifierDefaults):
    def __init__(self):
        super(TransformerClassifierDefaults, self).__init__()
        self.set_trans_dropout(0.2)
        self.set_trans_positional_size(16)
        self.set_trans_positional_logic('embedding')

    def set_trans_dropout(self, dropout: float):
        self.set('trans_dropout', dropout)

    def set_trans_positional_size(self, size: int):
        self.set('trans_pos_size', size)

    def set_trans_positional_logic(self, logic: str):
        self.set('trans_pos_logic', logic)


class TransformerFraudClassifier(_BinaryClassifierSingle):
    def __init__(self, tensor_def: TensorDefinition, attention_heads: int, feedforward_size: int,
                 linear_layers: List[int], defaults=TransformerClassifierDefaults()):
        self._attention_heads = attention_heads
        self._feedforward_size = feedforward_size
        super(TransformerFraudClassifier, self).__init__(tensor_def, linear_layers, defaults)

    def init_body(self) -> Union[Layer, None]:
        do = self.defaults.get_float('trans_dropout')
        ps = self.defaults.get_int('trans_pos_size')
        pl = self.defaults.get_str('trans_pos_logic')
        series_size = [s for s in self.tensor_def.shapes if len(s) == 3][0][1]
        b = TransformerBody(
            self.head.output_size, series_size, ps, self._attention_heads, self._feedforward_size, do, pl
        )
        return b


class TransformerFraudClassifierMulti(TransformerFraudClassifier):
    def __init__(self, tensor_def: TensorDefinitionMulti, attention_heads: int, feedforward_size: int,
                 linear_layers: List[int], defaults=TransformerClassifierDefaults()):
        self._t_def_m = tensor_def
        super(TransformerFraudClassifierMulti, self).__init__(
            tensor_def.label_tensor_definition, attention_heads, feedforward_size, linear_layers, defaults
        )
        self._label_index = NumpyListDataSetMulti.label_index(tensor_def)

    def init_head(self) -> Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHeadMulti(self._t_def_m, do, mn, mx)

    def init_body(self) -> Union[Layer, None]:
        do = self.defaults.get_float('trans_dropout')
        ps = self.defaults.get_int('trans_pos_size')
        pl = self.defaults.get_str('trans_pos_logic')
        head = self._val_is_multi_head(self.head)
        lys = []
        for td, hs in zip(self._t_def_m.tensor_definitions, [h.output_size for h in head.heads]):
            # Only Rank 3 Tensor Definition are Series
            if td.rank == 3:
                # Get the length of the series, it is the second dimension of the shape.
                s_length = [s[1] for s in td.shapes if len(s) == 3][0]
                lys.append(TransformerBody(hs, s_length, ps, self._attention_heads, self._feedforward_size, do, pl))
            else:
                lys.append(None)
        return BodyMulti(self.head, lys)

    @property
    def label_index(self) -> int:
        return self._label_index
