"""
Module for classifier Models
(c) 2020 d373c7
"""
import logging
import torch
import torch.nn as nn
from .common import _LossBase, PyTorchModelException, ModelDefaults, _History, _Model
from ..layers import SingleClassBinaryOutput, LinDropAct, TensorDefinitionHead, TensorDefinitionHeadMulti, Attention
from ..layers import LSTMBody, GRUBody, BodyMulti, BodySequential, ConvolutionalBody1d, AttentionLastEntry
# noinspection PyProtectedMember
from ..layers.common import _Layer
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

    def emb_dim(self, minimum: int, maximum: int, dropout: float):
        self.set('emb_min_dim', minimum)
        self.set('emb_max_dim', maximum)
        self.set('emb_dropout', dropout)

    def set_linear_batch_norm(self, flag: bool) -> None:
        """Define if a batch norm layer will be added before the final hidden layer.

        :return: None
        """
        self.set('lin_batch_norm', flag)

    def set_inter_layer_drop_out(self, dropout: float) -> None:
        """Sets the interlayer drop out parameter. Interlayer dropout is the drop out between linear layers.

        :param dropout: Float number. Defined the amount of dropout to apply between linear layers.
        :return: None
        """
        self.set('lin_interlayer_drop_out', dropout)


class BinaryClassifier(_Model):
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
        super(BinaryClassifier, self).__init__(defaults)
        self._val_layers(layers)
        self._tensor_def = tensor_def
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

    @property
    def label_index(self) -> int:
        raise NotImplemented(f'Class label index should be implemented by children')

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

    def init_head(self) -> _Layer:
        raise NotImplemented('Should be implemented by Children')

    def init_body(self) -> Union[_Layer, None]:
        raise NotImplemented('Should be implemented by Children')

    def embedding_weights(self, feature: FeatureCategorical, as_numpy: bool = False):
        w = self.head.embedding_weight(feature)
        if as_numpy:
            w = w.cpu().detach().numpy()
        return w


class FeedForwardFraudClassifier(BinaryClassifier):
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
        self._t_def = tensor_def
        self._label_index = self._t_def.learning_categories.index(LEARNING_CATEGORY_LABEL)
        super(FeedForwardFraudClassifier, self).__init__(tensor_def, layers, defaults)

    def init_head(self) -> _Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHead(self._t_def, do, mn, mx)

    def init_body(self) -> Union[_Layer, None]:
        return None

    @property
    def label_index(self) -> int:
        return self._label_index


class FeedForwardFraudClassifierMulti(BinaryClassifier):
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
        self._t_def_m = tensor_def
        super(FeedForwardFraudClassifierMulti, self).__init__(tensor_def.label_tensor_definition, layers, defaults)
        self._label_index = NumpyListDataSetMulti.label_index(tensor_def)

    def init_head(self) -> _Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHeadMulti(self._t_def_m, do, mn, mx)

    def init_body(self) -> Union[_Layer, None]:
        # Make None layer. This will just concat the layers in the BodyMulti
        lys = [None for _ in self._t_def_m.tensor_definitions]
        ly = BodyMulti(self.head, lys)
        return ly

    @property
    def label_index(self) -> int:
        return self._label_index


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


class RecurrentFraudClassifier(BinaryClassifier):
    _node_types = ['LSTM', 'GRU']

    def _val_node_type(self, node_type: str):
        if node_type not in self._node_types:
            raise PyTorchModelException(
                f'Node type must be one of <{self._node_types}>. Got <{node_type}>'
            )

    def __init__(self, tensor_def: TensorDefinition, node_type: str, recurrent_features: int, recurrent_layers: int,
                 linear_layers: List[int], defaults=RecurrentClassifierDefaults()):
        self._val_node_type(node_type)
        self._t_def = tensor_def
        self._node_type = node_type
        self._recurrent_features = recurrent_features
        self._recurrent_layers = recurrent_layers
        self._dense = True
        self._label_index = self._t_def.learning_categories.index(LEARNING_CATEGORY_LABEL)
        super(RecurrentFraudClassifier, self).__init__(tensor_def, linear_layers, defaults)

    def init_head(self) -> _Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHead(self._t_def, do, mn, mx)

    def init_body(self) -> Union[_Layer, None]:
        dense = self.defaults.get_bool('rec_body_dense')
        batch_norm = self.defaults.get_bool('rec_batch_norm')
        attn_heads = self.defaults.get_int('rec_attention_heads')
        attn_do = self.defaults.get_float('rec_attention_dropout')
        if self._node_type == 'LSTM':
            rnn = LSTMBody(self.head.output_size, self._recurrent_features, self._recurrent_layers, dense, batch_norm)
        else:
            rnn = GRUBody(self.head.output_size, self._recurrent_features, self._recurrent_layers, dense, batch_norm)
        # Add attention if needed
        if attn_heads != 0:
            # attn = Attention(self.head.output_size, attn_heads, attn_do)
            attn = AttentionLastEntry(0, 1, 32)
            return BodySequential(self.head.output_size, [attn, rnn])
        else:
            return rnn

    @property
    def label_index(self) -> int:
        return self._label_index


class RecurrentFraudClassifierMulti(RecurrentFraudClassifier):
    def __init__(self, tensor_def: TensorDefinitionMulti, node_type: str, recurrent_features: int,
                 recurrent_layers: int, linear_layers: List[int], defaults=RecurrentClassifierDefaults()):
        self._t_def_m = tensor_def
        super(RecurrentFraudClassifierMulti, self).__init__(
            tensor_def.label_tensor_definition, node_type, recurrent_features, recurrent_layers, linear_layers, defaults
        )
        self._label_index = NumpyListDataSetMulti.label_index(tensor_def)

    def init_head(self) -> _Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHeadMulti(self._t_def_m, do, mn, mx)

    def init_body(self) -> _Layer:
        dense = self.defaults.get_bool('rec_body_dense')
        batch_norm = self.defaults.get_bool('rec_batch_norm')
        attn_heads = self.defaults.get_int('rec_attention_heads')
        attn_do = self.defaults.get_float('rec_attention_dropout')
        head = RecurrentFraudClassifierMulti._val_is_multi_head(self.head)
        lys = []
        for td, hs in zip(self._t_def_m.tensor_definitions, [h.output_size for h in head.heads]):
            # Only Rank 3 Tensor Definition are Series
            if td.rank == 3:
                # Add attention if needed
                if self._node_type == 'LSTM':
                    rnn = LSTMBody(hs, self._recurrent_features, self._recurrent_layers, dense, batch_norm)
                else:
                    rnn = GRUBody(hs, self._recurrent_features, self._recurrent_layers, dense, batch_norm)
                if attn_heads != 0:
                    # attn = Attention(self.head.output_size, attn_heads, attn_do)
                    attn = AttentionLastEntry(0, 1, 32)
                    seq = BodySequential(hs, [attn, rnn])
                    lys.append(seq)
                else:
                    lys.append(rnn)
            else:
                lys.append(None)
        return BodyMulti(self.head, lys)

    @property
    def label_index(self) -> int:
        return self._label_index


class ConvolutionalClassifierDefaults(ClassifierDefaults):
    def __init__(self):
        super(ConvolutionalClassifierDefaults, self).__init__()
        self.set_dense(True)
        self.set_conv_dropout(0.1)

    def set_dense(self, dense: bool):
        self.set('conv_body_dense', dense)

    def set_conv_dropout(self, dropout: float):
        self.set('conv_dropout', dropout)


class ConvolutionalFraudClassifier(BinaryClassifier):
    def __init__(self, tensor_def: TensorDefinition, conv_layers: List[Tuple[int, int, int]],
                 linear_layers: List[int], defaults=ConvolutionalClassifierDefaults()):
        self._t_def = tensor_def
        self._conv_layers = conv_layers
        self._label_index = self._t_def.learning_categories.index(LEARNING_CATEGORY_LABEL)
        super(ConvolutionalFraudClassifier, self).__init__(tensor_def, linear_layers, defaults)

    def init_head(self) -> _Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHead(self._t_def, do, mn, mx)

    def init_body(self) -> Union[_Layer, None]:
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

    def init_head(self) -> _Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHeadMulti(self._t_def_m, do, mn, mx)

    def init_body(self) -> Union[_Layer, None]:
        do = self.defaults.get_float('conv_dropout')
        dense = self.defaults.get_bool('conv_body_dense')
        head = RecurrentFraudClassifierMulti._val_is_multi_head(self.head)
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
