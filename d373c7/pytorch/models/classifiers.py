"""
Module for classifier Models
(c) 2020 d373c7
"""
import logging
import torch
import torch.nn as nn
from .common import _LossBase, PyTorchModelException, ModelDefaults, _History, _Model
from ..layers import SingleClassBinaryOutput, LinDropAct, TensorDefinitionHead, TensorDefinitionHeadMulti
# noinspection PyProtectedMember
from ..layers.common import _Layer
from ..optimizer import _Optimizer, AdamWOptimizer
from ..loss import SingleLabelBCELoss
from ..data import NumpyListDataSetMulti
from ...features import TensorDefinition, TensorDefinitionMulti, LEARNING_CATEGORY_LABEL, FeatureLabelBinary
from ...features import FeatureCategorical
from typing import List, Dict


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
        self.set_batch_norm(True)
        self.set('lin_interlayer_drop_out', 0.1)

    def emb_dim(self, minimum: int, maximum: int, dropout: float):
        self.set('emb_min_dim', minimum)
        self.set('emb_max_dim', maximum)
        self.set('emb_dropout', dropout)

    def set_batch_norm(self, flag: bool) -> None:
        """Define if a batch norm layer will be added before the final hidden layer.

        :return: None
        """
        self.set('batch_norm', flag)

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

    def __init__(self, tensor_def: TensorDefinition, layers: List[int], defaults: ClassifierDefaults):
        super(BinaryClassifier, self).__init__(defaults)
        self._val_layers(layers)
        self._tensor_def = tensor_def
        self._val_has_lc_label(tensor_def)
        self._val_label(tensor_def)
        do = self.defaults.get_float('lin_interlayer_drop_out')
        bn = self.defaults.get_bool('batch_norm')
        ly = [(i, do) for i in layers]
        self.head = self.init_head()
        self.linear = LinDropAct(self.head.output_size, ly)
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
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.out(x)
        return x

    def init_head(self) -> _Model:
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
        self._t_def = tensor_def
        self._label_index = NumpyListDataSetMulti.label_index(tensor_def)
        super(FeedForwardFraudClassifierMulti, self).__init__(tensor_def.label_tensor_definition, layers, defaults)

    def init_head(self) -> _Layer:
        mn = self.defaults.get_int('emb_min_dim')
        mx = self.defaults.get_int('emb_max_dim')
        do = self.defaults.get_float('emb_dropout')
        return TensorDefinitionHeadMulti(self._t_def, do, mn, mx)

    @property
    def label_index(self) -> int:
        return self._label_index
