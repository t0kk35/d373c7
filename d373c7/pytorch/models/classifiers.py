"""
Module for classifier Models
(c) 2020 d373c7
"""
import logging
import torch
import torch.utils.data as data
from .common import _Model, _LossBase, PyTorchModelException, ModelDefaults, _TensorHeadModel, _History
from ..layers import SingleClassBinaryOutput, LinDropAct
from ..optimizer import _Optimizer, AdamWOptimizer
from ..loss import SingleLabelBCELoss
from ...features import TensorDefinition, LEARNING_CATEGORY_LABEL
from typing import List, Dict


logger = logging.getLogger(__name__)


class _ClassifierModel(_Model):
    @staticmethod
    def _val_has_lc_label(tensor_def: TensorDefinition):
        if LEARNING_CATEGORY_LABEL not in tensor_def.learning_categories:
            raise PyTorchModelException(
                f'Tensor Definition <{tensor_def.name}> does not have a label learning category. '
                f'Can not build a classifier without a label. Please the .set_label(xyz) on the tensor definition'
            )

    def __init__(self, tensor_def: TensorDefinition, defaults: ModelDefaults):
        super(_ClassifierModel, self).__init__(tensor_def, defaults)
        _ClassifierModel._val_has_lc_label(tensor_def)
        self._label_index = tensor_def.learning_categories.index(LEARNING_CATEGORY_LABEL)

    def get_y(self, ds: List[torch.Tensor]) -> List[torch.Tensor]:
        # Return the label with Learning Category 'LEARNING_CATEGORY_LABEL'
        return ds[self._label_index: self._label_index+1]

    def optimizer(self, lr=None, wd=None) -> _Optimizer:
        return AdamWOptimizer(self, lr, wd)

    @property
    def default_metrics(self) -> List[str]:
        return ['acc', 'loss']


class BinaryClassifierHistory(_History):
    loss_key = 'loss'
    acc_key = 'acc'

    def _val_argument(self, args) -> data.DataLoader:
        if not isinstance(args[0], data.DataLoader):
            raise PyTorchModelException(
                f'Argument during creation of {self.__class__.__name__} should have been a data loader. ' +
                f'Was {type(args[0])}'
            )
        else:
            return args[0]

    @staticmethod
    def _val_is_tensor(arg):
        if not isinstance(arg, torch.Tensor):
            raise PyTorchModelException(
                f'Expected this argument to be a Tensor. Got {type(arg)}'
            )

    @staticmethod
    def _val_is_tensor_list(arg):
        if not isinstance(arg, List):
            raise PyTorchModelException(
                f'Expected this argument to be List. Got {type(arg)}'
            )
        if not isinstance(arg[0], torch.Tensor):
            raise PyTorchModelException(
                f'Expected this arguments list to contain tensors. Got {type(arg[0])}'
            )

    @staticmethod
    def _val_same_shape(t1: torch.Tensor, t2: torch.Tensor):
        if not t1.shape == t2.shape:
            raise PyTorchModelException(
                f'Shape of these tensors should have been the same. Got {t1.shape} and {t2.shape}'
            )

    def __init__(self, *args):
        dl = self._val_argument(args)
        _History.__init__(self, dl)

        self._history = {m: [] for m in [BinaryClassifierHistory.loss_key, BinaryClassifierHistory.acc_key]}
        self._running_loss = 0
        self._running_correct_cnt = 0
        self._running_count = 0

    def end_step(self, *args):
        BinaryClassifierHistory._val_is_tensor(args[0])
        BinaryClassifierHistory._val_is_tensor_list(args[1])
        BinaryClassifierHistory._val_is_tensor(args[2])
        pr, lb, loss = args[0], args[1][0], args[2]
        BinaryClassifierHistory._val_same_shape(pr, lb)
        self._running_loss += loss.item()
        p = torch.ge(pr, 0.5)
        x = torch.eq(p, lb)
        y = torch.sum(x)
        self._running_correct_cnt += y.item()
        # self._running_correct_cnt += torch.sum(torch.eq(p, lb[0])).item()
        self._running_count += pr.shape[0]
        super(BinaryClassifierHistory, self).end_step(pr, lb, loss)

    def end_epoch(self):
        self._history[BinaryClassifierHistory.loss_key].append(round(self._running_loss/self.steps, 4))
        self._history[BinaryClassifierHistory.acc_key].append(round(self._running_correct_cnt/self.samples, 4))
        self._running_correct_cnt = 0
        self._running_count = 0
        self._running_loss = 0
        super(BinaryClassifierHistory, self).end_epoch()

    @property
    def history(self) -> Dict:
        return self._history

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
        self.set('lin_interlayer_drop_out', 0.1)

    def emb_dim(self, minimum: int, maximum: int, dropout: float):
        self.set('emb_min_dim', minimum)
        self.set('emb_max_dim', maximum)
        self.set('emb_dropout', dropout)


class FeedForwardFraudClassifier(_ClassifierModel, _TensorHeadModel):
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

    def __init__(self, tensor_def: TensorDefinition, layers: List[int], defaults=ClassifierDefaults()):
        super(FeedForwardFraudClassifier, self).__init__(tensor_def, defaults)
        self._val_layers(layers)
        do = self.defaults.get_float('lin_interlayer_drop_out')
        ly = [(i, do) for i in layers]
        self.linear = LinDropAct(self.head.output_size, ly)
        self.out = SingleClassBinaryOutput(self.linear.output_size)
        self._loss_fn = SingleLabelBCELoss()

    def forward(self, x):
        x = _TensorHeadModel.forward(self, x)
        x = self.linear(x)
        x = self.out(x)
        return x

    def loss_fn(self) -> _LossBase:
        return self._loss_fn

    def history(self, *args) -> _History:
        return BinaryClassifierHistory(args)
