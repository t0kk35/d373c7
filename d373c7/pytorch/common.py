"""
Imports for Pytorch Training functions
(c) 2020 d373c7
"""
import logging
import torch
import torch.utils.data as data
# noinspection PyProtectedMember
from typing import List, Dict, Tuple


logger = logging.getLogger(__name__)


def init_devices() -> Tuple[torch.device, torch.device]:
    """Set up the required PyTorch devices. This will check if there is a cuda device, if so it will use that as main
    device. If not the the cpu will used as main device.

    :return: Tuple of devices. The first is the main device to use, the second is an auxiliary cpu device.
    """

    logger.info(f'Torch Version : {torch.__version__}')

    # Set up the GPU if available. This will be the default device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.info(f'GPU found. Using GPU <{device.index}>')
        logger.info(f'Cuda Version  : {torch.version.cuda}')
    else:
        device = torch.device('cpu')
        logger.info(f'No GPU found ... Using CPU <{device}> as main device')

    # Also set up a cpu device
    cpu = torch.device('cpu')
    return device, cpu


class PyTorchTrainException(Exception):
    """Standard exception raised during training"""
    def __init__(self, message: str):
        super().__init__('Error in PyTorch Training: ' + message)


class _History:
    """Object that keeps track of metrics during training and testing

    :argument dl: A Data loader that will be iterated over in the training or validation loop.
    """
    def _val_argument(self, args) -> data.DataLoader:
        if not isinstance(args[0], data.DataLoader):
            raise PyTorchTrainException(
                f'Argument during creation of {self.__class__.__name__} should have been a data loader. ' +
                f'Was {type(args[0])}'
            )
        else:
            return args[0]

    @staticmethod
    def _val_is_tensor(arg):
        if not isinstance(arg, torch.Tensor):
            raise PyTorchTrainException(
                f'Expected this argument to be a Tensor. Got {type(arg)}'
            )

    @staticmethod
    def _val_is_tensor_list(arg):
        if not isinstance(arg, List):
            raise PyTorchTrainException(
                f'Expected this argument to be List. Got {type(arg)}'
            )
        if not isinstance(arg[0], torch.Tensor):
            raise PyTorchTrainException(
                f'Expected this arguments list to contain tensors. Got {type(arg[0])}'
            )

    def __init__(self, dl: data.DataLoader, history: Dict[str, List]):
        self._batch_size = dl.batch_size
        self._samples = len(dl.dataset)
        self._step = 0
        self._steps = len(dl)
        self._epoch = 0
        self._history = history

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def history(self) -> Dict:
        return self._history

    @property
    def epoch(self):
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    def start_step(self):
        self._step += 1

    def end_step(self, *args):
        pass

    def early_break(self) -> bool:
        pass

    def start_epoch(self):
        self._step = 0
        self._epoch += 1

    def end_epoch(self):
        pass
