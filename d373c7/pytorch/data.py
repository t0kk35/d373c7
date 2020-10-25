"""
Imports for Pytorch data
(c) 2020 d373c7
"""
import logging
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from ..features.common import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CATEGORICAL
from ..features.common import LEARNING_CATEGORY_LABEL
from ..features.tensor import TensorDefinition
from ..engines import NumpyList
from .common import PyTorchTrainException
from typing import List


logger = logging.getLogger(__name__)


class _DTypeHelper:
    DEFAULT_TYPES_PER_LEARNING_CATEGORY = {
        LEARNING_CATEGORY_CONTINUOUS: torch.float32,
        LEARNING_CATEGORY_BINARY: torch.float32,
        LEARNING_CATEGORY_CATEGORICAL: torch.long,
        LEARNING_CATEGORY_LABEL: torch.float32
    }

    @staticmethod
    def get_dtypes(tensor_definition: TensorDefinition) -> List[torch.dtype]:
        dtypes = []
        for lc in tensor_definition.learning_categories:
            d_type = _DTypeHelper.DEFAULT_TYPES_PER_LEARNING_CATEGORY.get(lc, None)
            if d_type is None:
                PyTorchTrainException(
                    f'Could not determine default Torch tensor data type for learning category <{lc}>'
                )
            else:
                dtypes.append(d_type)
        return dtypes


class NumpyListDataSet(Dataset):
    """Custom PyTorch 'Dataset' for numpy lists. The idea is that the data is kept in original numpy. Normally in the
    most condensed form. The numpy arrays are converted to Pytorch tensors on the fly, as each row is requested. This
    create a bit of CPU overhead, but as the numpy arrays should be more condensed, (the PyTorch tensors are mainly
    float32 and long), it should allow keeping more data in memory.

    :argument tensor_def: The tensor definition off of which the numpy list was created.
    :argument npl: The numpy list to use as base for the Pytorch Dataset.
    """
    @staticmethod
    def _val_built_from(npl: NumpyList, tensor_definition: TensorDefinition):
        if not npl.is_built_from(tensor_definition):
            raise PyTorchTrainException(
                f'The NumpyList does not seem to be built from the given TensorDefinition'
            )

    def __init__(self, tensor_def: TensorDefinition, npl: NumpyList):
        NumpyListDataSet._val_built_from(npl, tensor_def)
        self._npl = npl
        self._tensor_def = tensor_def
        self._dtypes = _DTypeHelper.get_dtypes(tensor_def)
        # Yes assign to CPU. We could directly allocate to the GPU, but then we can only use one worker :|
        self.device = torch.device('cpu')

    def __len__(self):
        return len(self._npl)

    def __getitem__(self, item: int) -> List[torch.Tensor]:
        res = [torch.as_tensor(array[item], dtype=dt, device=self.device)
               for array, dt in zip(self._npl.lists, self._dtypes)]
        return res

    def data_loader(self, device: torch.device, batch_size: int, num_workers: int = 1,
                    shuffle: bool = False, sampler: Sampler = None) -> DataLoader:
        """Create a Pytorch Data-loader for the underlying Data-set.

        :param device: The Pytorch device on which to create the data. Either CPU or GPU. Note that if the the device is
            set to GPU only one worker can be used.
        :param batch_size: The batch size for the Data-loader.
        :param num_workers: Number of workers to use in the Data-loader. Default = 1. If more than one worker is
            defined the device will default to 'cpu' because 'cuda' devices do not support multiple workers.
        :param shuffle: Flag to trigger random shuffling of the dataset. Default = False
        :param sampler: Sampler to use. Optional. Needs to be an instance of a Sampler (from the Pytorch library.
        :return: A Pytorch data-loader for this data-set. Ready to train.
        """
        # Cuda does not support multiple workers. Override if GPU
        if num_workers > 1:
            if self.device.type == 'cuda':
                logger.warning(f'Defaulted to using the cpu for the data-loader of <{self._tensor_def.name}>.' +
                               f' Multiple workers not supported by "cuda" devices. ')
            self.device = torch.device('cpu')
            dl = DataLoader(
                self, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, pin_memory=True, sampler=sampler
            )
        else:
            # Only CPU Tensors can be pinned
            pin = False if device.type == 'cuda' else True
            self.device = device
            dl = DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=pin, sampler=sampler)
        return dl


class ClassSampler:
    """ Class for creating a sampler.

    :argument npl: The Numpy List to sample.
    :argument tensor_definition: The Tensor definition used to create the numpy List
    """
    @staticmethod
    def _val_built_from(npl: NumpyList, tensor_definition: TensorDefinition):
        if not npl.is_built_from(tensor_definition):
            raise PyTorchTrainException(
                f'The NumpyList does not seem to be built from the given TensorDefinition'
            )

    def __init__(self, tensor_definition: TensorDefinition, npl: NumpyList):
        ClassSampler._val_built_from(npl, tensor_definition)
        self._npl = npl
        self._tensor_def = tensor_definition

    def over_sampler(self, replacement=True) -> Sampler:
        """Create a RandomWeightedSampler that balances out the classes. It'll more or less return an equal amount of
        each class. For a binary fraud label this would mean about as much fraud as non-fraud samples.

        :param replacement: Bool flag to trigger sample with replacement. With replacement a row can be drawn more
        than once
        """
        label_index = self._tensor_def.learning_categories.index(LEARNING_CATEGORY_LABEL)
        _, class_balance = self._npl.unique(label_index)
        weights = 1./torch.tensor(class_balance, dtype=torch.float)
        sample_weights = weights[self._npl.lists[label_index].astype(int)]
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self._npl),
            replacement=replacement
        )
        return train_sampler
