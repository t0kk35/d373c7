"""
Imports for Pytorch data
(c) 2020 d373c7
"""
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from ..features.common import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CATEGORICAL
from ..features.common import LEARNING_CATEGORY_LABEL
from ..features.tensor import TensorDefinition
from ..engines import NumpyList
from .common import PyTorchTrainException
from typing import List


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
    @staticmethod
    def _val_built_from(tensor_definition: TensorDefinition, npl: NumpyList):
        if not npl.is_built_from(tensor_definition):
            raise PyTorchTrainException(
                f'The NumpyList does not seem to be built from the given TensorDefinition'
            )

    def __init__(self, tensor_def: TensorDefinition, npl: NumpyList):
        NumpyListDataSet._val_built_from(tensor_def, npl)
        self._npl = npl
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
        # Cuda does not support multiple workers. Override if GPU
        if num_workers > 1:
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
    """ Class for creating a sampler. Samplers define how data is selected from the data loader.

    :argument tensor_definition: The Tensor definition used to create the numpy List
    :argument npl: The Numpy List to sample.
    """
    @staticmethod
    def _val_built_from(tensor_definition: TensorDefinition, npl: NumpyList):
        if not npl.is_built_from(tensor_definition):
            raise PyTorchTrainException(
                f'The NumpyList does not seem to be built from the given TensorDefinition'
            )

    @staticmethod
    def _val_batch_size(npl: NumpyList, batch_size: int, replacement: int):
        if not replacement and batch_size > len(npl):
            raise PyTorchTrainException(
                f'Can not create weighted random sampler with batch size <{batch_size}> which is smaller than '
                f'then length of the numpy-list <{len(npl)}> and replacement False'
            )

    def __init__(self, tensor_definition: TensorDefinition, npl: NumpyList):
        ClassSampler._val_built_from(tensor_definition, npl)
        self._npl = npl
        self._tensor_def = tensor_definition

    def over_sampler(self, batch_size: int, replacement=True) -> Sampler:
        """Create a RandomWeightedSampler that balances out the classes. It'll more or less return an equal amount of
        each class. For a binary fraud label this would mean about as much fraud as non-fraud samples.

        :param batch_size: The number of samples to draw. This should match the batch size of the data-loader.
        :param replacement: Bool flag to trigger sample with replacement. With replacement a row can be drawn more
            than once
        """
        ClassSampler._val_batch_size(self._npl, batch_size, replacement)
        label_index = self._tensor_def.learning_categories.index(LEARNING_CATEGORY_LABEL)
        _, class_balance = self._npl.unique(label_index)
        weights = 1./torch.tensor(class_balance, dtype=torch.float)
        sample_weights = weights[self._npl.lists[label_index].astype(int)]
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=batch_size,
            replacement=replacement
        )
        return train_sampler
