"""
Imports for Pytorch data
(c) 2020 d373c7
"""
import torch
from torch.utils.data import Dataset
from ..features.common import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_CATEGORICAL
from ..features.tensor import TensorDefinition
from ..engines import NumpyList
from .common import PyTorchTrainException

from typing import List


class _DTypeHelper:
    DEFAULT_TYPES_PER_LEARNING_CATEGORY = {
        LEARNING_CATEGORY_CONTINUOUS: torch.float32,
        LEARNING_CATEGORY_BINARY: torch.float32,
        LEARNING_CATEGORY_CATEGORICAL: torch.long
    }

    @staticmethod
    def get_dtypes(tensor_definition: TensorDefinition) -> List[torch.dtype]:
        dtypes = []
        for lc in tensor_definition.learning_categories:
            d_type = _DTypeHelper.DEFAULT_TYPES_PER_LEARNING_CATEGORY.get(lc, None)
            if d_type is None:
                PyTorchTrainException(
                    f''
                )
            else:
                dtypes.append(d_type)
        return dtypes


class NumpyListDataSet(Dataset):
    def __init__(self, tensor_def: TensorDefinition, npl: NumpyList):
        self._npl = npl
        self._dtypes = _DTypeHelper.get_dtypes(tensor_def)
        self.device = torch.device('cpu')

    def __len__(self):
        return len(self._npl)

    def __getitem__(self, item: int) -> List[torch.Tensor]:
        res = [torch.as_tensor(array[item], dtype=dt, device=self.device)
               for array, dt in zip(self._npl.lists, self._dtypes)]
        return res
