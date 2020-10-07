"""
Definition of a set of Numpy Helper classes.
(c) 2020 d373c7
"""
import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)


class NumpyListException(Exception):
    def __init__(self, message: str):
        super().__init__("Error Numpy-List: " + message)


class NumpyList:
    """Helper Class for a group of numpy arrays. It allows running operations like slicing, sampling, shuffling...
    consistently across a list of numpy arrays.

    Args:
        numpy_list: A List of numpy arrays. They must all be of the same length.
    """
    @staticmethod
    def _val_all_same_0_dim(numpy_list: List[np.array]):
        if len(set(list([n_entry.shape[0] for n_entry in numpy_list]))) > 1:
            raise NumpyListException(f'All Numpy arrays in a Numpy list must have the same number of rows')

    @staticmethod
    def _val_same_number_of_lists(numpy_list_l: 'NumpyList', numpy_list_r: 'NumpyList'):
        if numpy_list_l.number_of_lists != numpy_list_r.number_of_lists:
            raise NumpyListException(
                f'Both DataSet Lists must have the same number of numpy\'s. Left has {numpy_list_l.number_of_lists}. '
                f'Right has {numpy_list_r.number_of_lists}'
            )

    @staticmethod
    def _val_all_same_1_n_dim(numpy_l: np.array, numpy_r: np.array):
        if numpy_l.shape[1:] != numpy_r.shape[1:]:
            raise NumpyListException(
                f'Except for the 1st dimension, all numpy arrays must have the same shape. Left shape {numpy_l.shape}'
                f' Right shape {numpy_r.shape} '
            )

    @staticmethod
    def _val_label_has_1_dim(array: np.array):
        if len(array.shape) != 1:
            raise NumpyListException(
                f'The array containing the label for split should only have 1 dimension. it has {len(array.shape)}'
            )

    def __init__(self, numpy_list: List[np.array]):
        NumpyList._val_all_same_0_dim(numpy_list)
        self._numpy_list = numpy_list

    def __getitem__(self, item) -> np.array:
        return self._numpy_list[item]

    def __len__(self) -> int:
        if len(self._numpy_list) > 0:
            return len(self._numpy_list[0])

    @property
    def numpy_list(self) -> List[np.array]:
        return self._numpy_list

    @property
    def d_type_names(self) -> List[str]:
        return [array.dtype.name for array in self._numpy_list]

    @property
    def shapes(self) -> List[Tuple[int]]:
        """Get the shapes of the underlying numpy lists. Returns a list of Tuples. One tuple for each numpy in the class

        :return: A List of Tuples. Each Tuple contains the shape of a numpy
        """
        return [array.shape for array in self._numpy_list]

    @property
    def number_of_lists(self) -> int:
        """Returns the number of numpy arrays contained within this class

        :return: The number of numpy arrays in the list as int object.
        """
        return len(self._numpy_list)

    def pop(self, index: int) -> np.array:
        """Pop an numpy array from the list by index. This will return the numpy array at the index and remove it from
        the list.

        :param index: The index of the numpy array to be popped
        :return: A numpy array at 'index'. If it exists. Side effect: The numpy will be removed from the list
        """
        ret = self._numpy_list[index]
        del self._numpy_list[index]
        return ret

    def remove(self, index: int) -> None:
        """Remove an numpy array from the list.

        :param index: Index of the numpy list to be removed. If that index exists.
        """
        del self._numpy_list[index]

    # Function to shuffle various arrays randomly. This applies the same random order to each numpy.
    def shuffle(self) -> 'NumpyList':
        permutation = np.random.permutation(self._numpy_list[0].shape[0])
        shuffled = [sequence[permutation] for sequence in self._numpy_list]
        return NumpyList(shuffled)

    # Function to sample random records from various arrays. This samples the same random entries from each numpy.
    def sample(self, number_of_rows: int) -> 'NumpyList':
        permutation = np.random.permutation(number_of_rows)
        sampled = [sequence[permutation] for sequence in self._numpy_list]
        return NumpyList(sampled)

    # Function to slice various arrays in one go
    def slice(self, from_row_number=None, to_row_number=None) -> 'NumpyList':
        if from_row_number is not None and to_row_number is not None:
            sliced = [sequence[from_row_number:to_row_number] for sequence in self._numpy_list]
        elif from_row_number is not None:
            sliced = [sequence[from_row_number:] for sequence in self._numpy_list]
        elif to_row_number is not None:
            sliced = [sequence[:to_row_number] for sequence in self._numpy_list]
        else:
            sliced = [sequence for sequence in self._numpy_list]
        return NumpyList(sliced)

    # Function to split various numpy arrays into fraud non_fraud
    def split_fraud_non_fraud(self, label_index: int) -> Tuple['NumpyList', 'NumpyList']:
        NumpyList._val_label_has_1_dim(self._numpy_list[label_index])
        fraud_index = np.where(self._numpy_list[label_index] == 1)
        non_fraud_index = np.where(self._numpy_list[label_index] == 0)
        fraud = [sequence[fraud_index] for sequence in self._numpy_list]
        non_fraud = [sequence[non_fraud_index] for sequence in self._numpy_list]
        return NumpyList(fraud), NumpyList(non_fraud)

    # Function to concatenate 2 numpy_lists. It will concatenate each individual numpy in the list
    def concat(self, numpy_list: 'NumpyList') -> 'NumpyList':
        NumpyList._val_same_number_of_lists(self, numpy_list)
        new_list = []
        for numpy_l, numpy_r in zip(self._numpy_list, numpy_list):
            NumpyList._val_all_same_1_n_dim(numpy_l, numpy_r)
            new_list.append(np.concatenate([numpy_l, numpy_r], axis=0))
        return NumpyList(new_list)

    # Function to change the numpy type of the lists
    def as_type(self, numpy_type: str) -> 'NumpyList':
        new_list = [array.astype(numpy_type) for array in self._numpy_list]
        return NumpyList(new_list)

    # Function to split into test, validation an training
    def split_time(self, val_amount: int, test_amount: int) -> Tuple['NumpyList', 'NumpyList', 'NumpyList']:
        # Take x from end of lists as test
        test = self.slice(from_row_number=len(self)-test_amount, to_row_number=len(self))
        # Take another x from what is left at the end and not in test
        val = self.slice(from_row_number=len(self)-test_amount-val_amount, to_row_number=len(self)-test_amount)
        # Take rest
        train = self.slice(to_row_number=len(self)-test_amount-val_amount)
        return train, val, test
