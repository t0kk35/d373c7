"""
Definition of a set of Numpy Helper classes.
(c) 2020 d373c7
"""
import logging
import numpy as np
from ..features import TensorDefinition, LEARNING_CATEGORY_LABEL
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)


class NumpyListException(Exception):
    def __init__(self, message: str):
        super().__init__("Error Numpy-List: " + message)


class NumpyList:
    """Helper Class for a group of numpy arrays. It allows running operations like slicing, sampling, shuffling...
    consistently across a list of numpy arrays.

    :argument numpy_list: A List of numpy arrays. They must all be of the same length.
    """
    @staticmethod
    def _val_all_same_0_dim(numpy_list: List[np.array]):
        if len(set(list([n_entry.shape[0] for n_entry in numpy_list]))) > 1:
            raise NumpyListException(f'All Numpy arrays in a Numpy list must have the same number of rows')

    @staticmethod
    def _val_index_in_range(numpy_list: 'NumpyList', index: int):
        if index > numpy_list.number_of_lists - 1:
            raise NumpyListException(
                f'Trying to access index {index} in a numpy list of length {len(numpy_list)}'
            )

    @staticmethod
    def _slice_in_range(numpy_list: 'NumpyList', index: int):
        if index < 0:
            raise NumpyListException(f'Slice index can not be smaller than 0. Got {index}')
        if index > len(numpy_list):
            raise NumpyListException(
                f'Slice index can not go beyond length of lists. Got {index}, length {len(numpy_list)}'
            )

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

    @staticmethod
    def _val_is_integer_type(numpy_list: 'NumpyList', index: int):
        if not np.issubdtype(numpy_list.lists[index].dtype, np.integer):
            raise NumpyListException(
                f'List at index <{index}> is not of integer type. That is unexpected'
            )

    @staticmethod
    def _val_val_plus_test_smaller_than_length(npl: 'NumpyList', validation: int, test: int):
        if validation + test >= len(npl):
            raise NumpyListException(
                f'The number of validation <{validation}> + the number of test <{test}> records. Is bigger than the ' +
                f'Length of the Numpy List <{len(npl)}> '
            )

    @staticmethod
    def _val_single_label(tensor_def: TensorDefinition):
        if len(tensor_def.label_features()) < 1:
            raise NumpyListException(
                f'TensorDefinition <{tensor_def.name}> should have a label feature'
            )

        if len(tensor_def.label_features()) > 1:
            raise NumpyListException(
                f'TensorDefinition <{tensor_def.name} should only have one label feature>'
            )

    def __init__(self, numpy_list: List[np.array]):
        NumpyList._val_all_same_0_dim(numpy_list)
        self._numpy_list = numpy_list

    def __getitem__(self, subscript) -> np.array:
        if isinstance(subscript, slice):
            return self._slice(subscript.start, subscript.stop)
        elif isinstance(subscript, int):
            if subscript < 0:
                subscript += len(self)
            return self._slice(subscript, subscript+1)
        else:
            raise NumpyListException(f'Something went wrong. Got wrong subscript type f{subscript}')

    def __len__(self) -> int:
        if len(self._numpy_list) > 0:
            return len(self._numpy_list[0])

    def __repr__(self):
        return f'Numpy List with shapes: {self.shapes}'

    @property
    def lists(self) -> List[np.ndarray]:
        return self._numpy_list

    @property
    def dtype_names(self) -> List[str]:
        """Returns the names (i.e. as string) dtypes of the underlying numpy arrays.

        :return: List of string dtype string representations
        """
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

    def pop(self, index: int) -> np.ndarray:
        """Pop an numpy array from the list by index. This will return the numpy array at the index and remove it from
        the list.

        :param index: The index of the numpy array to be popped
        :return: A numpy array at 'index'. If it exists. Side effect: The numpy will be removed from the list
        """
        self._val_index_in_range(self, index)
        ret = self._numpy_list[index]
        del self._numpy_list[index]
        return ret

    def remove(self, index: int) -> None:
        """Remove an numpy array from the list.

        :param index: Index of the numpy list to be removed. If that index exists.
        """
        self._val_index_in_range(self, index)
        del self._numpy_list[index]

    def unique(self, index: int) -> (List[int], List[int]):
        """Return the unique entries and counts of a specific numpy list

        :param index: Index of the list for which to run the unique operation.
        :return: A Tuple, the first element is the unique entries, the second entry is the counts.
        """
        NumpyList._val_index_in_range(self, index)
        NumpyList._val_is_integer_type(self, index)
        val, cnt = np.unique(self.lists[index], return_counts=True)
        return list(val), list(cnt)

    def shuffle(self) -> 'NumpyList':
        """Shuffle the numpy arrays in the list across the 0 dimension. The shuffling is consistent across lists.
        Meaning that for instance all rows in the various arrays at index x of the input will be moved to index y.
        This will make sure samples are shuffled consistently

        :return: The shuffled numpy arrays.
        """
        permutation = np.random.permutation(self._numpy_list[0].shape[0])
        shuffled = [sequence[permutation] for sequence in self._numpy_list]
        return NumpyList(shuffled)

    def sample(self, number_of_rows: int) -> 'NumpyList':
        """Sample random 'number_of_rows' from each of the arrays in the list. Each array will be sampled consistently

        :param number_of_rows : The number of rows to sample from the arrays
        :return: The sampled list of numpy arrays
        """
        self._slice_in_range(self, number_of_rows)
        permutation = np.random.permutation(number_of_rows)
        sampled = [sequence[permutation] for sequence in self._numpy_list]
        return NumpyList(sampled)

    def _slice(self, from_row_number=None, to_row_number=None) -> 'NumpyList':
        """Slice all arrays in a numpy list.

        :param from_row_number:
        :param to_row_number:
        :return: The sliced numpy list
        """
        if from_row_number is not None and to_row_number is not None:
            self._slice_in_range(self, from_row_number)
            self._slice_in_range(self, to_row_number)
            sliced = [sequence[from_row_number:to_row_number] for sequence in self._numpy_list]
        elif from_row_number is not None:
            self._slice_in_range(self, from_row_number)
            sliced = [sequence[from_row_number:] for sequence in self._numpy_list]
        elif to_row_number is not None:
            self._slice_in_range(self, to_row_number)
            sliced = [sequence[:to_row_number] for sequence in self._numpy_list]
        else:
            sliced = [sequence for sequence in self._numpy_list]
        return NumpyList(sliced)

    def filter_label(self, tensor_def: TensorDefinition, label: Any) -> 'NumpyList':
        """Method to filter a specific class from the labels. It can for instance be used to filter Fraud or Non-Fraud

        :param tensor_def: The Tensor definition used to build the NumpyList
        :param label: The label value (class) we want to filter.
        :return: New filtered numpy list, filtered on the label value
        """
        NumpyList.is_built_from(self, tensor_def)
        NumpyList._val_single_label(tensor_def)
        label_index = tensor_def.learning_categories.index(LEARNING_CATEGORY_LABEL)
        index = np.where(self._numpy_list[label_index] == label)
        lists = [sequence[index] for sequence in self._numpy_list]
        return NumpyList(lists)

    def concat(self, numpy_list: 'NumpyList') -> 'NumpyList':
        """Function to concatenate 2 numpy_lists. It will concatenate each individual numpy in the list

        :param numpy_list: The numpy list to concatenate to the current list
        :return: A numpy list with the input numpy list concatenated to the current numpy list
        """
        NumpyList._val_same_number_of_lists(self, numpy_list)
        new_list = []
        for numpy_l, numpy_r in zip(self.lists, numpy_list.lists):
            NumpyList._val_all_same_1_n_dim(numpy_l, numpy_r)
            new_list.append(np.concatenate([numpy_l, numpy_r], axis=0))
        return NumpyList(new_list)

    # Function to change the numpy type of the lists
    def as_type(self, numpy_type: str) -> 'NumpyList':
        new_list = [array.astype(numpy_type) for array in self._numpy_list]
        return NumpyList(new_list)

    def split_time(self, val_number: int, test_number: int) -> Tuple['NumpyList', 'NumpyList', 'NumpyList']:
        """Split a numpy list into training, validation and test. Where the first portion of the data is training, the
        middle is the validation and the end of the data is the test. This is almost always the best way to split
        transactional data. First the 'test_number' of records data is taken from the end of the arrays. Of what is
        left the 'val_number' is taken all that is left is training.

        :param val_number: Number of records to allocate in the validation set
        :param test_number: Number of records to allocate in the test set.
        :return: Tuple of 3 numpy lists containing the training, validation and test data respectively
        """
        NumpyList._val_val_plus_test_smaller_than_length(self, val_number, test_number)
        # Take x from end of lists as test
        test = self._slice(from_row_number=len(self)-test_number, to_row_number=len(self))
        # Take another x from what is left at the end and not in test
        val = self._slice(from_row_number=len(self)-test_number-val_number, to_row_number=len(self)-test_number)
        # Take rest
        train = self._slice(to_row_number=len(self)-test_number-val_number)
        return train, val, test

    def is_built_from(self, tensor_definition: TensorDefinition) -> bool:
        """Method to validate that a Numpy list was likely built from a specific tensor definition. The data types can
        not be checked. The checks mainly revolve around the sizes of the lists.

        :param:tensor_definition. The tensor definition to be checked.
        :return: True of False. True if this Numpy List and TensorDefinition are compatible
        """
        if not tensor_definition.inference_ready:
            logger.info(f'Tensor Definition and Numpy list not compatible. Tensor Definition is not inference ready')
            return False

        lc = tensor_definition.learning_categories

        if len(lc) != self.number_of_lists:
            logger.info(f'Tensor Definition and Numpy list not compatible. Expected {len(lc)} lists in the Numpy list')
            return False

        for lc, npl in zip(lc, self.lists):
            f = tensor_definition.filter_features(lc, expand=True)
            if tensor_definition.rank == 2:
                shape = npl.shape[1] if len(npl.shape) > 1 else 1
                if len(f) != shape:
                    logger.info(
                        f'Tensor Definition and Numpy not compatible. '
                        f'Learning Type {lc.name} does not have same # elements'
                    )
                    return False
            else:
                logger.info(f'Tensor Definition and Numpy not compatible. Rank in definition {tensor_definition.rank}')
                return False

        # All good if we manage to get here.
        return True
