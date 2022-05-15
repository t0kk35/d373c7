"""
Definition of a set of Numpy Helper classes.
(c) 2020 d373c7
"""
import logging
import numpy as np
from ..features import TensorDefinition, TensorDefinitionMulti, LEARNING_CATEGORY_LABEL
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)


class NumpyListException(Exception):
    def __init__(self, message: str):
        super().__init__("Error Numpy-List: " + message)


class NumpyList:
    """
    Helper Class for a group of numpy arrays. It allows running operations like slicing, sampling, shuffling...
    consistently across a list of numpy arrays.

    Args:
        numpy_list (List[np.ndarray]) : A List of numpy arrays. They must all be of the same length.
    """
    @staticmethod
    def _val_all_same_0_dim(numpy_list: List[np.ndarray]):
        """
        Check that all arrays have the shape share in the 0th dimension.

        Args:
            numpy_list (List[np.ndarray]): The list of numpy arrays to validate

        Raises:
            NumpyListException : If the size of the 0th dimension is not the same for all arrays.

        Returns:
            None
        """
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
    def _val_all_same_1_n_dim(numpy_l: np.array, numpy_r: np.ndarray):
        if numpy_l.shape[1:] != numpy_r.shape[1:]:
            raise NumpyListException(
                f'Except for the 1st dimension, all numpy arrays must have the same shape. Left shape {numpy_l.shape}'
                f' Right shape {numpy_r.shape} '
            )

    @staticmethod
    def _val_label_has_1_dim(array: np.ndarray):
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

    def _val_is_built_from(self, tensor_def: TensorDefinition):
        if not self.is_built_from(tensor_def):
            raise NumpyListException(
                f'Looks like this Numpy list was not built from Tensor Definition <{tensor_def.name}>. ' +
                f'Check log for details.'
            )

    def __init__(self, numpy_list: List[np.ndarray]):
        NumpyList._val_all_same_0_dim(numpy_list)
        self._numpy_list = numpy_list

    def __getitem__(self, subscript) -> 'NumpyList':
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
        """
        Returns the names (i.e. as string) of the dtypes of the underlying numpy arrays.

        Returns:
            List of string dtype string representations
        """
        return [array.dtype.name for array in self._numpy_list]

    @property
    def shapes(self) -> List[Tuple[int]]:
        """
        Get the shapes of the underlying numpy lists. Returns a list of Tuples. One tuple for each numpy in the class

        Returns:
            A List of Tuples. Each Tuple contains the shape of a numpy
        """
        return [array.shape for array in self._numpy_list]

    @property
    def number_of_lists(self) -> int:
        """
        Returns the number of numpy arrays contained within this object

        Returns:
            The number of numpy arrays in the list as int object.
        """
        return len(self._numpy_list)

    def pop(self, index: int) -> np.ndarray:
        """
        Pop a numpy array from the list by index. This will return the numpy array at the index and remove it from
        the list.

        Args:
            index (int) : The index of the numpy array to be popped

        Returns:
            A numpy array at 'index'. If it exists. Side effect: The numpy will be removed from the list
        """
        self._val_index_in_range(self, index)
        ret = self._numpy_list[index]
        del self._numpy_list[index]
        return ret

    def remove(self, index: int) -> None:
        """
        Remove a numpy array from the list.

        Args:
            index (int) : Index of the numpy list to be removed. If that index exists.

        Returns:
            None
        """
        self._val_index_in_range(self, index)
        del self._numpy_list[index]

    def unique(self, index: int) -> (List[int], List[int]):
        """
        Return the unique sorted entries and counts of a specific array within this NumpyList

        Args:
            index (int): Index of the list for which to run the unique operation.

        Returns:
            A Tuple, the first element is the unique entries, the second entry is the counts.
        """
        NumpyList._val_index_in_range(self, index)
        NumpyList._val_is_integer_type(self, index)
        val, cnt = np.unique(self.lists[index], return_counts=True)
        return list(val), list(cnt)

    def shuffle(self) -> 'NumpyList':
        """
        Shuffle the numpy arrays in the list across the 0 dimension. The shuffling is consistent across lists.
        Meaning that for instance all rows in the various arrays at index x of the input will be moved to index y.
        This will make sure samples are shuffled consistently

        Returns:
            A NumpyList containing the shuffled numpy arrays.
        """
        permutation = np.random.permutation(self._numpy_list[0].shape[0])
        shuffled = [sequence[permutation] for sequence in self._numpy_list]
        return NumpyList(shuffled)

    def sample(self, number_of_rows: int) -> 'NumpyList':
        """
        Sample random 'number_of_rows' from each of the arrays in the list. Each array will be sampled consistently

        Args:
            number_of_rows (int): The number of rows to sample from the arrays

        Returns:
            The sampled list of numpy arrays in a NumpyList object
        """
        self._slice_in_range(self, number_of_rows)
        permutation = np.random.permutation(number_of_rows)
        sampled = [sequence[permutation] for sequence in self._numpy_list]
        return NumpyList(sampled)

    def _slice(self, from_row_number=None, to_row_number=None) -> 'NumpyList':
        """
        Slice all the arrays in this NumpyList

        Args:
            from_row_number (int): The start number
            to_row_number (int): The end number (exclusive)

        Returns:
            The sliced numpy list
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

    def filter_label(self, tensor_def: [TensorDefinition, TensorDefinitionMulti], label: Any) -> 'NumpyList':
        """
        Method to filter a specific class from the labels. It can for instance be used to filter Fraud or Non-Fraud

        Args:
            tensor_def (TensorDefinition): The TensorDefinition or TensorDefinitionMulti used to build the NumpyList
            label (Any): The label value (class) we want to filter.

        Returns:
            New filtered numpy list, filtered on the label value
        """
        if isinstance(tensor_def, TensorDefinitionMulti):
            self.multi_is_built_from(tensor_def)
            label_td = tensor_def.label_tensor_definition
            NumpyList._val_single_label(label_td)
            label_index = tensor_def.label_index
        else:
            self._val_is_built_from(tensor_def)
            NumpyList._val_single_label(tensor_def)
            label_index = tensor_def.learning_categories.index(LEARNING_CATEGORY_LABEL)

        labels = self._numpy_list[label_index]
        if len(labels.shape) == 2:
            labels = np.squeeze(labels)
        index = np.where(labels == label)
        lists = [sequence[index] for sequence in self._numpy_list]
        return NumpyList(lists)

    def concat(self, numpy_list: 'NumpyList') -> 'NumpyList':
        """
        Function to concatenate 2 numpy_lists. It will concatenate each individual numpy in the list

        Args:
            numpy_list: The numpy list to concatenate to the current list

        Returns:
            A numpy list with the input numpy list concatenated to the current numpy list
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
        """
        Split a numpy list into training, validation and test. Where the first portion of the data is training, the
        middle is the validation and the end of the data is the test. This is almost always the best way to split
        transactional data. First the 'test_number' of records data is taken from the end of the arrays. Of what is
        left the 'val_number' is taken all that is left is training.

        Args:
            val_number (int): Number of records to allocate to the validation set
            test_number (int): Number of records to allocate to the test set.

        Returns:
            Tuple of 3 numpy lists containing the training, validation and test data respectively
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
        """
        Method to validate that a Numpy list was likely built from a specific tensor definition. The data types can
        not be checked. The checks mainly revolve around the sizes of the lists.

        Args:
            tensor_definition (TensorDefinition): The tensor definition to be checked.

        Returns:
            True of False True if this Numpy List and TensorDefinition are compatible
        """
        if not tensor_definition.inference_ready:
            logger.info(f'Tensor Definition and Numpy list not compatible. Tensor Definition is not inference ready')
            return False

        if len(tensor_definition.learning_categories) != self.number_of_lists:
            logger.info(f'Tensor Definition and Numpy list not compatible. Number of Arrays in List does not match ' +
                        f'the number of Learning Categories in the Tensor Definition. ' +
                        f'Expected {len(tensor_definition.learning_categories)} lists in the Numpy list')
            return False

        for i, (s, npl) in enumerate(zip(tensor_definition.shapes, self.lists)):
            # All except Batch dim must match
            if s[1:] != npl.shape[1:]:
                logger.info(
                    f'Tensor Definition and Numpy not compatible. Expected shape {s}. Numpy has shape {np.shape} for' +
                    f'list number <{i+1}>'
                )
                return False

        # All good if we manage to get here.
        return True

    def multi_filter(self, tensor_def_m: TensorDefinitionMulti, tensor_def_filter: TensorDefinition) -> 'NumpyList':
        """
        Filters the lists of a specific tensor_definition out of the TensorDefinitionMultiHead

        Args:
            tensor_def_m (TensorDefinitionMulti): The TensorDefinitionMulti that was used to build the Numpy.
            tensor_def_filter (TensorDefinition): The TensorDefinition we want the lists for.

        Returns:
            New NumpyList object containing only the lists of the filtered TensorDefinition
        """
        # Build a list of accumulated counts.
        acc_counts = [0]
        for td in tensor_def_m.tensor_definitions:
            acc_counts.append(len(td.learning_categories) + acc_counts[-1])
        # the i-th element is the entry we need to filter out.
        i = tensor_def_m.tensor_definitions.index(tensor_def_filter)
        # Filter the Numpy lists. Use the current counts as start and end of a slice
        npl = NumpyList(self.lists[acc_counts[i]: acc_counts[i+1]])
        return npl

    def multi_is_built_from(self, tensor_def: TensorDefinitionMulti):
        """
        Method to validate that a Numpy list was likely built from a specific TensorDefinitionMultiHead object. The
        data types can not be checked. The checks mainly revolve around the sizes of the lists. This method supports
        multi-head.

        Args:
            tensor_def: The TensorDefinitionMultiHead object that needs to be checked.

        Returns:
            True of False. True if this Numpy List and TensorDefinition are compatible
        """
        # Split the Numpy list into the components from each TensorDefinition
        nps = [self.multi_filter(tensor_def, td) for td in tensor_def.tensor_definitions]
        r = [npl.is_built_from(td) for td, npl in zip(tensor_def.tensor_definitions, nps)]
        if False in r:
            return False
        else:
            return True
