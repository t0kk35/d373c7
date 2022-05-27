"""
Profile Numpy definitions. This profile is optimized for fast performance, using Numpy arrays and Numba.
(c) 2022 d373c7
"""
import logging
from typing import List

import numpy as np
import pandas as pd

from ..features.group import FeatureGrouper, TIME_PERIODS
from ..features.common import Feature

from typing import Tuple
from numba import jit

logger = logging.getLogger(__name__)


class ProfileNumpyException(Exception):
    def __init__(self, message: str):
        super().__init__('Error profiling: ' + message)


COUNT_INDEX = 0
MEAN_INDEX = 1
M2_INDEX = 2
MIN_INDEX = 3
MAX_INDEX = 4

ALL_AGGREGATORS = [
    COUNT_INDEX,
    MEAN_INDEX,
    M2_INDEX,
    MIN_INDEX,
    MAX_INDEX
]

NUMBER_OF_AGGREGATORS = len(ALL_AGGREGATORS)


class ProfileNumpy:
    def __init__(self, features: List[FeatureGrouper]):
        self._features = features
        tw_tp_ff_bf = [(f.time_window, f.time_period, f.filter_feature, f.base_feature) for f in features]
        u_tp_ff_bf = list(set([(tp, ff, bf) for _, tp, ff, bf in tw_tp_ff_bf]))
        self._array_shape = (len(u_tp_ff_bf), max([tw for tw, _, _, _ in tw_tp_ff_bf]), NUMBER_OF_AGGREGATORS)
        # A pre-built list of the aggregator indexes for each of the features of this profile
        self._aggregator_index = np.array([f.aggregator.key for f in self.features])
        # A pre-built list of the time_window for each of the features of this profile
        self._time_windows = np.array([f.time_window for f in self.features])
        # A pre-built ndarray of booleans that knows which base_feature needs to contribute to which element.
        self._base_feature_filters = np.empty((len(self.base_features), len(u_tp_ff_bf)), dtype=np.bool)
        for i, bf in enumerate(self.base_features):
            self._base_feature_filters[i] = np.isin(
                np.arange(self.array_shape[0]), [idx for idx, (_, _, ufb) in enumerate(u_tp_ff_bf) if ufb == bf]
            )
        # A pre-built array of booleans to filter out one specific feature from the self._array
        self._feature_filters = np.array([
            np.isin(
                np.arange(self.array_shape[0]), u_tp_ff_bf.index((f.time_period, f.filter_feature, f.base_feature))
            ) for f in features
        ])
        # A pre-built ndarray of ints that will keep track of which filter needs to be used for each element
        self._filter_index = np.array([
            self._find_index(ff, self.filter_features) for _, ff, _ in u_tp_ff_bf
        ])
        self._tp_filter = np.array([
            np.isin(np.arange(self.array_shape[0]), [idx for idx, (utp, _, _) in enumerate(u_tp_ff_bf) if utp == tp])
            for tp in TIME_PERIODS
        ])

    @property
    def array_shape(self) -> Tuple[int, int, int]:
        """
        Property which returns the shape of the Numpy array holding the profile element must have

        Returns:
            A Tuple of 3 ints. Containing the dimensions for the profile element Numpy array.
        """
        return self._array_shape

    def new_element_array(self) -> np.ndarray:
        return np.zeros(self.array_shape)

    @property
    def features(self) -> List[FeatureGrouper]:
        return self._features

    @property
    def feature_filters(self) -> np.ndarray:
        return self._feature_filters

    @property
    def base_filters(self) -> np.ndarray:
        return self._base_feature_filters

    @property
    def filter_indexes(self) -> np.ndarray:
        return self._filter_index

    @property
    def aggregator_indexes(self) -> np.ndarray:
        return self._aggregator_index

    @property
    def time_windows(self) -> np.ndarray:
        return self._time_windows

    @property
    def timeperiod_filters(self) -> np.ndarray:
        return self._tp_filter

    @property
    def base_features(self):
        lst = [f.base_feature for f in self.features]
        # Not using set here to make unique because we want to keep the order
        return list(dict.fromkeys(lst))

    @property
    def filter_features(self):
        lst = [f.filter_feature for f in self.features if f.filter_feature is not None]
        # Not using set here to make unique because we want to keep the order
        return list(dict.fromkeys(lst))

    @staticmethod
    def get_deltas(df: pd.DataFrame, time_feature: Feature) -> np.ndarray:
        """
        Helper method to get all the time deltas from between the rows. It is needed as input to some Numba
        jit-ed functions.
        It creates a Numpy of type int16 and shape (#rows_in_df X #time_periods). Each row contains the deltas
        between this row and the previous row for the time_periods. The columns are the time_periods, keyed by
        TimePeriod.key, so col[0] = TimePeriodDay, col[1] = TimePeriodWeek etc....

        Args:
            df (pd.DataFrame): A Pandas dataframe that contains a time_feature
            time_feature (Feature): The time feature in the Pandas

        Returns:
             A Numpy Array containing the deltas for all the TimePeriods.
        """
        # This is a bit of repetition of the TimePeriod logic. Turns out it takes a long time if you vectorize those
        # functions. This is a more Numpy like approach
        # If we only have one row then return zero's. The shifting does not work with one row.
        if len(df) == 1:
            return np.zeros((1, len(TIME_PERIODS)), dtype=np.int16)

        # Allocate output structure
        out = np.zeros((len(df), 3), dtype=np.int16)
        # Convert the time feature to a numpy array. This will contain datetime64 objects.
        npd = df[time_feature.name].to_numpy()
        # Day logic
        d = npd.astype('datetime64[D]')
        out[1:, 0] = d[1:] - d[:-1]
        # Week logic. The first line set w to be dayofweek starting at Monday (yeah 1970 was a Thursday.... Don't ask)
        w = d - ((d.view('int64') - 4) % 7)
        out[1:, 1] = (w[1:] - w[:-1]) // 7
        # Month logic
        m = npd.astype('datetime64[M]')
        out[1:, 2] = m[1:] - m[:-1]

        return out

    @staticmethod
    def _find_index(f: Feature, feature_list: List[Feature]) -> int:
        """
        Static Method which returns the index of a feature in a list. -1 is returned if the feature is not found.

        Args:
            f: (Feature). The feature to look up.
            feature_list: (List[Feature]). A list of features to use a lookup list.

        Returns:
            Index of f in feature_list
        """
        if f is None:
            return -1
        else:
            try:
                return feature_list.index(f)
            except ValueError:
                raise ProfileNumpyException(
                    f'Could not find index of {f.name} in list {[f.name for f in feature_list]}'
                )


# Some functions in the main body, these are numba jit-ed, as numba seems to not like class methods too much.
@jit(nopython=True, cache=True)
def profile_contrib(base_filters: np.ndarray, filter_index: np.ndarray, base_values: np.ndarray,
                    filter_values: np.ndarray, pe_array: np.ndarray) -> None:
    """
    Process a contribution for a profile-element. This will update all the aggregates of that specific profile element.
    This is a numba jit-ed function

    Args:
        base_filters (np.ndarray): A numpy array filter of type np.bool. Is an array that contains the filters for each
            base_feature in the profile. There is a row for each base filter, the columns on one specific row contain a
            filter that filters out the profile elements that should be updated by the respective base_feature.
            It has shape (#base_feature X #elements_in_profile)
            Values should be fetched with the `ProfileNumpy.base_filters` property
        filter_index (np.ndarray) : A numpy array that holds an index to the filter that needs to be applied to each
            element in the profile. Values should be fetched with the `ProfileNumpy.filter_indexes` property
        base_values (np.ndarray) : The contribution to add. Is a float64 ndarray with the values of the base features
            that need to contribute to this profile.
        filter_values (np.ndarray) : The values of the filter features [if any] of the elements.
        pe_array (np.ndarray) : The profile element array containing all the profiles. It is of shape
            (#elements X #max_time_window X #number_of_aggregators). Should be created with the
            `ProfileNumpy.new_element_array` method

    Returns:
        None
    """
    # Allocate a filter. This is a bool array with length #elements. It contains True if the respective element should
    # be updated because it's filter feature is either not set (it is -1) or evaluates to True
    flt_f = np.array([True for _ in filter_index])
    for i in range(filter_index.shape[0]):
        if filter_index[i] != -1 and filter_values[filter_index[i]] == 0:
            flt_f[i] = False

    # Iterate over the base features and contribute to the relevant elements.
    for i in range(base_filters.shape[0]):
        # Combine filters. This will now only update the elements with a specific base_feature AND it's filter is either
        # not set or True
        flt = base_filters[i] & flt_f
        # Count logic
        pe_array[flt, -1:, COUNT_INDEX] += 1
        # Calculate delta
        delta = base_values[i] - pe_array[flt, -1:, MEAN_INDEX]
        # Assign mean and M2
        pe_array[flt, -1:, MEAN_INDEX] += delta / pe_array[flt, -1:, COUNT_INDEX]
        pe_array[flt, -1:, M2_INDEX] += delta * (base_values[i] - pe_array[flt, -1:, MEAN_INDEX])
        # If count == 1, take contribution, else minimum of current minimum and contribution
        count_1_and_flt = (pe_array[:, -1:, COUNT_INDEX] == 1).reshape(flt.shape) & flt
        if np.any(count_1_and_flt):
            pe_array[count_1_and_flt, -1:, MIN_INDEX] = base_values[i]
        pe_array[flt, -1:, MIN_INDEX] = np.minimum(pe_array[flt, -1:, MIN_INDEX], base_values[i])
        # New max is max of previous max and contribution.
        pe_array[flt, -1:, MAX_INDEX] = np.maximum(pe_array[flt, -1:, MAX_INDEX], base_values[i])


@jit(nopython=True, cache=True)
def profile_aggregate(f_filter: np.ndarray, agg_index: np.ndarray, time_windows: np.ndarray,
                      pe_array: np.ndarray) -> np.ndarray:
    """
    Aggregate all the feature in a profile. This function will iterate over the features and will run the aggregation
    logic on the required time_period. This is a numba jit-ed function

    Args:
        f_filter (np.ndarray): A numpy array of type bool with shape (#group_features X #profile_elements). Each row
            contains a filter that can be used to select the correct profile element for that feature from the profile
            element array. Values should be fetched with the `ProfileNumpy.filter_features` property
        agg_index (np.ndarray): A numpy array of type int with shape (#group_features). Each row contains the `key`/id
            of an Aggregator object for a specific group feature of this profile. Values should be fetched with the
            `ProfileNumpy.aggregator_indexes` property
        time_windows (np.ndarray): A numpy array of type int with shape (#group_features). Each row the time window
            to be applied to a specific group feature of this profile. Values should be fetched with the
            `ProfileNumpy.time_windows` property
        pe_array (np.ndarray) : The profile element array containing all the profiles. It is of shape
            (#elements X #max_time_window X #number_of_aggregators). Should be created with the
            `ProfileNumpy.new_element_array` method

    Returns:
        Numpy array of shape (#group_features). It contains all the aggregated values for all the group features in the
            profile
    """
    out = np.zeros(f_filter.shape[0])
    for i in range(agg_index.shape[0]):
        # Sum Logic
        if agg_index[i] == 0:
            count = np.sum(pe_array[f_filter[i], -time_windows[i]:, COUNT_INDEX])
            sums = np.sum(
                pe_array[f_filter[i], -time_windows[i]:, MEAN_INDEX] *
                pe_array[f_filter[i], -time_windows[i]:, COUNT_INDEX]
            )
            if count != 0:
                mean = (sums / np.sum(pe_array[f_filter[i], -time_windows[i]:, COUNT_INDEX]))
            else:
                mean = 0.0
            out[i] = count * mean
        # Count Logic
        elif agg_index[i] == 1:
            out[i] = np.sum(pe_array[f_filter[i], -time_windows[i]:, COUNT_INDEX])
        # Min Logic
        elif agg_index[i] == 2:
            counts = np.reshape(pe_array[f_filter[i], -time_windows[i]:, COUNT_INDEX] > 0, (pe_array.shape[1],))
            minima = np.reshape(pe_array[f_filter[i], -time_windows[i]:, MIN_INDEX], (pe_array.shape[1],))
            out[i] = np.min(minima[counts])
        # Max Logic
        elif agg_index[i] == 3:
            out[i] = np.max(pe_array[f_filter[i], -time_windows[i]:, MAX_INDEX])
        # Mean Logic
        elif agg_index[i] == 4:
            sums = np.sum(
                pe_array[f_filter[i], -time_windows[i]:, MEAN_INDEX] *
                pe_array[f_filter[i], -time_windows[i]:, COUNT_INDEX]
            )
            counts = np.sum(pe_array[f_filter[i], -time_windows[i]:, COUNT_INDEX])
            if counts > 0:
                out[i] = sums / counts
            else:
                out[i] = 0
        # Stddev Logic
        elif agg_index[i] == 5:
            m2 = pe_array[f_filter[i], -1:, M2_INDEX]
            mean = pe_array[f_filter[i], -1:, MEAN_INDEX]
            count = pe_array[f_filter[i], -1:, COUNT_INDEX]
            for j in range(pe_array.shape[1]-2, pe_array.shape[1]-time_windows[i]-1, -1):
                c = count + pe_array[f_filter[i], j, COUNT_INDEX]
                delta = mean - pe_array[f_filter[i], j, MEAN_INDEX]
                delta2 = delta * delta
                mean = (
                    count * mean + pe_array[f_filter[i], j, COUNT_INDEX] *
                    pe_array[f_filter[i], j, MEAN_INDEX]
                ) / c
                m2 = m2 + pe_array[f_filter[i], j, M2_INDEX].item() + delta2 * (
                    count * pe_array[f_filter[i], j, COUNT_INDEX]
                ) / c
                count = c
            if count < 2:
                out[i] = 0
            else:
                out[i] = np.sqrt((m2/(count-1))).item()
        else:
            raise IndexError()
        # End of loop over features.
    return out


@jit(nopython=True, cache=True)
def profile_time_logic(tp_flt: np.ndarray, deltas: np.ndarray, pe_array: np.ndarray) -> None:
    """
    Run the Time logic on each of the profile elements. This is a numba jit-ed function

    Args:
        tp_flt (np.ndarray): An ndarray of type int. It has shape (#time_periods X #profile_element). It contains
            a row for each TimePeriod object. The row is a filter that filters out the elements using the respective
            TimePeriod object. Can be created with the `ProfileNumpy.timeperiod_filters` property
        deltas (np.ndarray): An ndarray of type int which should contain all the deltas between the current and the
            previous time. One value for each TimePeriod Object. Its shape is (#time_periods). The deltas
            can be created with the helper method `ProfileNumpy.get_deltas`.
        pe_array (np.ndarray) : The profile element array containing all the profiles. It is of shape
            (#elements X #max_time_window X #number_of_aggregators). Should be created with the
            `ProfileNumpy.new_element_array` method
    """
    for i in range(tp_flt.shape[0]):
        # Only run-time logic for the time-period if there is an element that uses it
        if np.any(tp_flt[i]):
            # If there is a delta shift up the time.
            if deltas[i] > 0:
                res = np.zeros_like(pe_array[tp_flt[i]])
                top = res.shape[1] - deltas[i]
                if top > 0:
                    res[:, :top] = pe_array[tp_flt[i], -top:]
                pe_array[tp_flt[i]] = res
