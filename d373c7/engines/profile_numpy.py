"""
Profile Numpy definitions
(c) 2022 d373c7
"""
import datetime as dt
import logging
from typing import List

import numpy as np

from ..features.group import FeatureGrouper, Aggregator, TimePeriod
from ..features.common import Feature
from ..features.expressions import FeatureFilter

from .profile import Profile, ProfileFieldBase, ProfileElement
from math import sqrt
from typing import Tuple, Callable, Optional

logger = logging.getLogger(__name__)


class ProfileNumpyException(Exception):
    def __init__(self, message: str):
        super().__init__('Error profiling: ' + message)


class ProfileElementNumpy(ProfileElement[Tuple[np.ndarray, dt.datetime, np.ndarray], np.ndarray]):
    count_index = 0
    mean_index = 1
    m2_index = 2
    min_index = 3
    max_index = 4

    def __init__(self, features: List[FeatureGrouper], base_order: List[Feature], filter_order: List[FeatureFilter]):
        # Create a numpy
        self._features = features
        tw_tp_ff_bf = [(f.time_window, f.time_period, f.filter_feature, f.base_feature) for f in features]
        u_tp_ff_bf = list(set([(tp, ff, bf) for _, tp, ff, bf in tw_tp_ff_bf]))
        self._array = np.zeros([len(u_tp_ff_bf), max([tw for tw, _, _, _ in tw_tp_ff_bf]), 5])
        self._tp_index: List[Tuple[TimePeriod, List[int]]] = [
            (tp, [idx for idx, (utp, _, _) in enumerate(u_tp_ff_bf) if utp == tp])
            for tp in list(set([tp for _, tp, _, _ in tw_tp_ff_bf]))
        ]
        self._bf_index: List[Tuple[int, List[int]]] = [
            (ProfileFieldNumpy.find_index(bf, base_order), [
                idx for idx, (_, _, ufb) in enumerate(u_tp_ff_bf) if ufb == bf
            ])
            for bf in list(set([bf for _, _, _, bf in tw_tp_ff_bf]))
        ]
        self._feature_index: List[int] = [
            u_tp_ff_bf.index((f.time_period, f.filter_feature, f.base_feature)) for f in features
        ]
        self._filter_order_index: List[int] = [
            ProfileFieldNumpy.find_index(ff, filter_order) for _, ff, _ in u_tp_ff_bf
        ]

    def merge(self, pe: 'ProfileElementNumpy'):
        count = self.array[:, :, self.count_index] + pe.array[:, :, self.count_index]
        delta = self.array[:, :, self.mean_index] - pe.array[:, :, self.min_index]
        delta2 = delta * delta
        self.array[:, :, self.mean_index] = (
                self.array[:, :, self.count_index] * self.array[:, :, self.mean_index] +
                pe.array[:, :, self.count_index] * pe.array[:, :, self.mean_index]
        ) / count
        self.array[:, :, self.m2_index] = self.array[:, :, self.m2_index] + pe.array[:, :, self.m2_index] + delta2 * (
                self.array[:, :, self.count_index] * pe.array[:, :, self.count_index]
        ) / count
        self.array[:, :, self.count_index] = count
        self.array[:, :, self.max_index] = np.maximum(
           self.array[:, :, self.max_index], pe.array[:, :, self.max_index]
        )
        # Update minimum only if count is > 0
        self.array[:, :, self.min_index][pe.array[:, :, self.count_index] > 0] = np.minimum(
            self.array[:, :, self.min_index], pe.array[:, :, self.min_index]
        )

    def contribute(self, contribution: Tuple[np.ndarray, dt.datetime, np.ndarray]):
        for bfi, ind in self._bf_index:
            flt_b = np.isin(np.arange(self._array.shape[0]), ind)
            # Select the last row of the time dimension (current timeperiod)
            # And make sure to only update entries where the filter is true or undefined.
            flt_f = np.array([1 if i == -1 else contribution[2][i] for i in self._filter_order_index]).astype(bool)
            flt = flt_b & flt_f
            # Add one to count
            self._array[flt, -1:, self.count_index] += 1
            # Calculate delta
            delta = contribution[0][bfi] - self._array[flt, -1:, self.mean_index]
            # Assign mean and M2
            self._array[flt, -1:, self.mean_index] += delta / self._array[flt, -1:, self.count_index]
            self._array[flt, -1:, self.m2_index] += delta * (
                    contribution[0][bfi] - self._array[flt, -1:, self.mean_index]
            )
            # If count == 1, take contribution, else minimum of current minimum and contribution
            count_1_and_flt = np.squeeze(self._array[:, -1:, self.count_index] == 1) & flt
            count_n_1_and_flt = np.squeeze(self._array[:, -1:, self.count_index] > 1) & flt
            self._array[:, -1:, self.min_index][count_1_and_flt] = contribution[0][bfi]
            self._array[:, -1:, self.min_index][count_n_1_and_flt] = np.minimum(
                self._array[:, -1:, self.min_index], contribution[0][bfi]
            )[count_n_1_and_flt]
            # New max is max of previous max and contribution.
            self._array[flt, -1:, self.max_index] = np.maximum(
                self._array[flt, -1:, self.max_index], contribution[0][bfi]
            )

    def aggregate(self, feature: Optional[FeatureGrouper]) -> np.ndarray:
        return np.array(
            [PROFILE_AGG_HELPER.aggregate(
                self, self._feature_index[i], f.time_window, f.aggregator
            ) for i, f in enumerate(self._features)]
        )

    def run_time_logic(self, previous_time: dt.datetime, current_time: dt.datetime):
        # Shift up the time periods. Iterate over each timeperiod. Select the indexes that use this tp
        for tp, ind in self._tp_index:
            delta = tp.delta_between(tp.start_period(previous_time), tp.start_period(current_time))
            # If there is a delta shift up the time.
            if delta > 0:
                self._array[ind] = np.pad(
                    self._array[ind], ((0, 0), (0, delta), (0, 0)), mode='constant'
                )[:, delta:]

    @property
    def array(self) -> np.ndarray:
        return self._array


class ProfileAggregatorHelper:
    def __init__(self):
        self._functions: List[Callable[[ProfileElementNumpy, int, int], float]] = [
            self._sum,
            self._count,
            self._min,
            self._max,
            self._mean,
            self._stddev
        ]
        self._number_of_functions = len(self._functions)

    def aggregate(self, pe: ProfileElementNumpy, element_index: int, time_window: int, aggregator: Aggregator) -> float:
        if aggregator.key > self._number_of_functions:
            raise ProfileNumpyException(
                f'Risk of overflow, looking for aggregator with Key: {aggregator.key}. Probably using an unknown' +
                f' aggregator'
            )
        return self._functions[aggregator.key](pe, element_index, time_window)

    @staticmethod
    def _count(pe: ProfileElementNumpy, ind: int, time_window: int) -> float:
        return np.sum(pe.array[ind, -time_window:, ProfileElementNumpy.count_index]).item()

    @staticmethod
    def _sum(pe: ProfileElementNumpy, ind: int, time_window: int) -> float:
        return PROFILE_AGG_HELPER._count(pe, ind, time_window) * PROFILE_AGG_HELPER._mean(pe, ind, time_window)

    @staticmethod
    def _mean(pe: ProfileElementNumpy, ind: int, time_window: int) -> float:
        sums = np.sum(
            pe.array[ind, -time_window:, ProfileElementNumpy.mean_index] *
            pe.array[ind, -time_window:, ProfileElementNumpy.count_index]
        )
        if np.count_nonzero(sums) > 0:
            return (sums / np.sum(pe.array[ind, -time_window:, ProfileElementNumpy.count_index])).item()
        else:
            return 0.0

    @staticmethod
    def _stddev(pe: ProfileElementNumpy, ind: int, time_window: int) -> float:
        m2 = pe.array[ind, -1:, ProfileElementNumpy.m2_index].item()
        mean = pe.array[ind, -1:, ProfileElementNumpy.mean_index].item()
        count = pe.array[ind, -1:, ProfileElementNumpy.count_index].item()
        for i in range(pe.array.shape[1]-2, pe.array.shape[1]-time_window-1, -1):
            c = count + pe.array[ind, i, ProfileElementNumpy.count_index].item()
            delta = mean - pe.array[ind, i, ProfileElementNumpy.mean_index].item()
            delta2 = delta * delta
            mean = (
                    count * mean + pe.array[ind, i, ProfileElementNumpy.count_index].item() *
                    pe.array[ind, i, ProfileElementNumpy.mean_index].item()
            ) / c
            m2 = m2 + pe.array[ind, i, ProfileElementNumpy.m2_index].item() + delta2 * (
                    count * pe.array[ind, i, ProfileElementNumpy.count_index].item()
            ) / c
            count = c
        if count < 2:
            return 0
        else:
            return sqrt(m2/(count-1))

    @staticmethod
    def _min(pe: ProfileElementNumpy, ind: int, time_window: int) -> float:
        # Only take row with an actual count into consideration
        r = pe.array[ind, pe.array[ind, :, ProfileElementNumpy.count_index] > 0]
        # Now select the timeperiod, only the min_index and take the actual min
        return np.min(r[-(min(time_window, r.shape[0])), ProfileElementNumpy.min_index])

    @staticmethod
    def _max(pe: ProfileElementNumpy, ind: int, time_window: int) -> float:
        return np.max(pe.array[ind, -time_window:, ProfileElementNumpy.max_index])


PROFILE_AGG_HELPER = ProfileAggregatorHelper()


class ProfileFieldNumpy(ProfileFieldBase[Tuple[np.ndarray, dt.datetime, np.ndarray], np.ndarray]):
    def __init__(
            self, features: List[FeatureGrouper], base_order: List[Feature],
            filter_order: List[FeatureFilter]):
        super(ProfileFieldNumpy, self).__init__()
        self._element = ProfileElementNumpy(features, base_order, filter_order)

    def run_time_logic(self, current_time: dt.datetime):
        if self.start_period_time is None:
            self.start_period_time = current_time
        self._element.run_time_logic(self.start_period_time, current_time)
        self.start_period_time = current_time

    def contribute(self, contribution: Tuple[np.ndarray, dt.datetime, np.ndarray]):
        self._element.contribute(contribution)

    @property
    def element(self) -> ProfileElementNumpy:
        return self._element


# This class gets a tuple of arrays. They will contain the amount, the time, the filters.
class ProfileNumpy(Profile[Tuple[np.ndarray, dt.datetime, np.ndarray], np.ndarray]):
    def __init__(self, features: List[FeatureGrouper]):
        super(ProfileNumpy, self).__init__(features)
        self._profile_fields: List[ProfileFieldNumpy] = [
            ProfileFieldNumpy(features, self.base_features, self.filter_features)
        ]

    def extract_date_time(self, contribution: Tuple[np.ndarray, dt.datetime, np.ndarray]) -> dt.datetime:
        # Date should be the second element
        return contribution[1]

    @property
    def profile_fields(self) -> List[ProfileFieldNumpy]:
        return self._profile_fields

    def list(self, contribution: Tuple[np.ndarray, dt.datetime, np.ndarray]) -> np.ndarray:
        return self._profile_fields[0].element.aggregate(None)
