"""
Definition of the panda numpy engine. This engine will mainly read files into pandas and then potentially convert
them to numpy array for modelling/processing.
(c) 2020 d373c7
"""
import logging
import pathlib
import datetime as dt
import pandas as pd
import numpy as np
import multiprocessing as mp
import numba as nb
from numba import jit
from functools import partial
from itertools import groupby
from collections import OrderedDict
from typing import Dict, List, Callable, Type, Tuple, Optional, Union
from .common import EngineContext
from .profile_numpy import ProfileNumpy, ProfileNumpyStore, profile_aggregate, profile_time_logic, profile_contrib
from .numpy_helper import NumpyList
from ..features.common import Feature, FeatureTypeTimeBased, FEATURE_TYPE_CATEGORICAL
from ..features.common import FeatureTypeInteger, FeatureHelper
from ..features.common import LearningCategory, LEARNING_CATEGORIES_MODEL_INPUT, LEARNING_CATEGORY_LABEL
from ..features.base import FeatureSource, FeatureIndex, FeatureBin, FeatureRatio, FeatureConcat
from ..features.tensor import TensorDefinition, TensorDefinitionMulti
from ..features.expanders import FeatureExpander, FeatureOneHot
from ..features.normalizers import FeatureNormalize, FeatureNormalizeScale, FeatureNormalizeStandard
from ..features.normalizers import FeatureNormalizeLogBase
from ..features.expressions import FeatureExpression, FeatureExpressionSeries
from ..features.labels import FeatureLabel, FeatureLabelBinary
from ..features.group import FeatureGrouper, TimePeriod
from ..network.network_pandas import NetworkDefinitionPandas, NetworkNodeDefinitionPandas, NetworkEdgeDefinitionPandas

logger = logging.getLogger(__name__)


class EnginePandaNumpyException(Exception):
    def __init__(self, message: str):
        super().__init__('Error creating Panda source: ' + message)


# Panda types dictionary. Values are Tuples, the first entry in the type used for reading, the second for converting
PandaTypes: Dict[str, Tuple[str, Type]] = {
    'STRING': ('object', object),
    'FLOAT': ('float64', np.float64),
    'FLOAT_32': ('float32', np.float32),
    'INTEGER': ('int32', np.int32),
    'INT_8': ('int8', np.int8),
    'INT_16': ('int16', np.int16),
    'INT_64': ('int64', np.int64),
    'DATE': ('str', pd.Timestamp),
    'CATEGORICAL': ('category', pd.Categorical),
    'DATETIME': ('str', pd.Timestamp)
}


class EnginePandasNumpy(EngineContext):
    """
    Panda and Numpy engine. It's main function is to take build Panda and Numpy structures from given tensor definition.

    Args:
        num_threads (int): The maximum number of thread the engine will use during multiprocess processing

    Attributes:
        one_hot_prefix: The standard pre_fix to create one hot features.
    """
    def __init__(self, num_threads: int = 5):
        EngineContext.__init__(self)
        logger.info(f'Pandas Version : {pd.__version__}')
        logger.info(f'Numpy Version : {np.__version__}')
        self._num_threads = num_threads
        self.one_hot_prefix = '__'

    def _val_features_in_data_frame(self, df: pd.DataFrame, tensor_def: TensorDefinition):
        """Validation function to check if all features in a tensor definition are known data frame columns

        @param df: A Panda Dataframe
        @param tensor_def: A tensor definition
        @return: None
        """
        for feature in tensor_def.features:
            if isinstance(feature, FeatureExpander):
                names = [name for name in df.columns
                         if name.startswith(feature.base_feature.name + self.one_hot_prefix)]
            else:
                names = [name for name in df.columns if name == feature.name]
            if len(names) == 0:
                raise EnginePandaNumpyException(
                    f'During reshape, all features of tensor definition must be in the panda. Missing {feature.name}'
                )

    @staticmethod
    def _val_single_date_format_code(format_codes: List[str]):
        """Validation function to check that there is a 01_single format code for dates across a set of format codes.

        @param format_codes: List of format codes from feature definitions. Is a string
        @return: None
        """
        if len(format_codes) > 1:
            raise EnginePandaNumpyException(f'All date formats should be the same. Got {format_codes}')

    @staticmethod
    def _val_ready_for_inference(tensor_def: TensorDefinition, inference: bool):
        """Validation function to check if all feature are ready for inference. Some features have specific inference
        attributes that need to be set before an inference file can be made.

        @param tensor_def: The tensor that needs to be ready for inference.
        @param inference: Indication if we are inference mode or not.
        @return: None
        """
        if inference:
            if not tensor_def.inference_ready:
                raise EnginePandaNumpyException(
                    f'Tensor <{tensor_def.name}> not ready for inference. Following features not ready ' +
                    f' {tensor_def.features_not_inference_ready()}'
                )

#    @staticmethod
#    def _val_not_ready_for_inference(tensor_def: TensorDefinition, inference: bool):
#        """Validation function to check if any of the feature are ready for inference. Some features have specific
#        inference attributes that would get overwritten if .

#        @param tensor_def: The tensor that needs to be ready for inference.
#        @param inference: Indication if we are inference mode or not.
#        @return: None
#        """
#        if not inference:
#            if tensor_def.inference_ready:
#                raise EnginePandaNumpyException(
#                    f'Tensor <{tensor_def.name}> is ready for inference. Running with inference again would' +
#                    f' erase the inference attributes. Following features are ready ' +
#                    f' {tensor_def.features_not_inference_ready()}'
#                )

    @staticmethod
    def _val_features_defined_as_columns(df: pd.DataFrame, features: List[Feature]):
        """Validation function that checks if the needed columns are available in the Panda. Only root features which
        are not derived from other features need to be in the Panda. The rest of the features can obviously be
        derived from the root features.

        @param df: The base Panda data to be checked.
        @param features: List of feature to check.
        @return: None
        """
        root_features = [f for f in features if len(f.embedded_features) == 0]
        unknown_features = [f for f in root_features if f.name not in df.columns]
        if len(unknown_features) != 0:
            raise EnginePandaNumpyException(
                f'All root features of a tensor definition (i.e. non-derived features) must be in the input df. Did '
                f'not find {[f.name for f in unknown_features]}'
            )

    # @staticmethod
    # def _val_key_feature_available(tensor_definition: TensorDefinition, key_feature: Feature):
    #     if key_feature not in tensor_definition.features:
    #         raise EnginePandaNumpyException(
    #             f'The key feature used to build a series must be in the tensor_definition'
    #         )

    @staticmethod
    def _val_time_feature_available(tensor_definition: TensorDefinition, time_feature: Feature):
        if not isinstance(time_feature.type, FeatureTypeTimeBased):
            raise EnginePandaNumpyException(
                f'The time feature used to build a series must be date based. It is of type {time_feature.type}'
            )

        if time_feature not in tensor_definition.features:
            raise EnginePandaNumpyException(
                f'The time feature used to build a series must be in the data frame tensor_definition'
            )

    @staticmethod
    def _val_time_feature_needed(target_tensor_def: TensorDefinition, time_feature: Feature):
        if len(FeatureHelper.filter_feature(FeatureGrouper, target_tensor_def.embedded_features)) > 0:
            if time_feature is None:
                raise EnginePandaNumpyException(
                    f'There is a FeatureGrouper in the Tensor Definition to create. They need a time field to ' +
                    f' process. Please provide the parameter ''time_feature''.'
                )
            else:
                if not FeatureHelper.is_feature_of_type(time_feature, FeatureTypeTimeBased):
                    raise EnginePandaNumpyException(
                        f'The time feature used to build a series must be date based. It is of type {time_feature.type}'
                    )

    @staticmethod
    def _val_grouper_based(
            target_tensor_def: TensorDefinition) -> List[Tuple[FeatureGrouper, Optional[FeatureNormalize]]]:
        """
        Function that will validate that the target tensor definition contains only FeatureGroupers or Normalizer
        features with a FeatureGrouper as base feature.

        Args:
            target_tensor_def: The tensor definition to check

        Returns: List[Tuple[FeatureGrouper, Optional[FeatureNormalize]]]: List that contains all the feature groupers
        in the target_tensor def, either directly or as base_feature to a normalizer.
        """
        out: List[Tuple[FeatureGrouper, Optional[FeatureNormalize]]] = []
        for f in target_tensor_def.features:
            fg = FeatureHelper.filter_feature(FeatureGrouper, [f])
            if not len(fg) > 0:
                fn = FeatureHelper.filter_feature(FeatureNormalize, [f])
                if len(fn) > 0:
                    fge = FeatureHelper.filter_feature(FeatureGrouper, [fn[0].base_feature])
                    if not len(fge) > 0:
                        raise EnginePandaNumpyException(
                            f'The Target Tensor Definition should only contain FeatureGrouper features and ' +
                            f'FeatureNormalize with a FeatureGrouper as base_feature. Got {f.name} of type {type(f)}'
                        )
                    else:
                        out.append((fge[0], fn[0]))
            else:
                out.append((fg[0], None))

        return out

    @staticmethod
    def _val_all_same_type(features: List[Tuple[FeatureGrouper, Optional[FeatureNormalize]]]) -> Type:
        types = [gf.type for gf, _ in features]
        types.extend([nf.type for _, nf in features if nf is not None])
        types = list(set(types))
        if len(types) > 1:
            raise EnginePandaNumpyException(
                f'All features should have had the same type. Found types: {types}'
            )
        return EnginePandasNumpy.panda_type(features[0][0], None, False)

    @staticmethod
    def _val_all_network_dfs_same_length(network: NetworkDefinitionPandas) -> None:
        dfs = [len(nl.node_list) for nl in network.node_definition_list] + \
              [len(el.edge_list) for el in network.edge_definition_list]
        lens = list(set(dfs))
        if len(lens) > 1:
            raise EnginePandaNumpyException(
                f'The length of the dataframes of all Nodes and Definitions should have been the same. Found ' +
                f'lengths {lens}'
            )

    @property
    def num_threads(self):
        return self._num_threads if self._num_threads is not None else int(mp.cpu_count() * 0.8)

    @staticmethod
    def panda_type(feature: Feature, default: str = None, read: bool = True) -> Union[str, Type]:
        """
        Helper function that determines the panda (and numpy) data types for a specific feature. Base on the f_type

        @param feature: (Feature) A feature definition
        @param default: (string) A default data type
        @param read: (bool). Flag indicating whether to return the 'read' type or the 'interpret' type.
        Default is 'read'
        @return: (Union[str, Type] Panda (Numpy) data type as a string or object depending on the read parameter
        """
        if feature is None:
            return default
        panda_type = PandaTypes.get(feature.type.name, (default, default))
        if panda_type is None:
            raise EnginePandaNumpyException(f'Did not find panda type for {feature.name}')
        else:
            if read:
                return panda_type[0]
            else:
                return panda_type[1]

    @staticmethod
    def _parse_dates(dates: List[str], format_code: str) -> List[dt.datetime]:
        """
        Helper function to parse datetime structures from strings

        @param dates: A list of dates to parse (as string)
        @param format_code: The format code to apply
        @return: List of datetime type values
        """
        return [dt.datetime.strptime(d, format_code) for d in dates]

    def _set_up_date_parser(self, date_features: List[FeatureSource]) -> Optional[partial]:
        """
        Helper function to which sets-up a data parser for a specific format. The date parser is used by the pandas
        read_csv function.

        @param date_features: (List[FeaturesSource]) the Features of type date which need to be read
        @return: (Optional[partial]). A function (the date parser) or none if there were no explicitly defined formats
        """
        if len(date_features) != 0:
            format_codes = list(set([d.format_code for d in date_features]))
            self._val_single_date_format_code(format_codes)
            return partial(self._parse_dates, format_code=format_codes[0])
        else:
            return None

    def from_df(self, target_tensor_def: TensorDefinition, df: pd.DataFrame, df_tensor_def: TensorDefinition,
                inference: bool, time_feature: Optional[Feature] = None) -> pd.DataFrame:
        """
        Construct a Panda according to a tensor definition from another Panda. This is useful to construct derived
        features. One can first read the panda with from_csv to get the source features and then run this function to
        build all derived features

        Same as from_df but internal only. Should not be called externally.

        @param target_tensor_def: The target TensorDefinition to construct.
        @param df: The input pandas DataFrame to re-construct
        @param df_tensor_def: The TensorDefinition used to construct the pandas DataFrame
        @param inference: Indicate if we are inferring or not. If True [COMPLETE]
        @param time_feature: (Feature) Optional Parameter. If processing requires a time field to be known (depends on
        features the method needs to create)
        @return: A Panda with the fields as defined in the tensor_def.
        """
        logger.info(f'Building Panda for : <{target_tensor_def.name}> from DataFrame. Inference mode <{inference}>')
        self._val_ready_for_inference(df_tensor_def, inference)
        self._val_time_feature_needed(target_tensor_def, time_feature)
        self._val_features_defined_as_columns(df, target_tensor_def.embedded_features)
        # Start processing Use the FeatureProcessor class.
        df = _FeatureProcessor.process(
            df, target_tensor_def.features, inference, self.one_hot_prefix, self.num_threads, time_feature
        )
        # Only return base features in the target_tensor_definition. No need to return the embedded features.
        # Remember that expander features can contain multiple columns.
        df = self.reshape(target_tensor_def, df)
        # Don't forget to set the Tensor definition rank if in inference mode
        if not inference:
            target_tensor_def.rank = len(df.shape)
        logger.info(f'Done creating {target_tensor_def.name}. Shape={df.shape}')
        return df

    def from_csv(self, target_tensor_def: TensorDefinition, file: str, delimiter: chr = ',', quote: chr = "'",
                 time_feature: Optional[Feature] = None, inference: bool = True) -> pd.DataFrame:

        """
        Construct a Panda according to a tensor definition by reading a csv file.

        Args:
            target_tensor_def: (TensorDefinition) The input tensor definition
            file: (str) File to read. This must be a complete file path
            delimiter: (chr) The delimiter used in the file. Default is ','
            quote: (chr) Quote character. Default is "'"
            time_feature (Feature). Optional. Feature to use for time-based calculations. Some features need to know
                about the time such as for instance Grouper features. Only needs to be provided if the target_tensor_def
                contains features that need time.
            inference: (bool) Indicate if we are inferring or not. If True [COMPLETE]
        Returns:
            A Panda with the fields as defined in the tensor_def.
        """
        # TODO probably need to check that if not inference, that all feature are ready for inference.
        self._val_time_feature_needed(target_tensor_def, time_feature)
        # Start by reading the SourceFeatures. Set to correct Panda Type
        file_instance = pathlib.Path(file)
        if not file_instance.exists():
            raise EnginePandaNumpyException(f' path {file} does not exist or is not a file')
        logger.info(f'Building Panda for : {target_tensor_def.name} from file {file}')
        need_to_build = target_tensor_def.embedded_features
        # Make sure to also build the time feature and stuff it needs
        if time_feature is not None and time_feature not in need_to_build:
            need_to_build.append(time_feature)
            need_to_build.extend(time_feature.embedded_features)
        source_features = FeatureHelper.filter_feature(FeatureSource, need_to_build)
        source_feature_names = [field.name for field in source_features]
        source_feature_types = {
            feature.name: EnginePandasNumpy.panda_type(feature, read=True) for feature in source_features
        }
        date_features: List[FeatureSource] = FeatureHelper.filter_feature_type(FeatureTypeTimeBased, source_features)
        date_feature_names = [f.name for f in date_features]
        # Set up some specifics for the date/time parsing
        date_parser = self._set_up_date_parser(date_features)
        infer_datetime_format = True if date_parser is None else True

        df = pd.read_csv(
            file,
            sep=delimiter,
            usecols=source_feature_names,
            dtype=source_feature_types,
            quotechar=quote,
            parse_dates=date_feature_names,
            date_parser=date_parser,
            infer_datetime_format=infer_datetime_format
        )

        built_features = list(set(source_features + date_features))
        td = TensorDefinition(f'Built Features', built_features)
        df = self.from_df(td, df, td, inference=inference, time_feature=time_feature)
        need_to_build = [f for f in need_to_build if f not in built_features]
        # Repeatedly call from_df, each time building more features as the embedded (dependent) features have been built
        # Note that the _FeatureProcessor has side effects. It changes the original df, for performance reasons. In
        # order to avoid repeated concatenate and stuff.
        i = 1
        while len(need_to_build) > 0:
            if i > 20:
                raise EnginePandaNumpyException(
                    f'Exiting. Did more that {i} iterations trying to build {target_tensor_def.name}.' +
                    f'Potential endless loop.'
                )
            ready_to_build = [f for f in need_to_build if all(ef in built_features for ef in f.embedded_features)]
            ready_to_build = list(set(ready_to_build))
            # Start processing Use the FeatureProcessor class.
            df = _FeatureProcessor.process(
                df, ready_to_build, inference, self.one_hot_prefix, self.num_threads, time_feature
            )
            built_features = built_features + ready_to_build
            td = TensorDefinition(f'Built Features', built_features)
            # Make sure df is in the correct layout.
            df = self.reshape(td, df)
            need_to_build = [f for f in need_to_build if f not in built_features]
            i = i+1

        # Reshape df so that it matches the target_tensor_def
        df = self.reshape(target_tensor_def, df)
        # Don't forget to set the Tensor definition rank if in inference mode
        if not inference:
            target_tensor_def.rank = len(df.shape)
        return df

    def reshape(self, tensor_def: TensorDefinition, df: pd.DataFrame):
        """
        Reshape function. Can be used to reshuffle the columns in a Panda. The columns will be returned according to
        the exact order as the features of the tensor definition. Columns that are not in the tensor definition as
        feature will be dropped.

        Args:
            df: Input Panda.
            tensor_def: The tensor definition according which to reshape
        Returns:
            A panda with the columns as defined in tensor_def
        """
        logger.info(f'Reshaping DataFrame to: {tensor_def.name}')
        self._val_features_in_data_frame(df, tensor_def)
        col_names = []
        for feature in tensor_def.features:
            if isinstance(feature, FeatureExpander):
                col_names.extend(
                    [name for name in df.columns
                     if name.startswith(feature.base_feature.name + self.one_hot_prefix)]
                )
            else:
                col_names.append(feature.name)
        df = df[[name for name in col_names]]
        return df

    def to_numpy_list(self, tensor_def: TensorDefinition, df: pd.DataFrame) -> NumpyList:
        """Method to convert a Pandas Dataframe to an object of type NumpyList. A NumpyList is an object which contains
        multiple Numpy arrays. The Numpy array can be turned into Pytorch Tensors.

        @param tensor_def: The TensorDefinition Object used to create the Pandas DataFrame.
        @param df: The Pandas DataFrame to convert to a NumpyList object.
        @return: A NumpyList Object containing Numpy arrays for the data in the Pandas DataFrame.
        """
        self._val_features_defined_as_columns(df, tensor_def.features)

        n = []
        for lc in tensor_def.learning_categories:
            td = TensorDefinition(lc.name, tensor_def.filter_features(lc))
            hf = td.highest_precision_feature
            d_type = self.panda_type(hf)
            logger.info(f'Converting DataFrame to Numpy of type: {d_type}')
            tdf = self.reshape(td, df)
            npy = tdf.to_numpy(dtype=d_type)
            # Squeeze off last dim if is size is 1
            if len(npy.shape) > 0 and npy.shape[-1] == 1:
                npy = np.squeeze(npy, axis=len(npy.shape)-1)
            n.append(npy)

        npl = NumpyList(n)
        # Don't forget to set the shapes.
        tensor_def.shapes = [(-1, *s[1:]) for s in npl.shapes]
        return npl

    def multi_to_numpy_list(self, tensor_def: TensorDefinitionMulti, df: [pd.DataFrame]) -> NumpyList:
        """Method to convert a Pandas Dataframe to an object of type NumpyList. A NumpyList is an object which contains
        multiple Numpy arrays. The Numpy array can be turned into Pytorch Tensors. Other than the regular to_numpy
        method, this method supports multi-head input, it takes a list of TensorDefinitions and DataFrames.

        @param tensor_def: The TensorDefinitionMulti object used to create the Pandas DataFrame.
        @param df: The *list* of Pandas DataFrames to convert to a NumpyList object.
        @return: A NumpyList Object containing Numpy arrays for the data in the Pandas DataFrame.
        """
        npl = [self.to_numpy_list(td, df) for td, df in zip(tensor_def.tensor_definitions, df)]
        npl = [lst for n in npl for lst in n.lists]
        npl = NumpyList(npl)
        return npl

    @staticmethod
    def _process_key_stacked(rows: pd.DataFrame, time_field: Feature,
                             lc_features: List[Tuple[LearningCategory, List[Feature], str]], window: int):

        # First sort rows on time_field. We want to go up in time as we process.
        rows.sort_values(by=[time_field.name], ascending=True, inplace=True)
        # Keep the original index. We'll need it to restore de order of the input data. (Might not be ordered by time)
        indexes = rows.index

        # Enrich the series. Run the FeatureSeriesExpression logic. Note this is a list of lists which we flatten
        sf: List[FeatureExpressionSeries] = [
            f for fs in [
                FeatureHelper.filter_feature(FeatureExpressionSeries, f_lst) for _, f_lst, _ in lc_features
            ] for f in fs
        ]
        for f in sf:
            t = EnginePandasNumpy.panda_type(f)
            rows[f.name] = f.expression(rows[[p.name for p in f.param_features]]).astype(t)

        # Convert everything to numpy for performance. This creates a numpy per each LC, with all feature of that LC.
        np_series = [rows[[f.name for f in f_lst]].to_numpy(np_type) for _, f_lst, np_type in lc_features]
        # np_series = [ns for ns in np_series if ns.shape[1] != 0]

        def process_row(i: int):
            x = indexes[i]
            # Select the listed fields by name
            s = [ns[max(0, i - window + 1):i + 1] for ns in np_series]
            # Pad if incomplete. I.e. There were less than length rows before this row.
            s = [np.concatenate((np.zeros((window - e.shape[0], e.shape[1]), dtype=e.dtype), e))
                 if e.shape[0] < window else e for e in s]
            # Return a list of numpy arrays, the first one is the original index
            return [x] + s

        # Process all the rows. Return a padded Numpy array of the correct length.
        lists = [process_row(i) for i in range(len(rows))]
        return lists

    def to_series_stacked(self, target_tensor_def: TensorDefinition, file: str, key_feature: Feature,
                          time_feature: Feature, window: int, delimiter: chr = ',', quote: chr = "'",
                          inference: bool = True) -> NumpyList:
        """
        Method that will create a stacked series. It will 'group' the features (for instance per customer),
        order according to a date-time field and create a sliding time window. The length of this window is a parameter.
        The output is the same number of records as the input file, but each transaction will be pre-pended with the
        previous x (depending on the length) transactions.
        As input, it takes a file, and it returns a 3D tensor per learning category.

        Args:
            target_tensor_def: Tensor Definition to use for the series creation. The features that should be stacked.
            file: The file name we want to turn into a stacked sequence
            key_feature: The feature to use as key. This is the feature that will be used for grouping
            time_feature: The feature to use as date-time field. This field will be used to order the transactions
            window: Requested size for sliding window, defined how many transaction will be pre-pending to the
                current transaction. (I.e. Length -1)
            delimiter: The delimiter used in the file. Default is ','
            quote: Quote character. Default is "'"
            inference: Indicate if we are inferring or not. If True [COMPLETE]
        Returns:
            NumpyList Object. It will contain one list per Learning Category. The lists will be 3D Tensors.
                (Batch x Series-length x Number-of-features-for the LC)
        """
        # TODO Need to check if ready for inference
        # self._val_ready_for_inference(target_tensor_def, inference)
        # Series Expressions are built as we stack. They need to be built in the _process_key_stacked function
        f_features = FeatureHelper.filter_not_feature(FeatureExpressionSeries, target_tensor_def.features)
        # Add all parameter features
        f_features.extend([
            p for f in FeatureHelper.filter_feature(FeatureExpressionSeries, target_tensor_def.features)
            for p in f.param_features
        ])
        f_features.extend([key_feature, time_feature])
        # Create an 'unstacked' dataframe of the features. Do this first so it is ready for inference.
        td = TensorDefinition('InternalKeyTime', list(set(f_features)))
        df = self.from_csv(td, file, delimiter, quote, inference=inference)

        # Make List of Tuple[LearningCategory, List[Feature]], PandaType
        # Do this once, so we do not need to run this per each customer.
        lc_features = []
        for lc in LEARNING_CATEGORIES_MODEL_INPUT:
            f = target_tensor_def.filter_features(lc, True)
            t = TensorDefinition('dummy', f)
            hpf = t.highest_precision_feature if len(t) > 0 else None
            pt = self.panda_type(hpf, lc.default_panda_type)
            if len(t.features) > 0:
                lc_features.append((lc, f, pt))
        # Sort by LC. We always want the same order for the LCs. The models will depend on it.
        lc_features.sort(key=lambda x: x[0])

        # Get the label Feature and type
        l_feature = target_tensor_def.label_features(True)
        if len(l_feature) > 0:
            l_td = TensorDefinition('dummy_label', l_feature)
            hpl = l_td.highest_precision_feature
            l_type = self.panda_type(hpl, LEARNING_CATEGORY_LABEL.default_panda_type)
        else:
            l_type = None

        # Create partial function for the
        key_function = partial(
            self._process_key_stacked,
            time_field=time_feature,
            lc_features=lc_features,
            window=window
        )

        logger.info(f'Start creating stacked series for Target Tensor Definition <{target_tensor_def.name}> ' +
                    f'using {self.num_threads} process(es)')

        # Now stack the data....
        with mp.Pool(self.num_threads) as p:
            series = p.map(key_function, [rows for _, rows in df.groupby(key_feature.name)])

        series = [s for keys in series for s in keys]
        # Need to sort to get back in the order of the index
        series.sort(key=lambda x: x[0])
        series = [np.array(s) for s in list(zip(*series))[1:]]
        # Add the label(s). If there is one. It's not required to have a label.
        if len(l_feature) != 0:
            labels = df[[f.name for f in l_feature]].to_numpy().astype(l_type)
            series.append(labels)
        logger.info(f'Returning series of types {[str(s.dtype) for s in series]}.')
        # Turn it into a NumpyList
        series = NumpyList(series)
        # Don't forget to set the Rank and shape
        target_tensor_def.rank = 3
        target_tensor_def.shapes = [(-1, *s[1:]) for s in series.shapes]
        logger.info(f'Done creating {target_tensor_def.name}. Shapes={series.shapes}')
        return series

    @staticmethod
    def _get_log_fn(f: FeatureNormalizeLogBase) -> Optional[Callable]:
        if f.log_base is None:
            return None
        if f.log_base == 'e':
            return np.log
        elif f.log_base == '10':
            return np.log10
        elif f.log_base == '2':
            return np.log2
        else:
            raise EnginePandaNumpyException(
                f'Problem processing Normalizer feature {f.name}. ' +
                f'Did not find function to calculated log-base {f.log_base}'
            )

    @staticmethod
    def _normalize_frequency_std(
            freq: np.ndarray, fn: List[Tuple[FeatureNormalizeStandard, int]],
            log_fn: Optional[Callable], inference: bool):

        ind = [i for _, i in fn]
        if len(ind) == 0:
            return

        # Deltas to add per feature if log transform is used
        deltas = np.array([f.delta for f, _ in fn])

        if not inference:
            if log_fn is None:
                mean = freq[:, :, ind].mean(axis=(0, 1))
                stddev = freq[:, :, ind].std(axis=(0, 1))
            else:
                mean = log_fn((freq[:, :, ind]+deltas)).mean(axis=(0, 1))
                stddev = log_fn((freq[:, :, ind]+deltas)).std(axis=(0, 1))
            for i, (f, _) in enumerate(fn):
                f.mean = mean[i].item()
                f.stddev = stddev[i].item()
            else:
                mean = np.array([f.mean for f, _ in fn])
                stddev = np.array([f.stddev for f, _ in fn])

            if log_fn is None:
                freq[:, :, ind] = ((freq[:, :, ind] - mean) / stddev)
            else:
                freq[:, :, ind] = ((log_fn(freq[:, :, ind]+deltas) - mean) / stddev)

    @staticmethod
    def _normalize_frequency_scale(
            freq: np.ndarray, fn: List[Tuple[FeatureNormalizeScale, int]],
            log_fn: Optional[Callable], inference: bool):

        ind = [i for _, i in fn]
        if len(ind) == 0:
            return

        # Deltas to add per feature if log transform is used
        deltas = np.array([f.delta for f, _ in fn])

        if not inference:
            if log_fn is None:
                mn = freq[:, :, ind].min(axis=(0, 1))
                mx = freq[:, :, ind].max(axis=(0, 1))
            else:
                mn = log_fn(freq[:, :, ind].min(axis=(0, 1))+deltas)
                mx = log_fn(freq[:, :, ind].max(axis=(0, 1))+deltas)
            for i, (f, _) in enumerate(fn):
                f.minimum = mn[i].item()
                f.maximum = mx[i].item()
        else:
            mn = np.array([f.minimum for f, _ in fn])
            mx = np.array([f.maximum for f, _ in fn])

        if log_fn is None:
            freq[:, :, ind] = ((freq[:, :, ind] - mn) / (mx - mn))
        else:
            freq[:, :, ind] = ((log_fn(freq[:, :, ind]+deltas) - mn) / (mx - mn))

    @staticmethod
    def _normalize_frequency(freq: np.ndarray, fn: List[Optional[FeatureNormalizeLogBase]], inference: bool) -> None:

        fnl: Dict[Type[FeatureNormalizeLogBase], Callable] = {
            FeatureNormalizeScale: EnginePandasNumpy._normalize_frequency_scale,
            FeatureNormalizeStandard: EnginePandasNumpy._normalize_frequency_std
        }

        if freq.shape[2] != len(fn):
            raise EnginePandaNumpyException(
                f'Problem normalizing frequency. Frequency number of features is {freq.shape[2]}. Number of ' +
                f'Normalizers is {len(fn)}. Normalizer Features list {fn}'
            )

        ur = list(set([(type(f), EnginePandasNumpy._get_log_fn(f)) for f in fn if f is not None]))
        for t, lb in ur:
            nl: List[Tuple[FeatureNormalize, int]] = [(f, fn.index(f)) for f in fn if isinstance(f, t)]
            func = fnl[t]
            func(freq, nl, lb, inference)

    @staticmethod
    def _process_key_frequencies(rows: pd.DataFrame, time_feature: Feature,
                                 time_dict: Dict[Tuple[TimePeriod, int],
                                                 List[Tuple[FeatureGrouper, Optional[FeatureNormalize]]]],
                                 list_ind: np.ndarray,
                                 tp_ind: np.ndarray,
                                 out_type: Type
                                 ) -> Tuple[pd.Index, List[np.ndarray]]:

        # Keep the original index, we need that to restore the order.
        row_index = rows.index
        # First sort rows on time_field. We want to go up in time as we process.
        rows.sort_values(by=[time_feature.name], ascending=True, inplace=True)

        p = ProfileNumpy([f for _, fts in time_dict.items() for f, _ in fts])

        # Define the size of the output structure. We have 1 np array per time_window/time_period.
        # The shape will be (length of rows, the time window length, number of features for the window length).
        out_shapes = np.array([(rows.shape[0], tw,  len(fts)) for (_, tw), fts in time_dict.items()])

        out = _numba_process_frequencies(
            np.ones(len(rows), dtype=np.bool),
            rows[[f.name for f in p.base_features]].to_numpy(),
            rows[[f.name for f in p.filter_features]].to_numpy(),
            ProfileNumpy.get_deltas(rows, time_feature), p.base_filters, p.feature_filters, p.timeperiod_filters,
            p.aggregator_indexes, p.filter_indexes, list_ind, tp_ind, p.time_windows, p.array_shape,
            out_shapes, out_type
        )

        return row_index, out

    def to_series_frequencies(self, target_tensor_def: TensorDefinition, file: str, time_feature: Feature,
                              delimiter: chr = ',', quote: chr = "'", inference: bool = True) -> NumpyList:
        """
        Method that will turn a Pandas DataFrame into a frequency series. It works on FeatureGrouper features
        It will 'group' the features (for instance per customer), order according to a date-time field and output
        a set of frequencies in the form of numpy arrays
        There will be a numpy array of rank-3 per combination of Time Window/Time Period in the list of grouper
        features. So if you request a 2-day and a 3-day and 2-week FeatureGroupers, then the output will be 3 numpy
        arrays.
        Each will have number of rows as first dimension. The second dimension will be the Time-Window. So for above
        example, the first list would have a second dimension of 2, the second list a second dimension of 3, the third
        list would have a second dimension of 2. The third dimension will be the number of features with that specific
        time-window/time-period. So in above case the third dimension will be 1 for each of the 3 numpy arrays.

        Args:
            target_tensor_def (TensorDefinition): Tensor Definition to use for the series creation. The features that
                should be turned into frequencies.
            file (str): The name of the file to read. This must be a path.
            time_feature (Feature):  The feature to use as date-time field. This field will be used to order the
                transactions.
            delimiter (str): The delimiter used in the file. Default is ','
            quote (str): Quote character. Default is "'"
            inference (bool): Indicate if we are inferring or not. If True [COMPLETE]

        Returns:
            NumpyList Object. It will contain one list per unique time_window/time_period found in the of the
            FeatureGroupers. The rank of each numpy array will be 3 (BatchSize x TimeWindow x #Features)
        """
        gnf: List[Tuple[FeatureGrouper, Optional[FeatureNormalize]]] = self._val_grouper_based(target_tensor_def)
        out_type = self._val_all_same_type(gnf)
        # Group per Group feature. i.e. per key.
        # Also override the time window to 1.
        # We can do this because we really only need to ever keep 1 time period. We store the rest in the frequency
        # time dimension.
        group_dict: Dict[Feature,
                         Dict[Tuple[TimePeriod, int],
                              List[Tuple[FeatureGrouper, Optional[FeatureNormalize]]]]] = OrderedDict(
            (g, OrderedDict(
                (tpw, sorted([
                    (FeatureGrouper(
                        fg.name, fg.type, fg.base_feature, fg.group_feature, fg.filter_feature,
                        fg.time_period, 1, fg.aggregator), fn) for fg, fn in fl
                ], key=lambda x: x[0])) for tpw, fl in groupby(
                    sorted(gf, key=lambda x: (x[0].time_period, x[0].time_window)),
                    lambda x: (x[0].time_period, x[0].time_window)
                )
            )) for g, gf in groupby(sorted(gnf, key=lambda x: x[0].group_feature.name), lambda x: x[0].group_feature)
        )
        # Create an 'unstacked' dataframe of the features. Do this first so it is ready for inference.
        f_features = target_tensor_def.embedded_features
        f_features = FeatureHelper.filter_not_feature(FeatureNormalize, f_features)
        f_features = FeatureHelper.filter_not_feature(FeatureGrouper, f_features)
        f_features.append(time_feature)
        tdf = TensorDefinition('InternalFrequenciesTime', list(set(f_features)))
        df = self.from_csv(tdf, file, delimiter, quote, inference=inference)

        # Dictionary to keep the numpy arrays per time period/window
        series_dict: [Dict[Tuple[TimePeriod, int], np.ndarray]] = {}

        for g, td in group_dict.items():
            logger.info(f'Start creating aggregate grouper feature for <{g.name}> ' +
                        f'using {self.num_threads} process(es)')

            # The profile will return a flat list of all aggregators across time-periods/windows.
            # But we will need to create a numpy array per unique time-period/window.
            # The list indexes will keep track of which output of the profile goes to which numpy array
            list_ind = np.array([0] + [len(fts) for _, fts in td.items()], dtype=np.int8)
            for i in range(1, len(list_ind)):
                list_ind[i] = list_ind[i] + list_ind[i - 1]

            # We will need to know which TimePeriods were used; we keep the key per list
            tp_ind = np.array([tp.key for (tp, _), fts in td.items()])

            # Single Threaded processing
            if self.num_threads == 1:
                df.sort_values(by=[g.name, time_feature.name], ascending=True, inplace=True)
                # Set-up Numpy profile and variables. First row is always True.
                p = ProfileNumpy([f for _, fts in td.items() for f, _ in fts])
                same_key = pd.concat((
                    pd.Series([True]),
                    df[g.name].iloc[1:].reset_index(drop=True).eq(df[g.name].iloc[:-1].reset_index(drop=True))
                ))
                # Define the size of the output structure. We have 1 np array per time_window/time_period.
                # The shape will be (length of rows, the time window length, number of features for the window length).
                out_shapes = np.array([(len(df), tw, len(fts)) for (_, tw), fts in td.items()])

                series = _numba_process_frequencies(
                    same_key.to_numpy(),
                    df[[f.name for f in p.base_features]].to_numpy(),
                    df[[f.name for f in p.filter_features]].to_numpy(),
                    ProfileNumpy.get_deltas(df, time_feature), p.base_filters, p.feature_filters, p.timeperiod_filters,
                    p.aggregator_indexes, p.filter_indexes, list_ind, tp_ind, p.time_windows, p.array_shape,
                    out_shapes, out_type
                )
                index = df.index
            # Multithreading processing
            else:
                key_function = partial(
                    self._process_key_frequencies,
                    time_feature=time_feature,
                    time_dict=td,
                    list_ind=list_ind,
                    tp_ind=tp_ind,
                    out_type=out_type
                )

                # This will a List[Tuple[pd.Index,List[np.ndarray]]. There will be an entry in the main list per unique
                # group value.
                # The index will return the original index, which we need to restore.
                # The list within the Tuple will contain a numpy array per each unique TimePeriod/TimeWindow combination
                with mp.Pool(self.num_threads) as p:
                    series = p.map(key_function, [rows for _, rows in df.groupby(g.name)])

                # Flatten index lists, we will get one long list across all group values.
                index = np.concatenate([ind.to_numpy() for ind, _ in series])
                # Complex stuff to make one long numpy across all group values per each unique TimePeriod/TimeWindow
                series = [lst for _, lst in series]
                series = [np.concatenate([e[i] for e in series], axis=0) for i in range(len(series[0]))]

            # End difference single vs multithreading processing
            # Getting an array by the arg-sorted index effectively sorts it, this restores the order of the original df
            series = [a[index.argsort()] for a in series]

            # Concatenate the series with the same Time window/Time period
            for i, tpw in enumerate(td):
                try:
                    n = series_dict[tpw]
                    series_dict[tpw] = np.concatenate((n, series[i]), axis=2)
                except KeyError:
                    series_dict[tpw] = series[i]

            logger.info(f'Done creating aggregate grouper features for <{g.name}>.')

        # End for loop over dictionary
        # Normalize the frequencies
        for i, (tpw, a) in enumerate(series_dict.items()):
            fn = [
                n for td in group_dict.values()
                for ipw, fl in td.items() if ipw == tpw
                for _, n in fl
            ]
            self._normalize_frequency(a, fn, inference)

        # Turn it into a NumpyList
        series = NumpyList([n for _, n in series_dict.items()])
        # Don't forget to set the Rank and shape
        target_tensor_def.rank = 3
        target_tensor_def.shapes = [(-1, *s[1:]) for s in series.shapes]
        logger.info(f'Done creating {target_tensor_def.name}. Shapes={series.shapes}')

        return series

    def to_networks_ego(self, network: NetworkDefinitionPandas, time_feature: Feature,
                        node_tensor_definition: List[TensorDefinition],
                        hops: int) -> Tuple[List[np.ndarray], ...]:
        # check only binary or continuous node properties
        # check all feature types are the same per node.
        # Check only continuous edge properties
        # check if time_feature is in the edge list
        #self._val_all_network_dfs_same_length(network)
        # Select the node property names, we do not want the id in this list
        node_pr_names = [nd.tensor_definition.feature_names for nd in network.node_definition_list]
        for nprn, nd in zip(node_pr_names, network.node_definition_list):
            nprn.remove(nd.id_feature.name)
        # Get the names of the node id-features.
        node_id_names = [n.id_feature.name for n in network.node_definition_list]
        node_b_pr = tuple([
            n.node_list[pr].to_numpy()
            for n, pr in zip(network.node_definition_list, node_pr_names)
            if len(n.tensor_definition.binary_features()) > 0
        ])
        # Find the Binary node Features
        node_b_pr_lst = [n for n in network.node_definition_list if len(n.tensor_definition.binary_features()) > 0]
        node_b_pr_ind = np.array([
            node_b_pr_lst.index(n) if n in node_b_pr_lst else -1 for n in network.node_definition_list
        ], dtype=np.int8)

        node_c_pr = tuple([
            n.node_list[pr].to_numpy()
            for n, pr in zip(network.node_definition_list, node_pr_names)
            if len(n.tensor_definition.continuous_features()) > 0
        ])
        node_c_pr_lst = [n for n in network.node_definition_list if len(n.tensor_definition.continuous_features()) > 0]
        node_c_pr_ind = np.array([
            node_c_pr_lst.index(n) if n in node_c_pr_lst else -1
            for n in network.node_definition_list
        ], dtype=np.int8)

        node_gr_pr = [FeatureHelper.filter_feature(FeatureGrouper, t.features) for t in node_tensor_definition]
        node_st_def = [
            ProfileNumpyStore(g, n.node_list[n_id].to_numpy()) if len(g) > 0 else None
            for g, n_id, n in zip(node_gr_pr, node_id_names, network.node_definition_list)
        ]
        edge_pr_names = [
            [f.name for f in e.tensor_definition.features
             if f != e.id_feature and f != e.from_node_id and f != e.to_node_id and f != time_feature]
            for e in network.edge_definition_list
        ]
        edge_pr = tuple([
            e.edge_list[pr].to_numpy()
            for e, pr in zip(network.edge_definition_list, edge_pr_names) if len(pr) > 0
        ])
        edge_ind = tuple([network.replace_by_index(e) for e in network.edge_definition_list])
        node_names = [n.name for n in network.node_definition_list]
        edge_node_ind = np.array([
            (node_names.index(e.from_node.name), node_names.index(e.to_node.name))
            for e in network.edge_definition_list
        ])
        enw = _numba_to_ego_networks(
            hops, node_b_pr, node_b_pr_ind, node_c_pr, node_c_pr_ind, edge_pr, edge_ind, edge_node_ind,
            tuple([ProfileNumpy.get_deltas(e.edge_list, time_feature, cumulative=True)
                   for e in network.edge_definition_list]),
            tuple([nsd.profile.base_filters if nsd is not None else np.zeros((0,)) for nsd in node_st_def]),
            tuple([nsd.profile.feature_filters if nsd is not None else np.zeros((0,)) for nsd in node_st_def]),
            tuple([nsd.profile.aggregator_indexes if nsd is not None else np.zeros((0,)) for nsd in node_st_def]),
            tuple([nsd.profile.filter_indexes if nsd is not None else np.zeros((0,)) for nsd in node_st_def]),
            tuple([nsd.profile.timeperiod_filters if nsd is not None else np.zeros((0,)) for nsd in node_st_def]),
            tuple([nsd.profile.time_windows if nsd is not None else np.zeros((0,)) for nsd in node_st_def]),
            tuple([len(g) for g in node_gr_pr]),
            tuple([nsd.new_store_array() for nsd in node_st_def if nsd is not None])
        )
        return enw


class _FeatureProcessor:
    """
    Worker class for feature processing. No real reason to make this a class other than to keep the base engine
    code concise.
    """
    # Adding logic for a new type of derived feature is done by adding a function in the which does the process
    # logic in the body of this class and adding it to the functions list in the process method.

    @staticmethod
    def _val_check_known_func(features: List[Feature], functions: Dict[Type[Feature], Callable]) -> None:
        """
        Validation function to see if we know how to build all the features.

        Args:
            features (List[Feature]): All feature that need to be built.
            functions (Dict[Type[Feature], Callable]): Dictionary with all known classes and their respective functions

        Returns:
             None
        """
        known_func = [f for s in functions.keys() for f in FeatureHelper.filter_feature(s, features)]
        unknown_func = [f for f in features if f not in known_func]
        if len(unknown_func) != 0:
            raise EnginePandaNumpyException(
                f'Do not know how to build field type. Can not build features: '
                f'{[field.name for field in unknown_func]}'
            )

    @staticmethod
    def _val_int_in_range(feature: FeatureIndex, d_type: np.dtype):
        v_min, v_max = np.iinfo(d_type).min, np.iinfo(d_type).max
        d_s = len(feature.dictionary)
        if d_s >= v_max:
            raise EnginePandaNumpyException(f'Dictionary of {feature.name} of size {d_s} too big for type {d_type}. '
                                            f'This will cause overflow. '
                                            f'Please choose a data type that can hold bigger numbers')

    @staticmethod
    def _val_int_is_binary(df: pd.DataFrame, feature: FeatureLabel):
        u = sorted(list(pd.unique(df[feature.base_feature.name])))
        if not u == [0, 1]:
            raise EnginePandaNumpyException(f'Binary Feature <{feature.name}> should only contain values 0 and 1 ')

    @staticmethod
    def _get_log_fn(f: FeatureNormalizeLogBase) -> Optional[Callable]:
        if f.log_base is None:
            return None
        if f.log_base == 'e':
            return np.log
        elif f.log_base == '10':
            return np.log10
        elif f.log_base == '2':
            return np.log2
        else:
            raise EnginePandaNumpyException(
                f'Problem processing Normalizer feature {f.name}. ' +
                f'Did not find function to calculated log-base {f.log_base}'
            )

    @staticmethod
    def _process_source_feature(df: pd.DataFrame, features: List[FeatureSource]) -> pd.DataFrame:
        # Apply defaults for source data fields of type 'CATEGORICAL'
        for feature in features:
            if feature.default is not None:
                if feature.type == FEATURE_TYPE_CATEGORICAL:
                    if feature.default not in df[feature.name].cat.categories.values:
                        df[feature.name] = df[feature.name].cat.add_categories(feature.default)
                df[feature.name].fillna(feature.default, inplace=True)
        return df

    @staticmethod
    def _process_normalize_feature(df: pd.DataFrame, features: List[FeatureNormalize], inference: bool) -> pd.DataFrame:
        # First Create a dictionary with mappings of fields to expressions. Run all at once at the end.
        if len(features) == 0:
            return df

        kwargs = {}
        for feature in features:
            fn = feature.name
            bfn = feature.base_feature.name
            if isinstance(feature, FeatureNormalizeScale):
                log_fn = _FeatureProcessor._get_log_fn(feature)
                if not inference:
                    feature.minimum = df[bfn].min() if log_fn is None else log_fn(df[bfn].min()+feature.delta)
                    feature.maximum = df[bfn].max() if log_fn is None else log_fn(df[bfn].max()+feature.delta)
                logger.info(f'Create {fn} Normalize/Scale {bfn}. Min. {feature.minimum:.2f} Max. {feature.maximum:.2f}')
                if log_fn is None:
                    kwargs[fn] = (df[bfn] - feature.minimum) / (feature.maximum - feature.minimum)
                else:
                    kwargs[fn] = (log_fn(df[bfn]+feature.delta) - feature.minimum) / (feature.maximum - feature.minimum)
            elif isinstance(feature, FeatureNormalizeStandard):
                log_fn = _FeatureProcessor._get_log_fn(feature)
                if not inference:
                    feature.mean = df[bfn].mean() if log_fn is None else log_fn(df[bfn]+feature.delta).mean()
                    feature.stddev = df[bfn].std() if log_fn is None else log_fn(df[bfn]+feature.delta).std()
                logger.info(f'Create {fn} Normalize/Standard {bfn}. Mean {feature.mean:.2f} Std {feature.stddev:.2f}')
                if log_fn is None:
                    kwargs[fn] = (df[bfn] - feature.mean) / feature.stddev
                else:
                    kwargs[fn] = (log_fn(df[bfn]+feature.delta) - feature.mean) / feature.stddev
            else:
                raise EnginePandaNumpyException(
                    f'Unknown feature normaliser type {feature.__class__.name}')

        # Update the Pandas dataframe. All normalizations are applied at once.
        df = df.assign(**kwargs)
        # Return the Pandas dataframe
        return df

    @staticmethod
    def _process_one_hot_feature(df: pd.DataFrame, features: List[FeatureOneHot], inference: bool,
                                 one_hot_prefix: str) -> pd.DataFrame:
        if len(features) == 0:
            return df

        # Keep original feature so we can add later.
        original_df = df[[f.base_feature.name for f in features]]

        if not inference:
            # Use pandas function to get the one-hot features. Set the 'expand names' inference attribute
            columns = [feature.base_feature.name for feature in features]
            df = pd.get_dummies(df, prefix_sep=one_hot_prefix, columns=columns)
            for oh in features:
                oh.expand_names = [c for c in df.columns if c.startswith(oh.base_feature.name + one_hot_prefix)]
        else:
            # During inference the values might be different. Need to make sure the number of columns matches
            # the training values. Values that were not seen during training will be removed.
            # Values that were seen during training but not at inference need to be added with all zeros
            columns = [feature.base_feature.name for feature in features]
            df = pd.get_dummies(df, prefix_sep=one_hot_prefix, columns=columns)
            # Add features seen at non-inference (training), but not at inference
            n_defined = []
            for f in features:
                defined = [col for col in df.columns if col.startswith(f.base_feature.name + one_hot_prefix)]
                x = [n for n in f.expand_names if n not in defined]
                n_defined.extend(x)
            if len(n_defined) > 0:
                kwargs = {nd: 0 for nd in n_defined}
                df = df.assign(**kwargs).astype('int8')
            # Remove features not seen at non-inference (training) but seen at inference
            n_defined = []
            for f in features:
                x = [col for col in df.columns
                     if col.startswith(f.base_feature.name + one_hot_prefix)
                     and col not in f.expand_names]
                n_defined.extend(x)
            if len(n_defined) > 0:
                df = df.drop(n_defined, axis=1)

        # Add back the original features. We might need them later
        return pd.concat([original_df, df], axis=1)

    @staticmethod
    def _process_index_feature(df: pd.DataFrame, features: List[FeatureIndex], inference: bool) -> pd.DataFrame:
        # Set dictionary if not in inference mode. Assume we want to build an index.
        if not inference:
            for feature in features:
                feature.dictionary = {cat: i + 1 for i, cat in enumerate(df[feature.base_feature.name].unique())}
        # Map the dictionary to the panda
        for feature in features:
            t = np.dtype(EnginePandasNumpy.panda_type(feature))
            # Check for int overflow. There could be too many values for the int type.
            if FeatureHelper.is_feature_of_type(feature, FeatureTypeInteger):
                _FeatureProcessor._val_int_in_range(feature, t)
            # For Panda categories we can not just fill the nans, they might not be in the categories and cause errors
            # So we must add 0 to the categories if it does not exist and then 'fill-na'.
            if df[feature.base_feature.name].dtype.name == 'category':
                if 0 not in df[feature.base_feature.name].cat.categories:
                    df[feature.base_feature.name] = df[feature.base_feature.name].cat.add_categories([0])
            df[feature.name] = df[feature.base_feature.name].map(feature.dictionary).fillna(0).astype(t)
        return df

    @staticmethod
    def _process_label_feature(df: pd.DataFrame, features: List[FeatureLabel]) -> pd.DataFrame:
        for feature in [f for f in features if isinstance(f, FeatureLabelBinary)]:
            _FeatureProcessor._val_int_is_binary(df, feature)
            df[feature.name] = df[feature.base_feature.name].copy().astype('int8')
        return df

    @staticmethod
    def _process_bin_feature(df: pd.DataFrame, features: List[FeatureBin], inference: bool) -> pd.DataFrame:
        # Add the binning features
        for feature in features:
            if not inference:
                # Geometric space can not start for 0
                mx = df[feature.base_feature.name].max()
                if feature.scale_type == 'geometric':
                    mn = max(df[feature.base_feature.name].min(), 1e-1)
                    bins = np.geomspace(mn, mx, feature.number_of_bins)
                else:
                    mn = df[feature.base_feature.name].min()
                    bins = np.linspace(mn, mx, feature.number_of_bins)
                # Set last bin to max possible value. Otherwise, unseen values above the biggest bin go to 0.
                bins[-1] = np.finfo(bins.dtype).max
                # Set inference attributes
                feature.bins = list(bins)
            bins = np.array(feature.bins)
            t = np.dtype(EnginePandasNumpy.panda_type(feature))
            labels = np.array(feature.range).astype(np.dtype(t))
            cut = pd.cut(df[feature.base_feature.name], bins=bins, labels=labels)
            df[feature.name] = cut.cat.add_categories(0).fillna(0)
        return df

    @staticmethod
    def _process_expr_features(df: pd.DataFrame, features: List[FeatureExpression]) -> pd.DataFrame:
        # Add the expression fields. Just call the expression with the parameter names. Use vectorization. Second best
        # to Native vectorization and faster than apply.
        for feature in features:
            t = EnginePandasNumpy.panda_type(feature, read=False)
            df[feature.name] = \
                np.vectorize(feature.expression, otypes=[t])(df[[f.name for f in feature.param_features]])
        return df

    @staticmethod
    def _process_ratio_features(df: pd.DataFrame, features: List[FeatureRatio]) -> pd.DataFrame:
        if len(features) == 0:
            return df
        # Add Ratio features. Simple division with some logic to avoid errors and 0 division. Note that pandas return
        # inf if the denominator is 0, and nan if both the numerator and the denominator are 0.
        # Do all ratios in one go using assign
        kwargs = {}
        for feature in features:
            t = np.dtype(EnginePandasNumpy.panda_type(feature))
            bfn = feature.base_feature.name
            dfn = feature.denominator_feature.name
            kwargs[feature.name] = df[bfn].div(df[dfn]).replace([np.inf, np.nan], 0).astype(t)
        # Apply concatenations
        df = df.assign(**kwargs)
        return df

    @staticmethod
    def _process_concat_features(df: pd.DataFrame, features: List[FeatureConcat]):
        if len(features) == 0:
            return df
        # Do all concatenations in one go using assign. Cast to string. If the feature is a category that is needed
        kwargs = {}
        for feature in features:
            kwargs[feature.name] = df[feature.base_feature.name].astype(str) + \
                                   df[feature.concat_feature.name].astype(str)
        # Apply concatenations
        df = df.assign(**kwargs)
        return df

    @staticmethod
    def _process_grouper_features(df: pd.DataFrame, features: List[FeatureGrouper],
                                  num_threads: int, time_feature: Feature) -> pd.DataFrame:
        if len(features) == 0:
            return df
        # Group per Group feature. i.e. per key.
        group_dict: Dict[Feature, List[FeatureGrouper]] = OrderedDict(
            sorted([
                (g, list(gf)) for g, gf in groupby(
                    sorted(features, key=lambda x: x.group_feature), lambda x: x.group_feature
                )
            ], key=lambda x: x[0]))

        for g, gf in group_dict.items():
            logger.info(f'Start creating aggregate grouper feature for <{g.name}> ' +
                        f'using {num_threads} process(es)')

            # Single process processing
            if num_threads == 1:
                df.sort_values(by=[g.name, time_feature.name], ascending=True, inplace=True)
                # Set-up Numpy profile and variables. First row is always True.
                p = ProfileNumpy(gf)
                same_key = pd.concat((
                    pd.Series([True]),
                    df[g.name].iloc[1:].reset_index(drop=True).eq(df[g.name].iloc[:-1].reset_index(drop=True))
                ))
                # Run numba jit-ed loop over the row in the df and process.
                out = _numba_process_grouper(
                    same_key.to_numpy(),
                    df[[f.name for f in p.base_features]].to_numpy(),
                    df[[f.name for f in p.filter_features]].to_numpy(),
                    ProfileNumpy.get_deltas(df, time_feature), p.base_filters, p.feature_filters, p.timeperiod_filters,
                    p.aggregator_indexes, p.filter_indexes, p.time_windows, len(gf), p.array_shape
                )
                ags = pd.DataFrame(
                    {f.name: pd.Series(np.squeeze(out[:, i]), index=df.index).astype(EnginePandasNumpy.panda_type(f))
                     for i, f in enumerate(gf)}
                )
                df = pd.concat([df, ags], axis=1)
            # Multi process processing
            else:
                key_function = partial(
                    _FeatureProcessor._process_grouper_key,
                    group_features=gf,
                    time_feature=time_feature
                )
                with mp.Pool(num_threads) as p:
                    dfs = p.map(key_function, [rows for _, rows in df.groupby(g.name)])
                df = pd.concat(dfs, axis=0)

            logger.info(f'Start creating aggregate grouper features for <{g.name}> ')

        # Restore Original Sort
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _process_grouper_key(rows: pd.DataFrame,
                             group_features: List[FeatureGrouper],
                             time_feature: Feature) -> pd.DataFrame:
        rows.sort_values(by=time_feature.name, ascending=True, inplace=True)
        p = ProfileNumpy(group_features)
        # Run numba jit-ed loop over the row in the df and process.
        out = _numba_process_grouper(
            np.ones(len(rows), dtype=np.bool),
            rows[[f.name for f in p.base_features]].to_numpy(),
            rows[[f.name for f in p.filter_features]].to_numpy(),
            ProfileNumpy.get_deltas(rows, time_feature), p.base_filters, p.feature_filters, p.timeperiod_filters,
            p.aggregator_indexes, p.filter_indexes, p.time_windows, len(group_features), p.array_shape
        )
        ags = pd.DataFrame(
            {f.name: pd.Series(np.squeeze(out[:, i]), index=rows.index).astype(EnginePandasNumpy.panda_type(f))
             for i, f in enumerate(group_features)}
        )
        return pd.concat([rows, ags], axis=1)

    @classmethod
    def process(cls, df: pd.DataFrame, features: List[Feature], inference: bool, one_hot_prefix: str,
                num_threads: int, time_feature: Optional[Feature]) -> pd.DataFrame:
        """
        Class method which will create a DataFrame of derived features. It will apply logic depending on the type of
        feature.

        Args:
            df: Pandas Dataframe containing the base features. The raw features as found in for instance a file
            features: List of Features to create.
            inference: Boolean value indicating if we are running in inference or not.
            one_hot_prefix: String, the prefix to be used for one-hot-encoded features
            num_threads: (int) The number of threads to use for multiprocessing
            time_feature: (Feature) Optional Parameter. Only needed if the processing requires a time field.

        Returns:
            A Pandas Dataframe containing all the features from the features list parameter
        """
        functions: Dict[Type[Feature], Callable] = {
            FeatureSource: partial(
                cls._process_source_feature,
                features=FeatureHelper.filter_feature(FeatureSource, features)
            ),
            FeatureNormalize: partial(
                cls._process_normalize_feature,
                features=FeatureHelper.filter_feature(FeatureNormalize, features),
                inference=inference
            ),
            FeatureOneHot: partial(
                cls._process_one_hot_feature,
                features=FeatureHelper.filter_feature(FeatureOneHot, features),
                inference=inference,
                one_hot_prefix=one_hot_prefix
            ),
            FeatureIndex: partial(
                cls._process_index_feature,
                features=FeatureHelper.filter_feature(FeatureIndex, features),
                inference=inference
            ),
            FeatureLabel: partial(
                cls._process_label_feature,
                features=FeatureHelper.filter_feature(FeatureLabel, features)
            ),
            FeatureBin: partial(
                cls._process_bin_feature,
                features=FeatureHelper.filter_feature(FeatureBin, features),
                inference=inference
            ),
            FeatureExpression: partial(
                cls._process_expr_features,
                features=FeatureHelper.filter_feature(FeatureExpression, features)
            ),
            FeatureRatio: partial(
                cls._process_ratio_features,
                features=FeatureHelper.filter_feature(FeatureRatio, features)
            ),
            FeatureConcat: partial(
                cls._process_concat_features,
                features=FeatureHelper.filter_feature(FeatureConcat, features)
            ),
            FeatureGrouper: partial(
                cls._process_grouper_features,
                features=FeatureHelper.filter_feature(FeatureGrouper, features),
                num_threads=num_threads,
                time_feature=time_feature
            )
        }
        cls._val_check_known_func(features, functions)
        for fn in functions.values():
            df = fn(df=df)
        return df


# Jit-ed functions in the main body. Numba does not like class methods very much
@jit(nopython=True, cache=True)
def _numba_process_grouper(same_key: np.ndarray, base_values: np.ndarray, filter_values: np.ndarray,
                           deltas: np.ndarray, b_flt: np.ndarray, f_flt: np.ndarray, tp_flt: np.ndarray,
                           a_ind: np.ndarray, f_ind: np.ndarray, tw: np.ndarray, group_feature_cnt: int,
                           pe_shape: Tuple[int, int, int]) -> np.array:
    """
    A Numba jit-ed function to that creates grouper features for each row in a Numpy array.

    Args:
        same_key (np.ndarray): A Numpy array of type bool and shape (#row_to_process). It contains True if the row at
            a specific location contained the same key as the previous row. It is used to reset the profile element to
            zeros.
        base_values (np.ndarray): A Numpy array of type float. It contains the values of the base feature that need
            to contribute to the profile. It has shape (#rows_to_process X #base_features)
        filter_values (np.ndarray): A Numpy array of type int. Has the values for the filter features that are used
            in the profile. It has shape (#rows_to_process X #filter_features)
        deltas (np.ndarray): A Numpy array of type int with the deltas for each of the TimePeriods. For each row to
            process it contains the difference in time compare to the previous row.
            It has shape (#rows_to_process X #time_periods). It can be fetched with the `ProfileNumpy.get_deltas` method
        b_flt (np.ndarray): A numpy array filter of type np.bool. Is an array that contains the filters for each
            base_feature in the profile. There is a row for each base filter, the columns on one specific row contain a
            filter that filters out the profile elements that should be updated by the respective base_feature.
            It has shape (#base_feature X #elements_in_profile)
            Values should be fetched with the `ProfileNumpy.base_filters` property
        f_flt (np.ndarray): A numpy array of type bool with shape (#group_features X #profile_elements). Each row
            contains a filter that can be used to select the correct profile element for that feature from the profile
            element array. Values should be fetched with the `ProfileNumpy.filter_features` property
        tp_flt (np.ndarray): An ndarray of type int. It has shape (#time_periods X #profile_element). It contains
            a row for each TimePeriod object. The row is a filter that filters out the elements using the respective
            TimePeriod object. Can be created with the `ProfileNumpy.timeperiod_filters` property
        a_ind (np.ndarray): A numpy array of type int with shape (#group_features). Each row contains the `key`/id
            of an Aggregator object for a specific group feature of this profile. Values should be fetched with the
            `ProfileNumpy.aggregator_indexes` property
        f_ind (np.ndarray) : A numpy array that holds an index to the filter that needs to be applied to each
            element in the profile. Values should be fetched with the `ProfileNumpy.filter_indexes` property
        tw (np.ndarray): A numpy array of type int with shape (#group_features). Each row the time window
            to be applied to a specific group feature of this profile. Values should be fetched with the
            `ProfileNumpy.time_windows` property
        group_feature_cnt (int): Number of FeatureGroupers used in the creation of the profile
        pe_shape Tuple(int, int, int): The shape of the profile element Numpy array. It has 3 dimensions. Values can
            be fetched with the `ProfileNumpy.array_shape` property.
    Returns:
        An Numpy array containing all the aggregate values for GrouperFeatures in this profile.
            It has shape (#rows_to_process X #grouper_features)
    """
    out = np.zeros((base_values.shape[0], group_feature_cnt))
    p = np.zeros(pe_shape)
    for i in range(base_values.shape[0]):
        if i > 0 and not same_key[i]:
            # Reset profile values
            p.fill(0.0)
        else:
            profile_time_logic(tp_flt, deltas[i], p)
        profile_contrib(b_flt, f_ind, base_values[i], filter_values[i], p)
        out[i, :] = profile_aggregate(f_flt, a_ind, tw, p)
    return out


@jit(nopython=True, cache=True)
def _numba_process_frequencies(same_key: np.ndarray, base_values: np.ndarray, filter_values: np.ndarray,
                               deltas: np.ndarray, b_flt: np.ndarray, f_flt: np.ndarray, tp_flt: np.ndarray,
                               a_ind: np.ndarray, f_ind: np.ndarray, l_ind: np.ndarray, tp_ind: np.ndarray,
                               tw: np.ndarray, pe_shape: Tuple[int, int, int],
                               out_shapes: np.ndarray, out_type: Type) -> List[np.ndarray]:
    """
    Process a frequency with a jit-ed function.
    """
    out = [
        np.zeros((out_shapes[i, 0], out_shapes[i, 1], out_shapes[i, 2]), out_type) for i in range(out_shapes.shape[0])
    ]
    p = np.zeros(pe_shape)
    for i in range(base_values.shape[0]):
        if i > 0:
            if not same_key[i]:
                # Reset profile values
                p.fill(0.0)
            else:
                # Always current row where the previous row stopped. (If the same key obviously)
                for j in range(len(out)):
                    out[j][i] = out[j][i - 1]
                profile_time_logic(tp_flt, deltas[i], p)

        profile_contrib(b_flt, f_ind, base_values[i], filter_values[i], p)
        # Iterate over TimePeriods
        for j in range(tp_ind.shape[0]):
            # Run Time-logic, see if the last values have to be shifted up.
            if deltas[i, tp_ind[j]] > 0:
                res = np.zeros_like(out[j][i, :, :])
                top = res.shape[0] - deltas[i, tp_ind[j]]
                if top > 0:
                    res[:top, :] = out[j][i, -top:, :]
                out[j][i, :, :] = res
            # Assign the values to last timeperiod bin (2nd dimension of the result structures).
            out[j][i, -1] = profile_aggregate(f_flt, a_ind, tw, p)[l_ind[j]:l_ind[j + 1]]

    return out


@nb.jit(nopython=True, cache=True)
def _numba_to_ego_networks(hops: int,
                           node_properties_binary: Tuple[np.ndarray, ...],
                           node_properties_binary_indexes: np.ndarray,
                           node_properties_continuous: Tuple[np.ndarray, ...],
                           node_properties_continuous_indexes: np.ndarray,
                           edge_properties: Tuple[np.ndarray, ...],
                           edge_indexes: Tuple[np.ndarray, ...],
                           edge_node_indexes: np.ndarray,
                           deltas: Tuple[np.ndarray, ...],
                           b_flt: Tuple[np.ndarray, ...],
                           f_flt: Tuple[np.ndarray, ...],
                           a_ind: Tuple[np.ndarray, ...],
                           f_ind: Tuple[np.ndarray, ...], tp_flt: Tuple[np.ndarray, ...],
                           tw: Tuple[np.ndarray, ...],
                           group_feature_count: Tuple[int, ...],
                           node_store: Tuple[np.ndarray, ...]) -> Tuple[List[np.ndarray], ...]:
    # Allocate output structure for the node features
    out_n_g_f = [np.zeros((0, c), dtype=np.float32) for c in group_feature_count]
    out_n_id = [np.zeros((0,), dtype=np.uint32) for _ in range(len(node_properties_binary_indexes))]

    # Allocate output structures for the edges
    out_e_id = [np.zeros((0,), dtype=np.uint32) for _ in edge_properties]

    # Allocate a structure to keep the last update delta per each entry in the profile store
    if len(node_store) > 0:
        delta_store = [np.zeros((ns.shape[0], 3), dtype=np.int32) for ns in node_store]
    else:
        delta_store = [np.zeros((0, 3), dtype=np.int32)]

    node_indexes = np.unique(edge_node_indexes)

    for i in range(len(edge_indexes[0])):
        for j in range(len(edge_properties)):
            e_i = edge_indexes[j][i]
            e_p = edge_properties[j][i, 0]
            # Update the node features for both from and to node
            # for tf in range(2):
            #     ni = edge_node_indexes[j, tf]
            #     if len(node_store) > ni and node_store[ni].shape[0] != 0:
            #         profile_time_logic(tp_flt[ni], deltas[ni][i], node_store[ni][e_i[tf]])
            #         profile_contrib(
            #             b_flt[ni], f_ind[ni], np.array(e_p).reshape((1,)), np.array([0]), node_store[ni][e_i[tf]]
            #         )
            #         delta_store[ni][e_i[tf]] = deltas[ni][i]
        # Create an ego network around the 'from' node. This will return a unique list of indexes per node-type
        ego_n, ego_e = _numba_create_ego_net(i, hops, 3, deltas, edge_indexes, edge_node_indexes)
        # Now get all the properties for each node
        out_ego_n_g = [np.zeros((ego_n[i].shape[0], c), dtype=np.float32) for i, c in enumerate(group_feature_count)]
        out_ego_e = [np.zeros((ego_e[i].shape[0], 2), dtype=np.float32) for i, _ in enumerate(group_feature_count)]
        # Get the Group features for the nodes in the ego-network
        # for s in range(len(node_store)):
        #     for n in range(ego_n[s].shape[0]):
        #         delta = deltas[s][i] - delta_store[s][ego_n[s][n]]
        #         profile_time_logic(tp_flt[s], delta, node_store[s][ego_n[s][n]])
        #         out_ego_n_g[s][n] = profile_aggregate(f_flt[s], a_ind[s], tw[s], node_store[s][ego_n[s][n]])
        # Build output, first add the nodes
        for e in range(len(node_indexes)):
            out_n_id[e] = np.concatenate((
                out_n_id[e],
                np.full((ego_n[e].shape[0],), i, dtype=np.uint32)
            ), axis=0)
            # Needs a for loop
            # out_n_g_f[e] = np.concatenate((out_n_g_f[e], out_ego_n_g[e]), axis=0)
        # Then add the edges
        for e in range(len(out_ego_e)):
            out_e_id[e] = np.concatenate((
                out_e_id[e],
                np.full((sum([out_ego_e[e].shape[0] for e in range(len(out_ego_e))]),), i, dtype=np.uint32)
            ), axis=0)

    return out_n_id, out_n_g_f, out_e_id


@nb.jit(nopython=True, cache=True)
def _numba_create_ego_net(i: int, number_of_hops: int, look_back_days: int,
                          deltas: Tuple[np.ndarray],
                          edge_indexes: Tuple[np.ndarray, ...],
                          edge_node_indexes: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Set-up an indexes array per each unique edge_node_index value
    eni = np.unique(edge_node_indexes)
    max_nodes_per_hop = 10
    n_indexes = [np.zeros((1,), dtype=np.uint32) for _ in eni]
    # And for the edges.
    e_indexes = [np.zeros((0, 2), dtype=np.uint32) for _ in edge_indexes]
    for h in range(number_of_hops):
        for ei in range(len(edge_indexes)):
            # Set start node for unique node-index if no node have been added
            if n_indexes[edge_node_indexes[ei, 0]].shape[0] == 1:
                n_indexes[edge_node_indexes[ei, 0]] = np.array(edge_indexes[ei][i, 0]).reshape((1,))
            flt_f = np.zeros((edge_indexes[ei].shape[0],), dtype=np.bool_)
            flt_t = np.zeros((edge_indexes[ei].shape[0],), dtype=np.bool_)
            start = np.searchsorted(deltas[ei][:, 0], deltas[ei][i, 0] - look_back_days)
            for j in nb.prange(start, i+1):
                flt_f[j] = _numba_is_in_edge_index(edge_indexes[ei][j, 0], n_indexes[edge_node_indexes[ei, 0]])
                flt_t[j] = _numba_is_in_edge_index(edge_indexes[ei][j, 1], n_indexes[edge_node_indexes[ei, 1]])

            # Random select max_nodes per hop for the to-nodes. This should avoid making the neighborhoods too large.
            flt_t_r = np.zeros_like(flt_t)
            flt_t_r[np.random.permutation(np.nonzero(flt_t)[0])[:max_nodes_per_hop]] = True

            n_indexes[edge_node_indexes[ei, 1]] = np.unique(
                np.hstack((n_indexes[edge_node_indexes[ei, 1]], edge_indexes[ei][:, 1][flt_f].flatten()))
            )
            n_indexes[edge_node_indexes[ei, 0]] = np.unique(
                np.hstack((n_indexes[edge_node_indexes[ei, 0]], edge_indexes[ei][:, 0][flt_t_r].flatten()))
            )
            # On the last iteration set the edges.
            if h == number_of_hops - 1:
                e_indexes[ei] = np.concatenate((e_indexes[ei], edge_indexes[ei][flt_t_r | flt_f]), axis=0)

    return n_indexes, e_indexes


@nb.jit(nb.uint32(nb.uint32, nb.uint32[:]), nopython=True, cache=True)
def _numba_is_in_edge_index(index: int, indexes: np.ndarray) -> bool:
    """
    Small 'isin()' function. Numba does not seem to support the numpy built-in isin.
    Args:

        index (int): The index value to look-up
        indexes (np.ndarray): Numpy array of shape(x,) containing a set of indexes. This is the look-up list in which
            the index is checked.
    Returns:
        bool. True if either the index value is in the indexes

    """
    i = np.searchsorted(indexes, index)
    if i < indexes.shape[0] and indexes[i] == index:
        return True
    else:
        return False
