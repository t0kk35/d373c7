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
from functools import partial
from typing import Dict, List
from .common import EngineContext
from .numpy_helper import NumpyList
from ..features.common import Feature, FeatureTypeTimeBased, FEATURE_TYPE_CATEGORICAL, FeatureInferenceAttributes
from ..features.common import FeatureTypeInteger
from ..features.common import LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_LABEL
from ..features.common import LEARNING_CATEGORY_CONTINUOUS
from ..features.base import FeatureSource, FeatureIndex, FeatureBin
from ..features.tensor import TensorDefinition, TensorDefinitionMulti, LEARNING_CATEGORIES
from ..features.expanders import FeatureExpander, FeatureOneHot
from ..features.normalizers import FeatureNormalize, FeatureNormalizeScale, FeatureNormalizeStandard
from ..features.expressions import FeatureExpression, FeatureExpressionSeries
from ..features.labels import FeatureLabel, FeatureLabelBinary

logger = logging.getLogger(__name__)


class EnginePandaNumpyException(Exception):
    def __init__(self, message: str):
        super().__init__('Error creating Panda source: ' + message)


PandaTypes: Dict[str, str] = {
    'STRING': 'object',
    'FLOAT': 'float64',
    'FLOAT_32': 'float32',
    'INTEGER': 'int32',
    'INT_8': 'int8',
    'INT_16': 'int16',
    'INT_64': 'int64',
    'DATE': 'str',
    'CATEGORICAL': 'category',
    'DATETIME': 'datetime64'
}


class EnginePandasNumpy(EngineContext):
    """Panda and Numpy engine. It's main function is to take build Panda and Numpy structures from given
    tensor definition.

    Args:
        num_threads: The maximum number of thread the engine will use during multi-process processing

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

        :param df: A Panda Dataframe
        :param tensor_def: A tensor definition
        :return: None
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

        :param format_codes: List of format codes from feature definitions. Is a string
        :return: None
        """
        if len(format_codes) > 1:
            raise EnginePandaNumpyException(f'All date formats should be the same. Got {format_codes}')

    @staticmethod
    def _val_ready_for_inference(tensor_def: TensorDefinition, inference: bool):
        """Validation function to check if all feature are ready for inference. Some features have specific inference
        attributes that need to be set before an inference file can be made.

        :param tensor_def: The tensor that needs to be ready for inference.
        :param inference: Indication if we are inference mode or not.
        :return: None
        """
        if inference:
            if not tensor_def.inference_ready:
                nr = []
                for f in tensor_def.embedded_features:
                    if isinstance(f, FeatureInferenceAttributes):
                        if not f.inference_ready:
                            nr.append(f)
                raise EnginePandaNumpyException(
                    f'Tensor <{tensor_def.name}> not ready for inference. Following features not ready {nr}'
                )

    @staticmethod
    def _val_features_defined_as_columns(df: pd.DataFrame, features: List[Feature]):
        """Validation function that checks if the needed columns are available in the Panda. Only root features which
        are not derived from other features need to be in the Panda. The rest of the features can obviously be
        derived from the root features.

        :param df: The base Panda data to be checked.
        :param features: List of feature to check.
        :return: None
        """
        root_features = [f for f in features if len(f.embedded_features) == 0]
        unknown_features = [f for f in root_features if f.name not in df.columns]
        if len(unknown_features) != 0:
            raise EnginePandaNumpyException(
                f'All root features of a tensor definition (i.e. non-derived features) must be in the input df. Did '
                f'not find {[f.name for f in unknown_features]}'
            )

    @staticmethod
    def _val_check_known_logic(all_features: List[Feature], known_features: List[List[Feature]]):
        """Validation function to see if we know how to build all the features.

        :param all_features: All feature that need to be built.
        :param known_features: List of List of feature that were identified as having logic.
        :return: None
        """
        known_logic = [f for s in known_features for f in s]
        unknown_logic = [f for f in all_features if f not in known_logic]
        if len(unknown_logic) != 0:
            raise EnginePandaNumpyException(
                f'Do not know how to build field type. Can not build features: '
                f'{[field.name for field in unknown_logic]}'
            )

    @staticmethod
    def _val_key_feature_available(tensor_definition: TensorDefinition, key_feature: Feature):
        if key_feature not in tensor_definition.features:
            raise EnginePandaNumpyException(
                f'The key feature used to build a series must be in the tensor_definition'
            )

    @staticmethod
    def _val_time_feature_available(tensor_definition: TensorDefinition, time_feature: Feature):
        if not isinstance(time_feature.type, FeatureTypeTimeBased):
            raise EnginePandaNumpyException(
                f'The time feature used to build a series must be date based. It is of type {time_feature.type}'
            )

        if time_feature not in tensor_definition.features:
            raise EnginePandaNumpyException(
                f'The time feature used to build a series must be in the tensor_definition'
            )

    @property
    def num_threads(self):
        return self._num_threads

    @staticmethod
    def panda_type(feature: Feature, default: str = None) -> str:
        """Helper function that determines the panda (and numpy) data types for a specific feature. Base on the f_type

        :param feature: A feature definition
        :param default: A default data type (string)
        :return: Panda (Numpy) data type as a string
        """
        if feature is None:
            return default
        panda_type = PandaTypes.get(feature.type.name, default)
        if panda_type is None:
            raise EnginePandaNumpyException(f'Did not find panda type for {feature.name}')
        else:
            return panda_type

    @staticmethod
    def _parse_dates(dates: List[str], format_code: str) -> List[dt.datetime]:
        """Helper function to parse datetime structures from strings

        :param dates: A list of dates to parse (as string)
        :param format_code: The format code to apply
        :return: List of datetime type values
        """
        return [dt.datetime.strptime(d, format_code) for d in dates]

    def from_df(self, tensor_def: TensorDefinition, df: pd.DataFrame, inference: bool) -> pd.DataFrame:
        """Construct a Panda according to a tensor definition from another Panda. This is useful to construct derived
        features. One can first read the panda with from_csv to get the source features and then run this function to
        build all derived features

        :param tensor_def: The tensor_definition
        :param df: The input panda to re-construct
        :param inference: Indicate if we are inferring or not. If True [COMPLETE]
        :return: A Panda with the fields as defined in the tensor_def.
        """
        logger.info(f'Building Panda for : <{tensor_def.name}> from DataFrame. Inference mode <{inference}>')
        self._val_ready_for_inference(tensor_def, inference)
        all_features = tensor_def.features
        self._val_features_defined_as_columns(df, tensor_def.embedded_features)

        source_features = [field for field in all_features if isinstance(field, FeatureSource)]
        expression_features = [field for field in all_features if isinstance(field, FeatureExpression)]
        normalizer_features = [field for field in all_features if isinstance(field, FeatureNormalize)]
        index_features = [field for field in all_features if isinstance(field, FeatureIndex)]
        one_hot_features = [field for field in all_features if isinstance(field, FeatureOneHot)]
        label_features = [field for field in all_features if isinstance(field, FeatureLabel)]
        bin_features = [field for field in all_features if isinstance(field, FeatureBin)]

        self._val_check_known_logic(
            all_features,
            [
                source_features,
                expression_features,
                normalizer_features,
                index_features,
                one_hot_features,
                label_features,
                bin_features
            ])

        # Start processing
        df = _FeatureProcessor.process_source_feature(df, source_features)
        df = _FeatureProcessor.process_expr_features(df, expression_features)
        df = _FeatureProcessor.process_one_hot_feature(df, one_hot_features, inference, self.one_hot_prefix)
        df = _FeatureProcessor.process_index_feature(df, index_features, inference)
        df = _FeatureProcessor.process_normalize_feature(df, normalizer_features, inference)
        df = _FeatureProcessor.process_label_feature(df, label_features)
        df = _FeatureProcessor.process_bin_feature(df, bin_features, inference)

        # Only return base features in the tensor_definition. No need to return the embedded features.
        # Remember that expander features can contain multiple columns.
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

        # Don't forget to set the Tensor definition rank if in inference mode
        if not inference:
            tensor_def.rank = len(df.shape)
        logger.info(f'Done creating {tensor_def.name}. Shape={df.shape}')
        return df

    def from_csv(self, tensor_def: TensorDefinition, file: str, delimiter: chr = ',',
                 quote: chr = "'", inference: bool = True) -> pd.DataFrame:
        """Construct a Panda according to a tensor definition by reading a csv file.

        :param tensor_def: The input tensor definition
        :param file: File to read. This must be a complete file path
        :param delimiter: The delimiter used in the file. Default is ','
        :param quote: Quote character. Default is "'"
        :param inference: Indicate if we are inferring or not. If True [COMPLETE]
        :return: A Panda with the fields as defined in the tensor_def.
        """
        # Start by reading the SourceFeatures. Set to correct Panda Type
        file_instance = pathlib.Path(file)
        if not file_instance.exists():
            raise EnginePandaNumpyException(f' path {file} does not exist or is not a file')
        logger.info(f'Building Panda for : {tensor_def.name} from file {file}')
        all_features = tensor_def.features
        source_features = [field for field in all_features if isinstance(field, FeatureSource)]
        source_feature_names = [field.name for field in source_features]
        source_feature_types = {feature.name: EnginePandasNumpy.panda_type(feature) for feature in source_features}
        date_features = [feature for feature in source_features if isinstance(feature.type, FeatureTypeTimeBased)]
        date_feature_names = [f.name for f in date_features]
        # Set up some specifics for the date/time parsing
        if len(date_features) != 0:
            format_codes = list(set([d.format_code for d in date_features]))
            self._val_single_date_format_code(format_codes)
            date_parser = partial(self._parse_dates, format_code=format_codes[0])
            infer_datetime_format = False
        else:
            date_parser = None
            infer_datetime_format = True

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
        # Call from_df so all derived feature are calculated
        df = self.from_df(tensor_def, df, inference)
        return df

    def reshape(self, tensor_def: TensorDefinition, df: pd.DataFrame):
        """Reshape function. Can be used to reshuffle the columns in a Panda. The columns will be returned in the exact
        order as the features of the tensor definition. Columns that are not in the tensor definition as feature will
        be dropped.

        :param df: Input Panda.
        :param tensor_def: The tensor definition according which to reshape
        :return: A panda with the columns as defined in tensor_def
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
        """Method to convert a Pandas Dataframe to a object of type NumpyList. A NumpyList is an object which contains
        multiple Numpy arrays. The Numpy array can be turned into Pytorch Tensors.

        :param tensor_def: The TensorDefinition Object used to create the Pandas DataFrame.
        :param df: The Pandas DataFrame to convert to a NumpyList object.
        :return: A NumpyList Object containing Numpy arrays for the data in the Pandas DataFrame.
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
        return npl

    def multi_to_numpy_list(self, tensor_def: TensorDefinitionMulti, df: [pd.DataFrame]) -> NumpyList:
        """Method to convert a Pandas Dataframe to a object of type NumpyList. A NumpyList is an object which contains
        multiple Numpy arrays. The Numpy array can be turned into Pytorch Tensors. Other than the regular to_numpy
        method, this method supports multi-head input, it takes a list of TensorDefinitions and DataFrames.

        :param tensor_def: The TensorDefinitionMulti object used to create the Pandas DataFrame.
        :param df: The *list* of Pandas DataFrames to convert to a NumpyList object.
        :return: A NumpyList Object containing Numpy arrays for the data in the Pandas DataFrame.
        """
        npl = [self.to_numpy_list(td, df) for td, df in zip(tensor_def.tensor_definitions, df)]
        npl = [lst for n in npl for lst in n.lists]
        npl = NumpyList(npl)
        return npl

    @staticmethod
    def _process_key_sequence(rows: pd.DataFrame, time_field: Feature,
                              i_con_features: List[Feature], i_bin_features: List[Feature],
                              i_cat_features: List[Feature], con_type: str, bin_type: str, cat_type: str,
                              s_con_features: List[Feature], s_bin_features: List[Feature],
                              s_cat_features: List[Feature], length: int):

        # First sort
        sort = rows.sort_values(by=[time_field.name], ascending=True)
        indexes = sort.index

        # Enrich the series.
        for f_lst in (s_con_features, s_bin_features, s_cat_features):
            for f in f_lst:
                if isinstance(f, FeatureExpressionSeries):
                    t = EnginePandasNumpy.panda_type(f)
                    sort[f.name] = f.expression(sort[[p.name for p in f.param_features]]).astype(t)
                else:
                    raise EnginePandaNumpyException(f'Don\'t know how to build series feature {f.name}')

        # Generator to make sure we have the correct Learning Category order.
        def return_in_order(i_con, i_bin, i_cat, s_con, s_bin, s_cat, cn_type, b_type, ct_type):
            for lc in LEARNING_CATEGORIES:
                if lc == LEARNING_CATEGORY_CONTINUOUS:
                    yield i_con + s_con, cn_type
                elif lc == LEARNING_CATEGORY_BINARY:
                    yield i_bin + s_bin, b_type
                elif lc == LEARNING_CATEGORY_CATEGORICAL:
                    yield i_cat + s_cat, ct_type
                elif lc == LEARNING_CATEGORY_LABEL:
                    pass
                else:
                    raise EnginePandaNumpyException(f'Very unexpected internal error <{lc}>')

        lists = return_in_order(i_con_features, i_bin_features, i_cat_features,
                                s_con_features, s_bin_features, s_cat_features,
                                con_type, bin_type, cat_type)
        # Convert everything to numpy for performance.
        np_series = [sort[[f.name for f in f_lst]].to_numpy(np_type) for f_lst, np_type in lists]
        np_series = [ns for ns in np_series if ns.shape[1] != 0]

        def process_row(i: int):
            x = indexes[i]
            # Select the listed fields by name
            s = [ns[max(0, i - length + 1):i + 1] for ns in np_series]
            # Pad if incomplete. I.e. There were less than length rows before this row.
            s = [np.concatenate((np.zeros((length - e.shape[0], e.shape[1]), dtype=e.dtype), e))
                 if e.shape[0] < length else e for e in s]
            return [x] + s

        # Process all the rows. Return a padded Numpy array of the correct length.
        lists = [process_row(i) for i in range(len(sort))]
        return lists

    # TODO switch position of df and tensor_def for consistency with to_numpy
    def to_series_stacked(self, df: pd.DataFrame, tensor_def: TensorDefinition, key_field: Feature,
                          time_field: Feature, length: int) -> NumpyList:

        self._val_key_feature_available(tensor_def, key_field)
        self._val_time_feature_available(tensor_def, time_field)

        # Split off the Series specific features.
        i_con_features = [
            f for f in tensor_def.continuous_features(True) if not isinstance(f, FeatureExpressionSeries)
        ]
        i_bin_features = [
            f for f in tensor_def.binary_features(True) if not isinstance(f, FeatureExpressionSeries)
        ]
        i_cat_features = [
            f for f in tensor_def.categorical_features(True) if not isinstance(f, FeatureExpressionSeries)
        ]
        s_con_features = [
            f for f in tensor_def.continuous_features(True) if isinstance(f, FeatureExpressionSeries)
        ]
        s_bin_features = [
            f for f in tensor_def.binary_features(True) if isinstance(f, FeatureExpressionSeries)
        ]
        s_cat_features = [
            f for f in tensor_def.categorical_features(True) if isinstance(f, FeatureExpressionSeries)
        ]
        l_feature = [
            f for f in tensor_def.label_features(True)
        ]

        con_td = TensorDefinition('dummy_con', i_con_features + s_con_features)
        bin_td = TensorDefinition('dummy_bin', i_bin_features + s_bin_features)
        cat_td = TensorDefinition('dummy_cat', i_cat_features + s_cat_features)
        lab_td = TensorDefinition('dummy_label', l_feature)

        con_type = con_td.highest_precision_feature if len(con_td) > 0 else None
        bin_type = bin_td.highest_precision_feature if len(bin_td) > 0 else None
        cat_type = cat_td.highest_precision_feature if len(cat_td) > 0 else None
        lab_type = lab_td.highest_precision_feature if len(lab_td) > 0 else None

        con_type = self.panda_type(con_type, 'float64')
        bin_type = self.panda_type(bin_type, 'int8')
        cat_type = self.panda_type(cat_type, 'int32')
        lab_type = self.panda_type(lab_type, 'float32')

        key_function = partial(
            self._process_key_sequence,
            time_field=time_field,
            i_con_features=i_con_features,
            i_bin_features=i_bin_features,
            i_cat_features=i_cat_features,
            s_con_features=s_con_features,
            s_bin_features=s_bin_features,
            s_cat_features=s_cat_features,
            con_type=con_type,
            bin_type=bin_type,
            cat_type=cat_type,
            length=length
        )

        if self.num_threads is not None:
            num_processes = self.num_threads
        else:
            num_processes = int(mp.cpu_count() * 0.8)
        logger.info(f'Start creating stacked series for Tensor Definition <{tensor_def.name}> '
                    f'using {num_processes} process(es)')

        with mp.Pool(num_processes) as p:
            series = p.map(key_function, [rows for _, rows in df.groupby(key_field.name)])
        series = [s for keys in series for s in keys]
        # Need to sort to get back in the order of the index
        series.sort(key=lambda x: x[0])
        series = [np.array(s) for s in list(zip(*series))[1:]]
        # Add the label(s)
        if len(l_feature) != 0:
            labels = df[[f.name for f in l_feature]].to_numpy().astype(lab_type)
            series.append(labels)
        logger.info(f'Returning series of types {[str(s.dtype) for s in series]}.')
        # Turn it into a NumpyList
        series = NumpyList(series)
        # Don't forget to set the Rank.
        tensor_def.rank = 3
        logger.info(f'Done creating {tensor_def.name}. Shapes={series.shapes}')
        return series


class _FeatureProcessor:
    """ Worker class for feature processing. No real reason to make this a class other than to keep the base engine
    code concise.
    """
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
    def process_source_feature(df: pd.DataFrame, features: List[FeatureSource]) -> pd.DataFrame:
        # Apply defaults for source data fields of type 'CATEGORICAL'
        for feature in features:
            if feature.default is not None:
                if feature.type == FEATURE_TYPE_CATEGORICAL:
                    if feature.default not in df[feature.name].cat.categories.values:
                        df[feature.name].cat.add_categories(feature.default, inplace=True)
                df[feature.name].fillna(feature.default, inplace=True)
        return df

    @staticmethod
    def process_normalize_feature(df: pd.DataFrame, features: List[FeatureNormalize], inference: bool) -> pd.DataFrame:
        # First Create a dictionary with mappings of fields to expressions. Run all at once at the end.
        for feature in features:
            fn = feature.name
            bfn = feature.base_feature.name
            kwargs = {}
            if isinstance(feature, FeatureNormalizeScale):
                if not inference:
                    feature.minimum = df[bfn].min()
                    feature.maximum = df[bfn].max()
                logger.info(f'Create {fn} Normalize/Scale {bfn}. Min. {feature.minimum:.2f} Max. {feature.maximum:.2f}')
                kwargs[fn] = (df[bfn] - feature.minimum) / (feature.maximum - feature.minimum)
            elif isinstance(feature, FeatureNormalizeStandard):
                if not inference:
                    feature.mean = df[bfn].mean()
                    feature.stddev = df[bfn].std()
                logger.info(f'Create {fn} Normalize/Standard {bfn}. Mean {feature.mean:.2f} Std {feature.stddev:.2f}')
                kwargs[fn] = (df[bfn] - feature.mean) / feature.stddev
            else:
                raise EnginePandaNumpyException(
                    f'Unknown feature normaliser type {feature.__class__.name}')
            # Update the Panda
            df = df.assign(**kwargs)
        # Return the Panda
        return df

    @staticmethod
    def process_one_hot_feature(df: pd.DataFrame, features: List[FeatureOneHot], inference: bool,
                                one_hot_prefix: str) -> pd.DataFrame:
        if not inference:
            # Use pandas function to get the one-hot features. Set the expand names inference attribute
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
        return df

    @staticmethod
    def process_index_feature(df: pd.DataFrame, features, inference: bool) -> pd.DataFrame:
        # Set dictionary if not in inference mode. Assume we want to build an index.
        if not inference:
            for feature in features:
                feature.dictionary = {cat: i + 1 for i, cat in enumerate(df[feature.base_feature.name].unique())}
        # Map the dictionary to the panda
        for feature in features:
            t = np.dtype(EnginePandasNumpy.panda_type(feature))
            # Check for int overflow. There could be too many values for the int type.
            if isinstance(feature.type, FeatureTypeInteger):
                _FeatureProcessor._val_int_in_range(feature, t)
            # For Panda categories we can not just fill the nans, they might not be in the categories and cause errors
            if df[feature.base_feature.name].dtype.name == 'category':
                df[feature.base_feature.name].cat.add_categories([0], inplace=True)
            df[feature.name] = df[feature.base_feature.name].map(feature.dictionary).fillna(0).astype(t)
        return df

    @staticmethod
    def process_label_feature(df: pd.DataFrame, features: List[FeatureLabel]) -> pd.DataFrame:
        for feature in [f for f in features if isinstance(f, FeatureLabelBinary)]:
            _FeatureProcessor._val_int_is_binary(df, feature)
            df[feature.name] = df[feature.base_feature.name].copy().astype('int8')
        return df

    @staticmethod
    def process_bin_feature(df: pd.DataFrame, features: List[FeatureBin], inference: bool) -> pd.DataFrame:
        # Add the binning features
        for feature in features:
            if not inference:
                # Geometric space can not start for 0
                if feature.scale_type == FeatureBin.ScaleTypeGeometric:
                    mn = max(df[feature.base_feature.name].min(), 1e-1)
                else:
                    mn = df[feature.base_feature.name].min()
                mx = df[feature.base_feature.name].max()
                if feature.scale_type == FeatureBin.ScaleTypeGeometric:
                    bins = np.geomspace(mn, mx, feature.number_of_bins)
                else:
                    bins = np.linspace(mn, mx, feature.number_of_bins)
                # Set inference attributes
                feature.bins = list(bins)
            bins = np.array(feature.bins)
            t = np.dtype(EnginePandasNumpy.panda_type(feature))
            labels = np.array(feature.range).astype(np.dtype(t))
            cut = pd.cut(df[feature.base_feature.name], bins=bins, labels=labels)
            df[feature.name] = cut.cat.add_categories(0).fillna(0)
        return df

    @staticmethod
    def process_expr_features(df: pd.DataFrame, features: List[FeatureExpression]) -> pd.DataFrame:
        # Add the expression fields. Just call the expression with the parameter names. Use vectorization. Second best
        # to Native vectorization and faster than apply.
        for feature in features:
            t = np.dtype(EnginePandasNumpy.panda_type(feature))
            df[feature.name] = np.vectorize(feature.expression)(df[[f.name for f in feature.param_features]])
            df[feature.name].astype(t)
        return df
