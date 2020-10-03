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
from functools import partial
from typing import Dict, List
from .common import EngineContext
from ..features.common import Feature, FeatureTypeTimeBased, FEATURE_TYPE_CATEGORICAL
from ..features.base import FeatureSource
from ..features.tensor import TensorDefinition
from ..features.expanders import FeatureExpander, FeatureOneHot

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
    'CATEGORICAL': 'category'
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
                names = [name for name in df.columns
                         if name == feature.name]
            if len(names) == 0:
                raise EnginePandaNumpyException(
                    f'During reshape, all features of tensor definition must be in the panda. Missing {feature.name}'
                )

    @staticmethod
    def _val_single_date_format_code(format_codes: List[str]):
        """Validation function to check that there is a single format code for dates across a set of format codes.

        :param format_codes: List of format codes from feature definitions. Is a string
        :return: None
        """
        if len(format_codes) > 1:
            raise EnginePandaNumpyException(f'All date formats should be the same. Got {format_codes}')

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
            raise EnginePandaNumpyException(f'Did not find panda type for {feature}')
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
        logger.info(f'Building Panda for : {tensor_def.name} from DataFrame')
        all_features = tensor_def.embedded_features
        source_features = [field for field in all_features if isinstance(field, FeatureSource)]
        one_hot_features = [field for field in all_features if isinstance(field, FeatureOneHot)]

        # Make sure we can make all fields
        unknown_fields = [field for field in all_features
                          if field not in source_features
                          and field not in one_hot_features]

        if len(unknown_fields) != 0:
            raise EnginePandaNumpyException(
                f'Do not know how to build field type. Can not build features: '
                f'{[field.name for field in unknown_fields]}'
            )

        # Start processing
        df = _FeatureProcessor.process_source_feature(df, source_features)
        df = _FeatureProcessor.process_one_hot_feature(df, one_hot_features, inference, self.one_hot_prefix)

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
        all_features = tensor_def.embedded_features
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


class _FeatureProcessor:
    """ Worker class for feature processing. No real reason to make this a class other than to keep the base engine
    code concise.
    """
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
    def process_one_hot_feature(df: pd.DataFrame, features: List[FeatureOneHot], inference: bool,
                                one_hot_prefix: str) -> pd.DataFrame:
        if not inference:
            # Use pandas function to get the one-hot features. Set the expand names inference attribute
            columns = [feature.base_feature.name for feature in features]
            df = pd.get_dummies(df, prefix_sep=one_hot_prefix, columns=columns)
            for oh in features:
                oh.expand_names = [c for c in df.columns if c.startswith(oh.base_feature.name + one_hot_prefix)]
        else:
            # Need to make sure we expand the same names
            # TODO need custom logic here.
            pass

        return df
