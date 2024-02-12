"""Contains the code for ICAPAI'21 paper "Counterfactual Explanations for Multivariate Time Series"

Authors:
    Emre Ates (1), Burak Aksar (1), Vitus J. Leung (2), Ayse K. Coskun (1)
Affiliations:
    (1) Department of Electrical and Computer Engineering, Boston University
    (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia
National Laboratories is a multimission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy’s National Nuclear Security Administration under Contract DENA0003525.
"""

import multiprocessing
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew, kurtosis

# from fast_features import generate_features

_TIMESERIES = None


class CheckFeatures(BaseEstimator, TransformerMixin):
    """
        Wrapper class that checks if the features used to train the classifier
        is the same as the features used to predict results
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        """
            Stores the columns names of all the generated features
            in the form of a list in a member variable. All names
            are represented by str.

            x = training data represented as a Pandas DataFrame
            y = training labels (not used in this class)
        """
        self.column_names = list(x.columns)
        if not self.column_names:
            logging.warning("Training data has no columns.")
        return self

    def transform(self, x, y=None):
        """
            Checks that the names of all the generated features from
            training and testing are the same. Prints an error if not
            and returns the dataframe provided in the arugment if so.

            x = testing data/data to compare with training data
            y = training labels (not used in this class)
        """
        argu_columns = list(x.columns)
        assert (
            self.column_names == argu_columns
        ), "Features of data from training doesn't match that of testing"
        return x


class TSFeatureGenerator(BaseEstimator, TransformerMixin):
    """Wrapper class for time series feature generation"""

    def __init__(self, trim=60, threads=multiprocessing.cpu_count(),
                 data_path=None,
                 features=['max', 'min', 'mean', 'std', 'skew', 'kurt',
                           'perc05', 'perc25', 'perc50', 'perc75', 'perc95']):
        self.features = features
        self.trim = trim
        self.threads = threads
        self.data_path = data_path

    def fit(self, x, y=None):
        """Extracts features
            x = training data represented as a Pandas DataFrame
            y = training labels (not used in this class)
        """
        return self

    def transform(self, x, y=None):
        """Extracts features
            x = testing data/data to compare with training data
            y = training labels (not used in this class)
        """
        global _TIMESERIES
        _TIMESERIES = x
        use_pool = self.threads != 1 and x.size > 100000
        if use_pool:
            pool = multiprocessing.Pool(self.threads)
        extractor = _FeatureExtractor(
            features=self.features, data_path=self.data_path,
            window_size=0, trim=self.trim)
        if isinstance(x, pd.DataFrame):
            index_name = 'node_id' if 'node_id' in x.index.names else 'nodeID'
            if not use_pool:
                result = [extractor(node_id)
                          for node_id in
                          x.index.get_level_values(index_name).unique()]
            else:
                result = pool.map(
                    extractor,
                    x.index.get_level_values(index_name).unique())
                pool.close()
                pool.join()
            result = pd.concat(result)
        else:
            # numpy array format compatible with Burak's notebooks
            if not use_pool:
                result = [extractor(i) for i in range(len(x))]
            else:
                result = pool.map(extractor, range(len(x)))
                pool.close()
                pool.join()
            result = np.concatenate(result, axis=0)
        return result


def _get_features(node_id, features=None, data_path=None, trim=60, **kwargs):
    global _TIMESERIES
    assert (
        features == ['max', 'min', 'mean', 'std', 'skew', 'kurt',
                     'perc05', 'perc25', 'perc50', 'perc75', 'perc95']
    )

    if data_path is not None:
        try:
            data = pd.read_hdf(
                data_path + '/table_{}.hdf'.format(node_id[-1]), node_id)
        except KeyError:
            data = _TIMESERIES.loc[node_id, :, :]
        if len(data) < trim * 2:
            return pd.DataFrame()
        return pd.DataFrame(
            generate_features(
                np.asarray(data.values.astype('float'), order='C'),
                trim
            ).reshape((1, len(data.columns) * 11)),
            index=[node_id],
            columns=[feature + '_' + metric
                     for metric in data.columns
                     for feature in features])
    elif isinstance(_TIMESERIES, pd.DataFrame):
        data = np.asarray(
            _TIMESERIES.loc[node_id, :, :].values.astype('float'),
            order='C')
        if len(data) < trim * 2:
            return pd.DataFrame()
        return pd.DataFrame(
            generate_features(data, trim).reshape(
                (1, len(_TIMESERIES.columns) * 11)),
            index=[node_id],
            columns=[feature + '_' + metric
                     for feature in features
                     for metric in _TIMESERIES.columns
                     ])
    else:
        data = np.asarray(_TIMESERIES[node_id].astype(float), order='C')
        # numpy array format
        if len(data) < trim * 2:
            return pd.DataFrame()
        return generate_features(data, trim).reshape(
            (1, _TIMESERIES.shape[2] * 11))


class _FeatureExtractor:
    """Needs _TIMESERIES to be set correctly"""
    def __init__(self, features=None, window_size=None, trim=None,
                 data_path=None):
        self.features = features
        self.window_size = window_size
        self.trim = trim
        self.data_path = data_path

    def __call__(self, node_id):
        return _get_features(
            node_id, features=self.features, data_path=self.data_path,
            window_size=self.window_size, trim=self.trim)


def generate_features(input_array, trim=0):
    # Ensure input is a numpy array
    input_array = np.asarray(input_array)
    
    # Trim the input array if needed
    if trim > 0:
        input_array = input_array[trim:-trim, :]
    
    # Initialize a list to store results
    results = []
    
    # Iterate over columns
    for col in input_array.T:  # Transpose to iterate over columns
        # Calculate statistics for each column
        max_val = np.amax(col)
        min_val = np.amin(col)
        mean_val = np.mean(col)
        std_val = np.std(col, ddof=0)  # Population standard deviation
        skew_val = skew(col, bias=True)  # Population skewness
        kurt_val = kurtosis(col, fisher=True, bias=True)  # Fisher's definition, bias-corrected
        
        # Percentiles
        perc05 = np.percentile(col, 5)
        perc25 = np.percentile(col, 25)
        perc50 = np.percentile(col, 50)
        perc75 = np.percentile(col, 75)
        perc95 = np.percentile(col, 95)
        
        # Append the results
        results.extend([max_val, min_val, mean_val, std_val, skew_val, kurt_val,
                        perc05, perc25, perc50, perc75, perc95])
    
    # Convert the results into a numpy array and reshape to match the expected output
    return np.array(results).reshape(-1, 11)
