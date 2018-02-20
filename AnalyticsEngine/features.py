from AnalyticsEngine import setup_logger
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from AnalyticsEngine.text_mining import tokenize_text

logger = setup_logger(__name__)


class ModelTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts a number of columns and return these columns"""

    def __init__(self, columns):
        """

        :param columns:
        """
        self.columns = columns

    def transform(self, X, y=None):

        if set(self.columns).issubset(set(X.columns.tolist())):
            return X[self.columns].values

        else:
            raise Exception('Columns declared, not in dataframe')

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""

        return self


class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


class WordLengthMetricsExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts text column, splits text in tokens and outputs average word length"""

    def __init__(self,
                 col_name,
                 split_type='simple',
                 metric='avg'):
        """

        :param split_type:
        """
        assert metric in ['avg', 'std']
        self.split_type = split_type
        self.col_name = col_name
        self.metric = metric

    def calculate_metric(self, words):
        """
        Helper code to compute average word length of a name
        :param words:
        :return:
        """
        if words:
            if self.metric == 'avg':
                return np.mean([len(word) for word in words])

            elif self.metric == 'std':
                return np.std([len(word) for word in words])

        else:
            return 0

    def transform(self, X, y=None):

        x = X[self.col_name].apply(lambda s: tokenize_text(text=s, split_type=self.split_type))

        return x.apply(self.calculate_metric)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts text column, returns sentence's length"""

    def __init__(self, col_name):
        """

        :param col_name:
        """
        self.col_name = col_name

    def transform(self, X, y=None):
        return X[self.col_name].apply(len)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class ContainsSpecialCharactersExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        """
        This class checks whether there are some given special characters in a text.
        :param col_name:
        """
        self.col_name = col_name
        self.SPECIAL_CHARACTERS = set("!@#$%^&*()_+-=")

    def transform(self, X, y=None):

        return X[self.col_name].apply(lambda s: bool(set(s) & self.SPECIAL_CHARACTERS))

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
