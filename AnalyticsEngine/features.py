from AnalyticsEngine import setup_logger
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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
