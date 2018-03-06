from tea import setup_logger, NEGATIVE_WORDS, POSITIVE_WORDS
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from tea.text_mining import tokenize_text
from tea.word_embedding import WordEmbedding

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


class TextColumnExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts the column with the text"""

    def __init__(self, column):
        """

        :param column:
        """
        self.column = column

    def transform(self, X, y=None):

        if {self.column}.issubset(set(X.columns.tolist())):
            return X[self.column]

        else:
            raise Exception('Columns declared, not in dataframe')

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""

        return self


class DenseTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


class SingleColumnDimensionReshaper(BaseEstimator, TransformerMixin):

    def __init__(self):
        """

        """
        pass

    def transform(self, X, y=None):
        return X.values.reshape(-1, 1)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""

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

        logger.info('Calculating {} for "{}" Column'.format(self.metric, self.col_name))
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
        logger.info('Calculating text length for "{}" Column'.format(self.col_name))
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
        logger.info('Checking whether text contains special characters for "{}" Column'.format(self.col_name))

        return X[self.col_name].apply(lambda s: bool(set(s) & self.SPECIAL_CHARACTERS))

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class NumberOfTokensCalculator(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts number of tokens in text"""

    def __init__(self, col_name):
        """
        :param col_name:
        """
        self.col_name = col_name

    def transform(self, X, y=None):
        logger.info('Counting number of tokens for "{}" Column'.format(self.col_name))
        return X[self.col_name].apply(lambda x: len(tokenize_text(x, split_type='thorough')))

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class HasSentimentWordsExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts number of tokens in text"""

    def __init__(self,
                 col_name,
                 count_type='boolean',
                 input_type='text',
                 sentiment='negative'):
        """
        :param col_name:
        """
        assert sentiment in ['negative', 'positive']
        assert count_type in ['boolean', 'counts']
        assert input_type in ['text', 'tokens']

        self.col_name = col_name
        self.sentiment = sentiment
        self.input_type = input_type
        self.count_type = count_type

        if self.sentiment == 'positive':

            self.words_set = POSITIVE_WORDS
        else:
            self.words_set = NEGATIVE_WORDS

    def calculate_boolean_output(self, inp):
        """
        This method checks whether a sentence contains at least one tokens that contains sentiment.

        :param inp:
        :return:
        """
        tokens = inp.split() if self.input_type == 'text' else inp

        for token in set(tokens):
            if token in self.words_set:
                return True

        return False

    def calculate_counts_output(self, inp):
        """
        This method counts the number of tokens that contain sentiment in a text.
        :param inp:
        :return:
        """
        tokens = inp.split() if self.input_type == 'text' else inp

        return sum([1 for t in tokens if t in self.words_set])

    def transform(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """

        logger.info('Searching for {} sentiment of tokens for "{}" Column'.format(self.sentiment, self.col_name))

        if self.count_type == 'boolean':
            return X[self.col_name].apply(self.calculate_boolean_output)

        else:
            return X[self.col_name].apply(self.calculate_counts_output)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class AverageSentenceEmbedding(BaseEstimator, TransformerMixin):
    """Takes in dataframe, the average of sentence's word embeddings"""

    def __init__(self, col_name):
        """
        :param col_name: the name of the column that has the full text of the document
        """
        self.col_name = col_name
        self.word_embeddings = WordEmbedding.get_word_embeddings()

    def transform(self, X, y=None):
        logger.info('Counting number of tokens for "{}" Column'.format(self.col_name))

        def calculate_sentence_word_embedding(sentence):
            sum_w_e = 0
            for token in sentence.split():
                sum_w_e += np.mean(self.word_embeddings.get(token, 0))
            return sum_w_e / len(sentence.split())

        return X[self.col_name].apply(lambda x: calculate_sentence_word_embedding(x))

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
