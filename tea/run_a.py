from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, MinMaxScaler

from tea.features import *
from tea.load_data import parse_reviews
from tea.run_models import run_grid_search
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

if __name__ == "__main__":

    data = parse_reviews(load_data=True)

    X_train = data.drop(['polarity'], axis=1)
    y_train = data['polarity']

    # print(X_train)
    text_length = Pipeline([
        ('text_length', TextLengthExtractor(col_name='text')),
        ('reshaper', SingleColumnDimensionReshaper())])

    avg_token_length = Pipeline([
        ('avg_length', WordLengthMetricsExtractor(col_name='text', metric='avg', split_type='simple')),
        ('reshaper', SingleColumnDimensionReshaper())])

    std_token_length = Pipeline([
        ('std_length', WordLengthMetricsExtractor(col_name='text', metric='std', split_type='simple')),
        ('reshaper', SingleColumnDimensionReshaper())])

    contains_spc_bool = Pipeline([
        ('bool_spc', ContainsSpecialCharactersExtractor(col_name='text')),
        ('reshaper', SingleColumnDimensionReshaper())])

    contains_dots_bool = Pipeline([
        ('bool_dots', ContainsSequencialChars(col_name='text', pattern='..')),
        ('reshaper', SingleColumnDimensionReshaper())])

    contains_excl_bool = Pipeline([
        ('bool_excl', ContainsSequencialChars(col_name='text',pattern='!!')),
        ('reshaper', SingleColumnDimensionReshaper())])

    sentiment_positive = Pipeline([
        ('sent_positive', HasSentimentWordsExtractor(col_name='text', sentiment='positive')),
        ('reshaper', SingleColumnDimensionReshaper())])

    sentiment_negative = Pipeline([
        ('sent_negative', HasSentimentWordsExtractor(col_name='text', sentiment='negative')),
        ('reshaper', SingleColumnDimensionReshaper())])


    embedding = Pipeline([
        ('embedding', SentenceEmbeddingExtractor(col_name='text', embedding_output='vector')),
        ('reshaper', SingleColumnDimensionReshaper())])

    n_tokens = Pipeline([
        ('n_tokens', NumberOfTokensCalculator(col_name='text')),
        ('reshaper', SingleColumnDimensionReshaper())])

    contains_uppercase = Pipeline([
        ('cont_uppercase', ContainsUppercaseWords(col_name='text')),
        ('reshaper', SingleColumnDimensionReshaper())])

    vect_based_features = Pipeline([('extract', TextColumnExtractor(column='text')),
                                    ('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer()),
                                    ('to_dense', DenseTransformer()), # transforms sparse to dense
                                    ])

    user_based_features = Pipeline([('extract',
                                     FeatureUnion(transformer_list=[
                                         ('text_length', text_length),
                                         ('avg_token_length', avg_token_length),
                                         ('std_token_length', std_token_length),
                                         ('contains_spc', contains_spc_bool),
                                         ('n_tokens', n_tokens),
                                         ('contains_dots_bool', contains_dots_bool),
                                         ('contains_excl_bool', contains_excl_bool),
                                         ('sentiment_positive', sentiment_positive),
                                         ('sentiment_negative', sentiment_negative),
                                         ('contains_uppercase', contains_uppercase)
                                     ])),
                                    ('scale', Normalizer())
                                    ])

    final_features = FeatureUnion(transformer_list=[
        #('vect_based_feat', vect_based_features),
        ('user_based_feat', user_based_features),
        # ('embedding_feat', embedding)
    ])

    final_pipeline = Pipeline([('features', final_features),
                               ('scaling', MinMaxScaler()),
                               ('clf', MultinomialNB())])

    for i in final_pipeline.steps:
        pprint(i)

    params = {
        'features__user_based_feat__extract__sentiment_positive__sent_positive__count_type': ['boolean','counts'],
        'features__user_based_feat__extract__sentiment_negative__sent_positive__count_type': ['boolean','counts'],
        'features__user_based_feat__extract__contains_uppercase__cont_uppercase__how': ['bool', 'count'],
        'clf__alpha': (0, 0.5, 1.0) # MultinomialNB
        # 'features__vect_based_feat__vect__min_df': (0.005, 0.01, 0.025, 0.05, 0.1),
        # 'features__vect_based_feat__vect__max_features': (None, 1000, 2500, 5000),
        # 'features__vect_based_feat__vect__stop_words': (None, 'english'),
        # 'features__vect_based_feat__vect__binary': (True, False),
        # 'features__vect_based_feat__vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams, bigrams, trigrams
        # 'features__vect_based_feat__tfidf__use_idf': (True, False),
        # 'features__vect_based_feat__tfidf__norm': ('l1', 'l2'),
        # 'features__vect_based_feat__tfidf__smooth_idf': (True, False), # do not use
        # 'features__vect_based_feat__tfidf__sublinear_tf': (True, False) # do not use
    }

    run_grid_search(X=X_train,
                    y=y_train,
                    pipeline=final_pipeline,
                    parameters=params,
                    scoring='accuracy')
