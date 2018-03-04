from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, OneHotEncoder, LabelEncoder
from tea.load_data import parse_reviews
from tea.features import *
from pprint import pprint
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.decomposition import PCA
from time import time
from sklearn.model_selection import GridSearchCV


def run_grid_search(X, y, pipeline, parameters, scoring='accuracy'):
    """

    :param X:
    :param y:
    :param pipeline:
    :param parameters:
    :param scoring:
    :return:
    """

    # find the best parameters for both the feature extraction and the classifier
    grid_search = GridSearchCV(pipeline,
                               parameters,
                               n_jobs=-1,
                               verbose=10,
                               refit=True,
                               return_train_score=True,
                               scoring=scoring)

    logger.info("Performing grid search...")
    logger.info("Pipeline: {}".format([name for name, _ in final_pipeline.steps]))
    logger.info("Parameters:")
    logger.info(parameters)

    t0 = time()
    grid_search.fit(X=X, y=y)

    logger.info("Completed in %0.3fs" % (time() - t0))
    logger.info("Best score: %0.3f" % grid_search.best_score_)
    logger.info("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == "__main__":

    data = parse_reviews(load_data=True)

    X_train = data.drop(['polarity'], axis=1)
    y_train = data['polarity']

    print(X_train)
    text_length = Pipeline([
        ('extract', TextLengthExtractor(col_name='text')),
        ('reshaper', SingleColumnDimensionReshaper())])

    avg_token_length = Pipeline([
        ('extract', WordLengthMetricsExtractor(col_name='text', metric='avg', split_type='simple')),
        ('reshaper', SingleColumnDimensionReshaper())])

    std_token_length = Pipeline([
        ('extract', WordLengthMetricsExtractor(col_name='text', metric='std', split_type='simple')),
        ('reshaper', SingleColumnDimensionReshaper())])

    contains_spc_bool = Pipeline([
        ('extract', ContainsSpecialCharactersExtractor(col_name='text')),
        ('reshaper', SingleColumnDimensionReshaper())])

    n_tokens = Pipeline([
        ('extract', NumberOfTokensCalculator(col_name='text')),
        ('reshaper', SingleColumnDimensionReshaper())])

    vect_based_features = Pipeline([('extract', TextColumnExtractor(column='text')),
                                    ('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer()),
                                    ('to_dense', DenseTransformer())])

    user_based_features = Pipeline([('extract',
                                     FeatureUnion(transformer_list=[
                                         ('text_length', text_length),
                                         ('avg_token_length', avg_token_length),
                                         ('std_token_length', std_token_length),
                                         ('contains_spc', contains_spc_bool),
                                         ('n_tokens', n_tokens)])),
                                    ('scale', Normalizer())
                                    ])

    final_features = FeatureUnion(transformer_list=[
        ('vect_based_feat', vect_based_features),
        ('user_based_feat', user_based_features)])

    final_pipeline = Pipeline([('features', final_features),
                               ('clf', MultinomialNB())])

    for i in final_pipeline.steps:
        pprint(i)

    params = {
        'features__user_based_feat__extract__avg_token_length__extract__split_type': ['simple',
                                                                                      'thorough'],
        'features__user_based_feat__extract__std_token_length__extract__split_type': ['simple',
                                                                                      'thorough'],

        'features__vect_based_feat__vect__min_df': (0.005, 0.01, 0.025, 0.05, 0.1),
        'features__vect_based_feat__vect__max_features': (None, 5000, 10000, 25000, 50000),
        'features__vect_based_feat__vect__stop_words': (None, 'english'),
        'features__vect_based_feat__vect__binary': (True, False),
        'features__vect_based_feat__vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams, bigrams, trigrams
        'features__vect_based_feat__tfidf__use_idf': (True, False),
        'features__vect_based_feat__tfidf__norm': ('l1', 'l2'),
        'features__vect_based_feat__tfidf__smooth_idf': (True, False),
        # 'features__vect_based_feat__tfidf__sublinear_tf': (True, False)
    }

    run_grid_search(X=X_train,
                    y=y_train,
                    pipeline=final_pipeline,
                    parameters=params,
                    scoring='accuracy')