from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, OneHotEncoder, LabelEncoder
from tea.load_data import parse_reviews, get_df_stratified_split_in_train_validation
from tea.features import *
from pprint import pprint
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from time import time
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    data = parse_reviews(load_data=True)

    res = get_df_stratified_split_in_train_validation(data=data,
                                                      label='polarity',
                                                      validation_size=0.2,
                                                      random_state=5)

    X_train = res['x_train']
    X_validation = res['x_validation']
    y_train = res['y_train']
    y_validation = res['y_validation']

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

    text_based_features = Pipeline([('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer())])

    features_set = Pipeline([('extract',
                              FeatureUnion(transformer_list=[
                                  ('text_length', text_length),
                                  ('avg_token_length', avg_token_length),
                                  ('std_token_length', std_token_length),
                                  ('contains_spc', contains_spc_bool),
                                  ('n_tokens', n_tokens)])),
                             # ('scale', Normalizer())
                             ])

    final_pipeline = Pipeline([('features', features_set),
                               ('clf', GaussianNB())])

    parameters = {
        'features__extract__avg_token_length__extract__split_type': ['simple', 'thorough'],
        'features__extract__std_token_length__extract__split_type': ['simple', 'thorough'],
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        # 'clf__alpha': (0.00001, 0.000001),
        # 'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__n_iter': (10, 50, 80),
        }

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(final_pipeline,
                               parameters,
                               n_jobs=-1,
                               verbose=10)

    logger.info("Performing grid search...")
    logger.info("Pipeline: {}".format([name for name, _ in final_pipeline.steps]))
    logger.info("Parameters:")
    logger.info(parameters)

    t0 = time()
    grid_search.fit(X=X_train, y=y_train)

    logger.info("Completed in %0.3fs" % (time() - t0))
    logger.info("Best score: %0.3f" % grid_search.best_score_)
    logger.info("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))
