from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tea.evaluation import create_clf_report, create_benchmark_plot
from tea.features import *
from tea.load_data import parse_reviews

if __name__ == "__main__":

    train_data = parse_reviews(load_data=False, file_type='train')
    test_data = parse_reviews(load_data=False, file_type='test')

    X_train = train_data.drop(['polarity'], axis=1)
    X_test = test_data.drop(['polarity'], axis=1)

    y_train = train_data['polarity']

    y_test = test_data['polarity']

    X_train_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_train))
    X_test_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_test))

    vect_based_features = Pipeline([('extract', TextColumnExtractor(column='text')),
                                    ('contractions', ContractionsExpander()),
                                    ('vect', CountVectorizer(binary=True, min_df=0.01,
                                                             ngram_range=(1, 1), stop_words=None)),
                                    ('tfidf', TfidfTransformer(norm='l2', smooth_idf=False, use_idf=False)),
                                    ('to_dense', DenseTransformer())  # transforms sparse to dense
                                    ])

    user_based_features = FeatureUnion(transformer_list=[
        ('text_length', TextLengthExtractor(col_name='text', reshape=True)),
        ('avg_token_length', WordLengthMetricsExtractor(col_name='text', metric='avg', split_type='thorough')),
        ('std_token_length', WordLengthMetricsExtractor(col_name='text', metric='std', split_type='thorough')),
        ('contains_spc', ContainsSpecialCharactersExtractor(col_name='text')),
        ('n_tokens', NumberOfTokensCalculator(col_name='text')),
        ('contains_dots_bool', ContainsSequentialChars(col_name='text', pattern='..')),
        ('contains_excl_bool', ContainsSequentialChars(col_name='text', pattern='!!')),
        ('sentiment_positive', HasSentimentWordsExtractor(col_name='text', sentiment='positive', count_type='counts')),
        ('sentiment_negative', HasSentimentWordsExtractor(col_name='text', sentiment='negative', count_type='boolean')),
        ('contains_uppercase', ContainsUppercaseWords(col_name='text', how='count'))])

    final_pipeline = Pipeline([
        ('features', FeatureUnion(transformer_list=[
            ('vect_based_feat', vect_based_features),
            ('user_based_feat', user_based_features)]
        )),
        ('scaling', StandardScaler()),
        # ('pca', PCA()),
        ('clf', LogisticRegression(C=0.1))])

    final_pipeline_without_clf = Pipeline([
        ('features', FeatureUnion(transformer_list=[
            ('vect_based_feat', vect_based_features),
            ('user_based_feat', user_based_features)]
        )),
        ('scaling', StandardScaler())])

    fitted_model = final_pipeline.fit(X=X_train_lemmatized, y=y_train)

    y_test_pred = fitted_model.predict(X_test_lemmatized)

    create_clf_report(y_true=y_test,
                      y_pred=y_test_pred,
                      classes=fitted_model.classes_)

    X_train_benchmark = final_pipeline_without_clf.fit_transform(X_train_lemmatized)
    X_test_benchmark = final_pipeline_without_clf.transform(X_test_lemmatized)
    benchmark_clf = LogisticRegression(C=0.1)

    create_benchmark_plot(train_X=X_train_benchmark,
                          train_y=y_train,
                          test_X=X_test_benchmark,
                          test_y=y_test,
                          clf=benchmark_clf,
                          min_y_lim=0)
