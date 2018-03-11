from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from tea.features import *
from tea.load_data import parse_reviews
from tea.run_models import run_grid_search

if __name__ == "__main__":
    data = parse_reviews(load_data=False)

    X_train = data.drop(['polarity'], axis=1)
    y_train = data['polarity']

    X_train_lemmatized = pd.DataFrame(LemmaExtractor(col_name='text').fit_transform(X_train))

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
        ('clf', LogisticRegression(C=0.1))
    ])

    model = final_pipeline.fit(X=X_train_lemmatized, y=y_train)

    print(model)
