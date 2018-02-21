from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, OneHotEncoder, LabelEncoder
from tea.load_data import parse_reviews, get_df_stratified_split_in_train_validation
from tea.features import *
from pprint import pprint
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

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

    all_features = FeatureUnion(transformer_list=[
        ('text_length', text_length),
        ('avg_token_length', avg_token_length),
        ('std_token_length', std_token_length),
        ('contains_spc', contains_spc_bool),
        ('n_tokens', n_tokens)])

    final_pipeline = Pipeline([('features', all_features),
                               # ('pca', PCA()),
                               ('clf', GaussianNB())])

    obj = final_pipeline.fit(X=X_train, y=y_train)