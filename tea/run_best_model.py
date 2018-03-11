from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from tea.features import *
from sklearn.preprocessing import LabelEncoder
from tea.load_data import parse_reviews
from tea.evaluation import create_clf_report, prec_recall_multi, plot_micro_prec_recall, \
    plot_micro_prec_recall_per_class, compute_roc_curve_area, plot_roc_multi
from sklearn.multiclass import OneVsRestClassifier
if __name__ == "__main__":
    train_data = parse_reviews(load_data=False, file_type='train')
    test_data = parse_reviews(load_data=False, file_type='test')

    le = LabelEncoder()

    X_train = train_data.drop(['polarity'], axis=1)
    X_test = test_data.drop(['polarity'], axis=1)

    y_train = le.fit_transform(train_data['polarity'])

    y_test = le.transform(test_data['polarity'])

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

    fitted_model = final_pipeline.fit(X=X_train_lemmatized, y=y_train)

    y_test_pred = fitted_model.predict(X_test_lemmatized)

    create_clf_report(y_true=y_test,
                      y_pred=y_test_pred,
                      classes=fitted_model.classes_)

    prec, recall, av_prec = prec_recall_multi(n_classes=2,
                                              X_test=X_test_lemmatized,
                                              Y_test=y_test,
                                              fitted_clf=fitted_model)

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(av_prec["micro"]))

    plot_micro_prec_recall(precision=prec,
                           recall=recall,
                           average_precision=av_prec)

    plot_micro_prec_recall_per_class(n_classes=2,
                                     precision=prec,
                                     recall=recall,
                                     average_precision=av_prec)

    fprdict, tprdict, roc_aucdict = compute_roc_curve_area(n_classes=2,
                                                           X_test=X_test_lemmatized,
                                                           y_test=y_test,
                                                           fittedclf=fitted_model)

    plot_roc_multi(fpr=fprdict, tpr=tprdict, roc_auc=roc_aucdict, n_classes=2)