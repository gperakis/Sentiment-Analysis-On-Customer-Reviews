from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import pandas as pd
from sklearn import metrics, svm
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import numpy as np

plt.rcParams['figure.figsize'] = (16, 8)


def create_confusion_matrix(actual, predicted, category):
    """
    Calculates the confusion matrix for a give category.
    :param actual: The actual labels of the data
    :param predicted: The predicted labels of the data
    :param category: The category we of the confusion matrix
    :return: dictionary, with the values of the confusion matrix
    """
    conf_matrix = dict()
    conf_matrix['TP'], conf_matrix['FP'], conf_matrix['TN'], conf_matrix['FN'] = 0, 0, 0, 0

    print('The category is: {}'.format(category))
    for sentence in predicted:
        if sentence in actual[predicted[sentence]] and predicted[sentence] == category:
            print('TP: Actual: {}, Predicted: {}'.format(category, category))
            conf_matrix['TP'] += 1
        elif sentence in actual[predicted[sentence]]:
            print('TN: Actual: not category, Predicted: not category'.format(predicted[sentence]))
            conf_matrix['TN'] += 1
        elif sentence not in actual[predicted[sentence]] and predicted[sentence] == category:
            print('FP: Actual: not category, Predicted: {}'.format(category))
            conf_matrix['FP'] += 1
        else:
            print('FN: Actual: {}, Predicted: {}'.format(category, predicted[sentence]))
            conf_matrix['FN'] += 1

    return conf_matrix


def calculate_evaluation_metrics(confusion_matrix):
    """
    Calculates the evaluation metrics of the model.
    :param confusion_matrix: The confusion matrix of the model.
    :return: dictionary, with the metrics of the model.
    """
    metrics = dict()

    metrics['precision'] = confusion_matrix.get('TP', 1) / (
            confusion_matrix.get('TP', 1) + confusion_matrix.get('FP', 1))
    metrics['recall'] = confusion_matrix.get('TP', 1) / (
            confusion_matrix.get('TP', 1) + confusion_matrix.get('FN', 1))
    metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

    return metrics


def create_clf_report(y_true, y_pred, classes):
    """
    This function calculates several metrics about a classifier and creates a mini report.

    :param y_true: iterable. An iterable of string or ints.
    :param y_pred: iterable. An iterable of string or ints.
    :param classes: iterable. An iterable of string or ints.
    :return: dataframe. A pandas dataframe with the confusion matrix.
    """
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred),
                             index=classes,
                             columns=['predicted_{}'.format(c) for c in classes])

    print("-" * 80, end='\n')
    print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("-" * 80)

    print("Confusion Matrix:", end='\n\n')
    print(confusion)

    print("-" * 80, end='\n')
    print("Classification Report:", end='\n\n')
    print(classification_report(y_true, y_pred, digits=3), end='\n')

    return confusion


def benchmark(clf, train_X, train_y, test_X, test_y):
    """
    This function calculates metrics for evaluation of a classifier over a training and a test set.

    :param clf: obj. An sklearn classifier
    :param train_X: array
    :param train_y: array
    :param test_X: array
    :param test_y: array
    :return: dict
    """
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)

    f1 = metrics.f1_score(test_y, pred, average='weighted')

    accuracy = metrics.accuracy_score(test_y, pred)

    print(" Acc: %f " % accuracy)

    result = {'f1': f1, 'accuracy': accuracy, 'train size': len(train_y),
              'test size': len(test_y), 'predictions': pred}

    return result


def create_benchmark_plot(train_X,
                          train_y,
                          test_X,
                          test_y,
                          clf,
                          splits=20,
                          plot_outfile=None,
                          y_ticks=0.025,
                          min_y_lim=0.4):
    """
    This method creates a benchmark plot.

    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :param clf:
    :param splits:
    :param plot_outfile:
    :param y_ticks:
    :param min_y_lim:
    :return:
    """

    results = {'train_size': [],
               'on_test': [],
               'on_train': []}

    # splitting the train X in n (almost) equal splits)
    train_x_splits = np.array_split(ary=train_X, indices_or_sections=splits, axis=0)

    # splitting the train y in the same splits as the train X
    train_y_splits = np.array_split(ary=train_y, indices_or_sections=splits, axis=0)

    # setting parameters for the graph.
    font_p = FontProperties()

    font_p.set_size('small')

    fig = plt.figure()
    fig.suptitle('Learning Curves', fontsize=20)

    ax = fig.add_subplot(111)
    ax.axis(xmin=0, xmax=train_X.shape[0] * 1.05, ymin=0, ymax=1.1)

    plt.xlabel('N. of training instances', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)

    plt.grid(True)

    plt.axvline(x=int(train_X.shape[0] * 0.3))
    plt.yticks(np.arange(0, 1.025, 0.025))

    if y_ticks == 0.05:
        plt.yticks(np.arange(0, 1.025, 0.05))

    elif y_ticks == 0.025:
        plt.yticks(np.arange(0, 1.025, 0.025))

    plt.ylim([min_y_lim, 1.025])

    # each time adds up one split and refits the model.
    for i in range(1, splits + 1):
        train_x_part = np.concatenate(train_x_splits[:i])
        train_y_part = np.concatenate(train_y_splits[:i])

        print(20 * '*')
        print('Split {} size: {}'.format(i, train_x_part.shape))

        results['train_size'].append(train_x_part.shape[0])

        result_on_test = benchmark(clf=clf,
                                   train_X=train_x_part,
                                   train_y=train_y_part,
                                   test_X=test_X,
                                   test_y=test_y)

        # calculates each time the metrics also on the test.
        results['on_test'].append(result_on_test['accuracy'])

        # calculates the metrics for the given training part
        result_on_train_part = benchmark(clf=clf,
                                         train_X=train_x_part,
                                         train_y=train_y_part,
                                         test_X=train_x_part,
                                         test_y=train_y_part)

        results['on_train'].append(result_on_train_part['accuracy'])

        line_up, = ax.plot(results['train_size'],
                           results['on_train'],
                           'o-',
                           label='Accuracy on Train')

        line_down, = ax.plot(results['train_size'],
                             results['on_test'],
                             'o-',
                             label='Accuracy on Test')

        plt.legend([line_up, line_down],
                   ['Accuracy on Train', 'Accuracy on Test'],
                   prop=font_p)

    if plot_outfile:
        fig.savefig(plot_outfile)

    plt.show()

    return results


if __name__ == "__main__":
    a = ['positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive',
         'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative']

    b = ['negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'positive',
         'positive', 'negative', 'negative', 'negative', 'negative', 'negative', 'positive']

    create_clf_report(y_true=a, y_pred=b, classes=['positive', 'negative'])

    # Another Example

    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target

    # split 80% train , 20% test
    digits_train_x = X[0: int(X.shape[0] * 0.8), :]
    digits_train_y = Y[0: int(X.shape[0] * 0.8)]
    digits_test_x = X[int(X.shape[0] * 0.8):, :]
    digits_test_y = Y[int(X.shape[0] * 0.8):]

    # normalize values between -1 and 1 with the simple minmax algorithm
    normalizer = MinMaxScaler(feature_range=(-1.0, 1.0))
    digits_train_x = normalizer.fit_transform(digits_train_x)
    digits_test_x = normalizer.transform(digits_test_x)

    from sklearn.utils import shuffle

    train_x_shuffled, train_y_shuffled = shuffle(digits_train_x,
                                                 digits_train_y,
                                                 random_state=1989)

    clf_svm = svm.LinearSVC(random_state=1989,
                            C=100.,
                            penalty='l2',
                            max_iter=1000)

    create_benchmark_plot(train_x_shuffled,
                          train_y_shuffled,
                          digits_test_x,
                          digits_test_y,
                          clf_svm,
                          10,
                          None,
                          0.025,
                          0.5)
