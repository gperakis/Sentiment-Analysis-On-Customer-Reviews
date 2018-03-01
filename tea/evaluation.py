from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import pandas as pd


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


if __name__ == "__main__":

    a = ['positive','positive','positive','positive','positive','positive','positive',
         'negative','negative','negative','negative','negative','negative','negative']

    b = ['negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'positive',
         'positive', 'negative', 'negative', 'negative', 'negative', 'negative', 'positive']

    x= create_clf_report(y_true=a, y_pred=b, classes=['positive', 'negative'])