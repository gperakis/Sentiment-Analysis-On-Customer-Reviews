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

