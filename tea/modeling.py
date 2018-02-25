from pprint import pprint


class Model:
    def __init__(self):
        pass

    @staticmethod
    def find_class_probabilities(data):
        """
        Calculates the probabilities of the classes.
        equation: number of documents in the class / total number of documents
        :param data: A dictionary with the classes and a list with the documents on each one.
        :return: double, probabilities for each class.
        """
        p = len(data.get('positive', []))
        n = len(data.get('negative', []))
        o = len(data.get('neutral', []))

        assert ((p + n + o) != 0)

        p_pos = p / (p + n + o)
        p_neg = n / (p + n + o)
        p_neu = o / (p + n + o)

        return p_pos, p_neg, p_neu

    @staticmethod
    def find_likelihood_probabilities(tokens, vocabulary):
        """
        Calculates the likelihood probabilities for each word for each class.
        :param tokens: A dictionary with the classes and sub-dictionaries with the tokens of each class.
        :param vocabulary: A dictionary with the vocabulary of all classes.
        :return: dictionary, with each word and their likelihood prob for each class.
        """
        words_probs = dict()
        for word in vocabulary:
            words_probs[word] = dict()

        for category in tokens:
            for word in tokens[category]:
                words_probs[word][category] = (tokens[category][word] + 1) / (len(tokens[category]) + len(vocabulary))

        return words_probs

    def fit_naive_bayes(self):
        pass

    def logistic_regression(self):
        pass

    @staticmethod
    def create_confusion_matrix():
        conf_matrix = dict()
        return conf_matrix

    @staticmethod
    def calculate_evaluation_metrics(confusion_matrix):
        """
        Calculates the evaluation metrics of the model.
        :param confusion_matrix: The confusion matrix of the model.
        :return: dictionary, with the metrics of the model.
        """
        metrics = dict()

        metrics['precision'] = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
        metrics['recall'] = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

        return metrics


if __name__ == '__main__':
    train = {'negative': ['just plain boring',
                          'entirely predictable and lacks energy',
                          'no surprises and very few laughs'],
             'positive': ['the most fun film of the summer',
                          'very powerful']}

    voc = {'just': 1,
           'plain': 1,
           'boring': 1,
           'entirely': 1,
           'predictable': 1,
           'and': 2,
           'lacks': 1,
           'energy': 1,
           'no': 1,
           'surprises': 1,
           'very': 2,
           'few': 1,
           'laughs': 1,
           'the': 2,
           'most': 1,
           'fun': 1,
           'film': 1,
           'of': 1,
           'summer': 1,
           'powerful': 1}

    train_tokens = {'negative': {'just': 1,
                                 'plain': 1,
                                 'boring': 1,
                                 'entirely': 1,
                                 'predictable': 1,
                                 'and': 2,
                                 'lacks': 1,
                                 'energy': 1,
                                 'no': 1,
                                 'surprises': 1,
                                 'very': 1,
                                 'few': 1,
                                 'laughs': 1},
                    'positive': {'the': 2,
                                 'most': 1,
                                 'fun': 1,
                                 'film': 1,
                                 'of': 1,
                                 'summer': 1,
                                 'very': 1,
                                 'powerful': 1}}

    pprint(train['negative'])

    model = Model()
    p_positive, p_negative, p_neutral = model.find_class_probabilities(train)

    print(p_positive, p_negative, p_neutral)

    word_probs = model.find_likelihood_probabilities(train_tokens, voc)
    pprint(word_probs)
