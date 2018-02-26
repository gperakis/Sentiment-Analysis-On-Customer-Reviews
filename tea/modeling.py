from pprint import pprint
import numpy as np
import operator


def tokenization(list_of_sentences):
    list_of_tokens = list()
    for sentence  in list_of_sentences:
        list_of_tokens.append(sentence.split())

    return list_of_tokens


class Model:
    def __init__(self):
        self.train_data = dict()
        self.class_probs = dict()
        self.word_likelihood = dict()
        self.categories = ['positive', 'negative']

        self.test_data = dict()
        self.conf_matrix = dict()

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

        class_prob = dict()
        class_prob['positive'] = p / (p + n + o)
        class_prob['negative'] = n / (p + n + o)
        class_prob['neutral'] = o / (p + n + o)

        return class_prob

    @staticmethod
    def find_likelihood_probabilities(tokens, vocabulary):
        """
        Calculates the likelihood probabilities for each word for each class.
        :param tokens: A dictionary with the classes and sub-dictionaries with the tokens of each class.
        :param vocabulary: A dictionary with the vocabulary of all classes.
        :return: dictionary, with each word and their likelihood prob for each class.
        """
        words_probs = dict()
        for category in tokens:
            words_probs[category] = dict()
            for word in tokens[category]:
                # words_probs[word][category] = (tokens[category][word] + 1) / (len(tokens[category]) + len(vocabulary))
                words_probs[category][word] = (tokens[category][word]) / (len(tokens[category]))

        return words_probs

    def fit_naive_bayes(self, data, tokens, vocabulary):
        """
        Fits the naive bayes algorithm in the data
        :param data: A dictionary with the classes and a list with the documents on each one.
        :param tokens: A dictionary with the classes and sub-dictionaries with the tokens of each class.
        :param vocabulary: A dictionary with the vocabulary of all classes.
        """
        self.class_probs = self.find_class_probabilities(data)
        self.word_likelihood = self.find_likelihood_probabilities(tokens, vocabulary)

    @staticmethod
    def compute_posterior(prior, likelihoods):
        """
        Calculates the posterior distribution of a sequence of words.
        :param prior: The prior probability of the class
        :param likelihoods: A list with the posterior probabilities of the words.
        :return: The posterior probability
        """
        lik = 1
        for word in likelihoods:
            if word != 0:
                lik *= word

        posterior = prior * lik
        return posterior

    # TODO: fix naive bayes calculation
    def predict_naive_bayes(self, data):
        """
        Predict with naive bayes algorithm in a given dataset
        :param data: A dictionary with the classes and a list with the documents on each one.
        :return: a dictionary with evaluation metrics: precision, recall and f1_score
        """
        pred = dict()

        for category in self.categories:
            print('Category: {}'.format(category))

            for sentence in data[category]:
                tokenized_sentence = sentence.split()
                print('Sentences: {}'.format(sentence))
                print(tokenized_sentence)

                posterior = dict()
                category_likelihoods = []
                for token in tokenized_sentence:
                    category_likelihoods.append(self.word_likelihood[category].get(token, 0))
                posterior[category] = self.compute_posterior(self.class_probs[category], category_likelihoods)
                # print(posterior)
                pred[sentence] = max(posterior.items(), key=operator.itemgetter(1))[0]
                # print(pred[sentence])
            break

        # for each_category in self.categories:
        #     # print(self.create_confusion_matrix(data, pred, each_category))
        #     break

        return self.calculate_evaluation_metrics(self.conf_matrix)

    def logistic_regression(self):
        pass

    @staticmethod
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

    @staticmethod
    def calculate_evaluation_metrics(confusion_matrix):
        """
        Calculates the evaluation metrics of the model.
        :param confusion_matrix: The confusion matrix of the model.
        :return: dictionary, with the metrics of the model.
        """
        metrics = dict()

        metrics['precision'] = confusion_matrix.get('TP', 1) / (confusion_matrix.get('TP', 1) + confusion_matrix.get('FP', 1))
        metrics['recall'] = confusion_matrix.get('TP', 1) / (confusion_matrix.get('TP', 1) + confusion_matrix.get('FN', 1))
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

    test = {'negative': ['predictable with no fun', 'predictable with few fun', 'very very fun'],
            'positive': ['very fun']}

    model = Model()
    model.fit_naive_bayes(train, train_tokens, voc)

    model.predict_naive_bayes(test)

