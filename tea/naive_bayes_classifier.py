import numpy as np
import string
import re
import math
from pprint import pprint


class NaiveBayesClassifier(object):
    """Implementation of Naive Bayes for binary classification"""

    def __init__(self):
        self.log_class_priors = {}
        self.word_counts = {}
        self.voc = set()

    @staticmethod
    def clean(text):
        """
        Cleans up a sequence/string by removing punctuation.
        :param text: string, the give sequence.
        :return: string, a cleaned sequence.
        """
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def tokenize(self, text):
        """
        Tokenizes a sequence/string into words.
        :param text: string, the give sequence.
        :return: a list with the tokens of the sequence.
        """
        text = self.clean(text).lower()
        return re.split('\W+', text)

    @staticmethod
    def get_word_counts(words):
        """
        Counts up how many of each word appears in a list of words.
        :param words: a list of strings.
        :return: a dictionary with the words and their frequency.
        """
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def fit(self, X, Y):
        """
        Computes log class priors. For each (document, label) pair, tokenize the document into words, add it to the
        vocabulary for each class and update the number of counts. Also add that word to the global vocabulary.
        :param X: a list with sentences
        :param Y: a list with the corresponding classes
        """
        n = len(X)
        self.log_class_priors['positive'] = math.log(sum(1 for label in Y if label == 1) / n)
        self.log_class_priors['negative'] = math.log(sum(1 for label in Y if label == 0) / n)
        self.word_counts['positive'] = {}
        self.word_counts['negative'] = {}

        for x, y in zip(X, Y):
            c = 'positive' if y == 1 else 'negative'
            counts = self.get_word_counts(self.tokenize(x))
            for word, count in counts.items():
                if word not in self.voc:
                    self.voc.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0

                self.word_counts[c][word] += count


    def predict(self, X):
        """
        Applies Naive Bayes directly. For each document, it iterates each of the words, computes the log likelihood,
        and sum them all up for each class. Then it adds the log class priors so as to compute the posterior and checks
        to see which score is bigger for that document.
        :param X: a list with sentences.
        :return: a list with the predicted labels.
        """
        result = []
        for x in X:
            counts = self.get_word_counts(self.tokenize(x))
            spam_score = 0
            ham_score = 0
            for word, _ in counts.items():
                if word not in self.voc:
                    continue

                # add Laplace smoothing
                log_w_given_spam = math.log((self.word_counts['positive'].get(word, 0.0) + 1) / (
                    sum(self.word_counts['positive'].values()) + len(self.voc)))
                log_w_given_ham = math.log((self.word_counts['negative'].get(word, 0.0) + 1) / (
                    sum(self.word_counts['negative'].values()) + len(self.voc)))

                spam_score += log_w_given_spam
                ham_score += log_w_given_ham

            spam_score += self.log_class_priors['positive']
            ham_score += self.log_class_priors['negative']

            if spam_score > ham_score:
                result.append(1)
            else:
                result.append(0)
        return result

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

        metrics['precision'] = confusion_matrix.get('TP', 1) / (
            confusion_matrix.get('TP', 1) + confusion_matrix.get('FP', 1))
        metrics['recall'] = confusion_matrix.get('TP', 1) / (
            confusion_matrix.get('TP', 1) + confusion_matrix.get('FN', 1))
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

        return metrics

if __name__ == '__main__':
    train_data = ['the most fun film of the summer',
                  'very powerful',
                  'just plain boring',
                  'entirely predictable and lacks energy',
                  'no surprises and very few laughs']

    train_target = [1, 1, 0, 0, 0]

    model = NaiveBayesClassifier()
    model.fit(train_data, train_target)

    test_data = ['predictable with no fun',
                 'predictable with few fun',
                 'very very fun',
                 'very fun']

    test_target = [0, 0, 0, 1]

    pred = model.predict(test_data)

    accuracy = sum(1 for i in range(len(pred)) if pred[i] == test_target[i]) / float(len(pred))
    print("{0:.4f}".format(accuracy))

