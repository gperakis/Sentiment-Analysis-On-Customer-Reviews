from tea import DATA_DIR


class WordEmbedding:
    def __init__(self):
        pass

    @staticmethod
    def get_word_embeddings(dimension=200):
        """
        This method reads the
        :return: dict. with the vocabulary word and its word embedding vector in a list
        """
        assert dimension in [50, 100, 200, 300]

        t = 'glove.6B.{}d.txt'.format(dimension)

        infile = "{}{}".format(DATA_DIR, t)

        with open(infile, 'rb') as in_file:
            text = in_file.read().decode("utf-8")

        word_embeddings = dict()
        for line in text.split('\n'):
            try:
                w_e_numbers = list(map(lambda x: float(x), line.split()[1:]))
                word_embeddings[line.split()[0]] = w_e_numbers
            except IndexError:
                pass

        return word_embeddings


if __name__ == '__main__':

    w_e = WordEmbedding.get_word_embeddings()

    print('the: {}'.format(w_e['the']))
    print('a: {}'.format(w_e['a']))
    print('egg: {}'.format(w_e['egg']))

    import numpy as np

    print('the: {}'.format(np.mean(w_e['the'])))


