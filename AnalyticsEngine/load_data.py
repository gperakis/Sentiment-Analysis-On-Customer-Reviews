from bs4 import BeautifulSoup as Soup
import pandas as pd
from bs4 import Tag
from AnalyticsEngine import DATA_DIR, setup_logger
import re

logger = setup_logger(__name__)


def parse_reviews(file='ABSA16_Laptops_Train_SB1_v2.xml',
                  save_data=True,
                  load_data=True):
    """

    :param file:
    :param save_data:
    :param load_data:
    :return:
    """
    path = "{}{}".format(DATA_DIR, file)

    if load_data:
        try:
            x = path.split('.')[-1]
            infile = re.sub(x, 'csv', path)
            logger.info('Loading file: {}'.format(infile))

            return pd.read_csv(infile)
        except FileNotFoundError:
            logger.warning('File Not Found on Data Directory. Creating a new one from scratch')

    data = list()

    handler = open(path).read()
    soup = Soup(handler, "lxml")

    reviews = soup.body.reviews

    for body_child in reviews:

        if isinstance(body_child, Tag):

            for body_child_2 in body_child.sentences:

                if isinstance(body_child_2, Tag):

                    opinions = body_child_2.opinions

                    # keeping only reviews that have a polarity
                    if opinions:
                        sentence = body_child_2.text.strip()
                        polarity = opinions.opinion.attrs.get('polarity')
                        data.append({'text': sentence, 'polarity': polarity})

    extracted_data = pd.DataFrame(data)

    if save_data:
        logger.info('Saving etracted reviews metadata from file: {}'.format(file))
        x = path.split('.')[-1]
        outfile = re.sub(x, 'csv', path)
        extracted_data.to_csv(outfile, encoding='utf-8', index=False)

    return extracted_data


if __name__ == "__main__":

    mydata = parse_reviews(load_data=True, save_data=True)
    print(mydata)
