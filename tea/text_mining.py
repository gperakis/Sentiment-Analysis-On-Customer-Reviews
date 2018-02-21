import re
import unicodedata


def remove_accents(input_str):
    """
    This method removes any accents from any string.

    :param input_str:
    :return:
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()


def tokenize_text(text, split_type='simple'):
    """

    :param text:
    :param split_type:
    :return:
    """
    assert split_type in ['simple', 'thorough']

    if text:
        if split_type == 'simple':
            return text.strip().split()

        else:
            label = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z])|(_|\.|-|!|'))", r' \1', text)
            name_list = re.sub('(_|\.|-|!|#)', "", label).split()
            # Getting rid of any digits and other leftovers attached to the words of the list
            name_list = [re.sub(r"(\d+)|('|,|\)|{|}|=|&|`)", ' ', x) for x in name_list]
            name_list = [item.strip().lower() for x in name_list for item in x.split()]
            return name_list

    else:
        return []


def extract_digits_from_text(text):
    """
    This function extracts any digits in a text
    :param text:
    :return:
    """
    return list(map(int, re.findall(r'\d+', text))) if text else []
