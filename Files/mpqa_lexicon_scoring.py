from nltk.tokenize import word_tokenize


def load_mpqa_lexicon(filename):
    """
    Load the MPQA lexicon from a given file and return positive and negative word lists.
    """
    pos_list = set()
    neg_list = set()

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if "word1=" in line:
                parts = line.split(" ")
                polarity = None
                word = None

                for part in parts:
                    if "word1=" in part:
                        word = part.split("=")[1].strip()
                    if "priorpolarity=" in part:
                        polarity = part.split("=")[1].strip()

                if polarity == "positive":
                    pos_list.add(word)
                elif polarity == "negative":
                    neg_list.add(word)

    return pos_list, neg_list


def compute_mpqa_score(text, pos_list, neg_list):
    """
    Compute the MPQA sentiment score of a text using provided positive and negative word lists.
    """
    pos_count = 0
    neg_count = 0

    words = word_tokenize(text)
    for word in words:
        if word in pos_list:
            pos_count += 1
        elif word in neg_list:
            neg_count += 1

    # We return the difference between positive and negative counts.
    return pos_count - neg_count
