import nltk
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize

# Download required corpora if not downloaded
nltk.download('opinion_lexicon', quiet=True)

def compute_bingliu_score(text):
    """
    Compute the Bing Liu Opinion Lexicon sentiment score of a text.
    """
    pos_list = set(opinion_lexicon.positive())
    neg_list = set(opinion_lexicon.negative())

    pos_count = 0
    neg_count = 0

    words = word_tokenize(text)
    for word in words:
        if word in pos_list:
            pos_count += 1
        elif word in neg_list:
            neg_count += 1

    # For this example, we return the difference between positive and negative counts.
    # But you can adjust this metric as you see fit.
    return pos_count - neg_count
