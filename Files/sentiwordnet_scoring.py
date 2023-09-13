import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required corpora if not downloaded
nltk.download('sentiwordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def get_wordnet_pos(treebank_tag):
    """
    Map POS tag to first character used by WordNet.
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def compute_sentiwordnet_score(text):
    """
    Compute the SentiWordNet sentiment score of a text.
    """
    sent_score = 0

    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)

        for word, pos in pos_tags:
            wn_pos = get_wordnet_pos(pos)
            if wn_pos:
                synsets = wn.synsets(word, pos=wn_pos)
                if synsets:
                    swn_synset = swn.senti_synset(synsets[0].name())
                    sent_score += swn_synset.pos_score() - swn_synset.neg_score()

    return sent_score

