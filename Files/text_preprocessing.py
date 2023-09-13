import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    # tokenize the text
    words = word_tokenize(text)

    # convert to lower case
    words = [w.lower() for w in words]

    # remove punctuation and special characters
    words = [''.join(c for c in w if c not in string.punctuation) for w in words]
    words = [w for w in words if w]

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    # stemming
    porter = PorterStemmer()
    words = [porter.stem(w) for w in words]

    return words
