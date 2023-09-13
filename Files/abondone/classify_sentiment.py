# Import necessary libraries
import pandas as pd
import os
from nltk.corpus import stopwords
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Function to clean the text
def process_text(s):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in s if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Now just remove any stopwords
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(clean_words)

def classify_reviews(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Apply the text cleaning function
    df['Processed_Review'] = df['Review'].apply(process_text)

    # Initialize VADER
    sid = SentimentIntensityAnalyzer()

    # Function to classify sentiment using polarity scores method
    def get_sentiment(text):
        # Create a SentimentIntensityAnalyzer object
        analysis = sid.polarity_scores(text)
        # Set sentiment
        if analysis['compound'] > 0:
            return 'positive'
        elif analysis['compound'] == 0:
            return 'neutral'
        else:
            return 'negative'

    # Apply function to the processed review column
    df['Sentiment'] = df['Processed_Review'].apply(get_sentiment)

    # Save the dataframe to a new CSV file
    df.to_csv(os.path.join(os.path.dirname(file_path), 'ClassifiedReviews.csv'), index=False)

    return df
