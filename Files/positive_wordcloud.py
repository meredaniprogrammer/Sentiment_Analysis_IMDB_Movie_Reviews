# generate_positive_wordcloud.py

from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt

# A list of positive words (You can expand or modify this list based on your requirements)
POSITIVE_WORDS = [
    "love", "great", "amazing", "best", "fantastic", "wonderful", "impressive",
    "incredible", "outstanding", "beautiful", "brilliant", "awesome",
    "excellent", "perfect", "positive", "superb", "strong", "favorite",
    "top", "enjoy", "happy", "joy", "fun", "good", "charm", "captivating",
    "unforgettable", "sweet", "pleasure", "recommend", "stunning", "worth",
    "enjoyable", "delightful", "inspiring", "must-see", "heartwarming",
    "lovely", "masterpiece", "gem", "classic", "adore", "uplifting", "like",
    "remarkable", "mesmerizing", "satisfying", "pleased", "entertaining"
]


def generate_positive_wordcloud(dataset_path):
    """
    Generates and displays a word cloud visualization for positive words from the provided dataset path.

    Args:
    - dataset_path (str): The path to the movie reviews dataset.

    Returns:
    - None: Displays the word cloud.
    """

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Filter only rows that contain positive words
    positive_reviews = df[df['Review'].str.contains('|'.join(POSITIVE_WORDS), case=False, na=False)]
    positive_text = ' '.join(positive_reviews['Review'].tolist())

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        stopwords=STOPWORDS,
        background_color='white',
        colormap='spring'
    ).generate(positive_text)

    # Display the generated word cloud image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

