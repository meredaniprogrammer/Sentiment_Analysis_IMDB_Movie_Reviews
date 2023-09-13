from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def generate_wordcloud(text):
    """
    Generates and displays a word cloud visualization from the provided text.

    Args:
    - text (str): The input text for the word cloud.

    Returns:
    - None: Displays the word cloud.
    """

    # Define additional stopwords
    custom_stopwords = {
        "film", "movie", "scene", "character", "director", "plot", "actor",
        "actress", "cinema", "score", "script", "production", "performance",
        "cast", "audience", "screen", "storyline", "dialogue", "shot",
        "theme", "genre", "music", "camera", "action", "comedy", "drama",
        "thriller", "romance", "review", "sequel", "editing", "sound",
        "setting", "costume", "makeup", "role", "climax", "ending", "premise",
        "twist", "background", "intro", "middle", "adaptation", "novel",
        "book", "art", "visuals", "graphics", "design", "original", "remake",
        "series", "episode", "star", "lead", "supporting", "narrative",
        "tension", "atmosphere", "pace", "slow", "fast", "emotional", "humor",
        "funny", "sad", "happy", "exciting", "boring", "classic", "modern",
        "special", "effects", "realistic", "fake", "critic", "public",
        "recommend", "suggest", "worth", "minute", "hour", "talent", "flop",
        "hit", "box", "office", "ticket", "sell", "view", "experience", "feel",
        "impressive", "disappointing", "impact", "moment", "highlight", "low",
        "high", "rating", "stars"
    }

    # Merge STOPWORDS and custom_stopwords
    stopwords = set(STOPWORDS).union(custom_stopwords)

    # Create and generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)

    # Display the generated word cloud image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

