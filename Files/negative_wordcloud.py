import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_negative_wordcloud(dataset_path):
    # Load the dataset
    data = pd.read_csv(dataset_path)

    # Filter reviews with negative sentiment (Assuming sentiment values are 'positive' and 'negative')
    negative_reviews = data[data['Sentiment'] == 'negative']['Review'].values

    # Join all the negative reviews into a single text
    negative_text = ' '.join(negative_reviews)

    # Generate the word cloud
    wordcloud = WordCloud(background_color="white", colormap="Reds", max_words=1000, contour_width=3,
                          contour_color='firebrick').generate(negative_text)

    # Display the word cloud
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
