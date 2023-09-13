import matplotlib.pyplot as plt


def plot_sentiment_distribution(data):
    """
    Function to plot the sentiment distribution in a pie chart.
    Assumes data contains a column 'Sentiment' with values 'Positive' and 'Negative'
    """

    # Drop NaN values
    data = data.dropna()

    # Ensure 'Sentiment' column exists
    if 'Sentiment' not in data.columns:
        print("Error: 'Sentiment' column is not present in the data!")
        return

    # Count positive and negative sentiments
    positive_count = sum(data['Sentiment'] == 'Positive')
    negative_count = sum(data['Sentiment'] == 'Negative')

    # Check if there's data to plot
    if positive_count + negative_count == 0:
        print("No data to plot!")
        return

    # Data to plot
    labels = 'Positive', 'Negative'
    sizes = [positive_count, negative_count]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0)  # explode 1st slice (i.e., 'Positive')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.title('Sentiment Distribution')
    plt.show()
