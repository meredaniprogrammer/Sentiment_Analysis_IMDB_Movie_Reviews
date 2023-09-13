# plot_metrics.py

import matplotlib.pyplot as plt


def plot_metrics(model_names, accuracies, precisions, recalls, f1_scores):
    """
    Function to plot the performance metrics for various machine learning models.

    Parameters:
    - model_names (list): Names of ML models.
    - accuracies (list): Accuracy scores corresponding to the models.
    - precisions (list): Precision scores corresponding to the models.
    - recalls (list): Recall scores corresponding to the models.
    - f1_scores (list): F1 scores corresponding to the models.
    """

    # Set the width and position of bars
    barWidth = 0.2
    r1 = [i for i in range(len(model_names))]
    r2 = [i + barWidth for i in r1]
    r3 = [i + barWidth for i in r2]
    r4 = [i + barWidth for i in r3]

    # Create bars
    plt.bar(r1, accuracies, width=barWidth, label='Accuracy')
    plt.bar(r2, precisions, width=barWidth, label='Precision')
    plt.bar(r3, recalls, width=barWidth, label='Recall')
    plt.bar(r4, f1_scores, width=barWidth, label='F1 Score')

    # Style the graph
    plt.xlabel('Models', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(model_names))], model_names)
    plt.legend()

    # Show the graph
    plt.show()
