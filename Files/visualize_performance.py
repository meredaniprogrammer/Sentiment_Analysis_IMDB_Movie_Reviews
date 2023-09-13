# visualize_performance.py

import matplotlib.pyplot as plt

def plot_performance_pie_chart(accuracy, precision, recall, f1):
    """
    Visualizes the provided performance metrics in a pie chart.

    Parameters:
        accuracy : Accuracy score
        precision : Precision score
        recall : Recall score
        f1 : F1 score
    """

    # Data to plot
    labels = 'Accuracy', 'Precision', 'Recall', 'F1 Score'
    sizes = [accuracy, precision, recall, f1]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice (i.e., 'Accuracy')

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.2f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.title("Performance Metrics Pie Chart")
    plt.show()


