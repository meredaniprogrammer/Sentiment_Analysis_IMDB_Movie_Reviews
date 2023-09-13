import matplotlib.pyplot as plt


def plot_pie_metrics(model_names, accuracies, precisions, recalls, f1_scores):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for model, acc, prec, rec, f1 in zip(model_names, accuracies, precisions, recalls, f1_scores):
        sizes = [acc, prec, rec, f1]
        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
        explode = (0.1, 0, 0, 0)  # explode 1st slice for better visibility

        plt.pie(sizes, explode=explode, labels=metrics, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(model + " Performance Metrics")
        plt.show()

