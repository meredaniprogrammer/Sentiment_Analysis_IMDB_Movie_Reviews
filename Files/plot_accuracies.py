# # plot_accuracies.py
#
# import plotly.graph_objects as go
#
# def plot_accuracies(models, accuracies):
#     """
#     Plot accuracies of various models on a line graph.
#
#     Args:
#     - models (list): List of model names
#     - accuracies (list): List of accuracy values corresponding to each model
#     """
#
#     # Create a line graph
#     fig = go.Figure(data=go.Scatter(x=models, y=accuracies, mode='lines+markers', hoverinfo='x+y'))
#
#     # Set plot title and axis labels
#     fig.update_layout(title='Model Accuracies',
#                       xaxis_title='Model Name',
#                       yaxis_title='Accuracy')
#
#     # Show the plot
#     fig.show()

# import matplotlib.pyplot as plt
#
# def plot_accuracy(models, accuracies):
#     """
#     Function to plot accuracy of various models.
#
#     Parameters:
#     - models: List of names of the models.
#     - accuracies: List of accuracy values corresponding to each model.
#     """
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(models, accuracies, marker='o', linestyle='-', color='b')
#     plt.xlabel('Models')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy of Various Models')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#
#     # Display accuracy value when hovering on the line graph
#     for i, txt in enumerate(accuracies):
#         plt.annotate(f"{txt:.2f}", (models[i], accuracies[i]), fontsize=9, ha='right')
#
#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt

def plot_accuracy(models, accuracies):
    """
    Function to plot accuracy of various models.

    Parameters:
    - models: List of names of the models.
    - accuracies: List of accuracy values corresponding to each model.
    """

    # Define custom colors for bars
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    plt.figure(figsize=(10, 5))

    # Create a bar chart with tiny bars and different colors
    plt.bar(models, accuracies, color=colors, width=0.2)

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Various Models')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display accuracy value on top of each bar with two decimal places
    for i, txt in enumerate(accuracies):
        plt.text(models[i], txt, f"{txt:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
