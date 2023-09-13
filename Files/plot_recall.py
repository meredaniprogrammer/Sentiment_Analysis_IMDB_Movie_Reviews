# import matplotlib.pyplot as plt
#
#
# def plot_recall(models, recalls):
#     """
#     Function to plot recall of various models.
#
#     Parameters:
#     - models: List of names of the models.
#     - recalls: List of recall values corresponding to each model.
#     """
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(models, recalls, marker='o', linestyle='-', color='b')
#     plt.xlabel('Models')
#     plt.ylabel('Recall')
#     plt.title('Recall of Various Models')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#
#     # Display recall value when hovering on the line graph
#     for i, txt in enumerate(recalls):
#         plt.annotate(f"{txt:.2f}", (models[i], recalls[i]), fontsize=9, ha='right')
#
#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt

def plot_recall(models, recalls):
    """
    Function to plot recall of various models.

    Parameters:
    - models: List of names of the models.
    - recalls: List of recall values corresponding to each model.
    """

    # Define custom colors for bars
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    plt.figure(figsize=(10, 5))

    # Create a bar chart with tiny bars and different colors
    plt.bar(models, recalls, color=colors, width=0.2)

    plt.xlabel('Models')
    plt.ylabel('Recall')
    plt.title('Recall of Various Models')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display recall value on top of each bar with two decimal places
    for i, txt in enumerate(recalls):
        plt.text(models[i], txt, f"{txt:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

