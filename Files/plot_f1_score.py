# import matplotlib.pyplot as plt
#
# def plot_f1score(models, f1_scores):
#     """
#     Function to plot F1-Score of various models.
#
#     Parameters:
#     - models: List of names of the models.
#     - f1_scores: List of F1-Score values corresponding to each model.
#     """
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(models, f1_scores, marker='o', linestyle='-', color='b')
#     plt.xlabel('Models')
#     plt.ylabel('F1-Score')
#     plt.title('F1-Score of Various Models')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#
#     # Display F1-Score value when hovering on the line graph
#     for i, txt in enumerate(f1_scores):
#         plt.annotate(f"{txt:.2f}", (models[i], f1_scores[i]), fontsize=9, ha='right')
#
#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt

def plot_f1score(models, f1_scores):
    """
    Function to plot F1-Score of various models.

    Parameters:
    - models: List of names of the models.
    - f1_scores: List of F1-Score values corresponding to each model.
    """

    # Define custom colors for bars
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    plt.figure(figsize=(10, 5))

    # Create a bar chart with tiny bars and different colors
    plt.bar(models, f1_scores, color=colors, width=0.2)

    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.title('F1-Score of Various Models')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display F1-Score value on top of each bar with two decimal places
    for i, txt in enumerate(f1_scores):
        plt.text(models[i], txt, f"{txt:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()



#-----------------------------------------
