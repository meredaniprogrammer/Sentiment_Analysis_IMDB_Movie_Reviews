import matplotlib.pyplot as plt

def plot_precisions(models, precisions):
    """
    Function to plot precision of various models.

    Parameters:
    - models: List of names of the models.
    - precisions: List of precision values corresponding to each model.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(models, precisions, marker='o', linestyle='-', color='b')
    plt.xlabel('Models')
    plt.ylabel('Precision')
    plt.title('Precision of Various Models')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display precision value when hovering on the line graph
    for i, txt in enumerate(precisions):
        plt.annotate(f"{txt:.2f}", (models[i], precisions[i]), fontsize=9, ha='right')

    plt.tight_layout()
    plt.show()
