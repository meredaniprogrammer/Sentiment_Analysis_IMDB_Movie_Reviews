# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
#
# def plot_confusion_matrices(y_test, preds, model_names):
#     """
#     Plot confusion matrices for multiple models side by side.
#
#     Parameters:
#     - y_test: True labels
#     - preds: List of predictions from models
#     - model_names: List of model names for labeling purposes
#     """
#
#     n_models = len(model_names)
#     plt.figure(figsize=(15, 5 * n_models))
#
#     for i, (pred, model_name) in enumerate(zip(preds, model_names)):
#         plt.subplot(n_models, 1, i + 1)
#         matrix = confusion_matrix(y_test, pred)
#         sns.heatmap(matrix, annot=True, fmt='g', cmap="YlGnBu",
#                     xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
#         plt.title(f"Confusion Matrix for {model_name}")
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#
#     plt.tight_layout()
#     plt.show()
#


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.widgets import Slider


def plot_confusion_matrices(y_test, preds, model_names):
    """
    Plot confusion matrices for multiple models with a slider to navigate between them.

    Parameters:
    - y_test: True labels
    - preds: List of predictions from models
    - model_names: List of model names for labeling purposes
    """

    n_models = len(model_names)
    current_index = [0]

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.75, bottom=0.25)

    slider_axis = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(slider_axis, 'Model Index', 0, n_models - 1, valinit=0, valstep=1, valfmt='%i')

    def update(val):
        index = int(slider.val)
        ax.clear()
        matrix = confusion_matrix(y_test, preds[index])
        sns.heatmap(matrix, annot=True, fmt='g', cmap="YlGnBu",
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax)
        ax.set_title(f"Confusion Matrix for {model_names[index]}")
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(None)
    plt.show()


