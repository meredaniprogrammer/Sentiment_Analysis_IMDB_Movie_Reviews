from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_auc(y_true, y_pred_proba, model_name):
    """
    Plots the ROC curve and calculates the AUC for a binary classification model.

    Args:
    - y_true (array-like): True binary labels.
    - y_pred_proba (array-like): Target scores, probabilities of the positive class.
    - model_name (str): Name of the model for title in the plot.

    Returns:
    - None: Displays the ROC curve with AUC.
    """

    # Specify 'Positive' as the positive class label
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba, pos_label='Positive')
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()
