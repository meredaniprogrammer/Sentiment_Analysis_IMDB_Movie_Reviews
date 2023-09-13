
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
#
#
# def train_rf_model(X_train, y_train, n_estimators=100, max_depth=None, sample_size=None):
#     """
#     Train a Random Forest Classifier.
#
#     Parameters:
#     - X_train: The feature matrix for training.
#     - y_train: The labels for training.
#     - n_estimators: The number of trees in the forest.
#     - max_depth: The maximum depth of the tree.
#     - sample_size: Number of samples to use for training. If None, use all.
#
#     Returns:
#     - model: Trained Random Forest model.
#     """
#
#     # If sample size is provided, sample the data
#     if sample_size:
#         idx = np.random.choice(np.arange(len(X_train)), sample_size, replace=False)
#         X_train = X_train[idx]
#         y_train = y_train.iloc[idx]
#
#     model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
#     model.fit(X_train, y_train)
#
#     return model


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def train_rf_model(X_train, y_train, sample_fraction=0.4, n_estimators=100, max_depth=None, class_weight='balanced', random_state=42):
    """
    Function to train a Random Forest classifier model.

    Parameters:
    X_train (array-like or sparse matrix): the training data
    y_train (array-like): the training labels
    sample_fraction (float): fraction of samples to pick for training
    n_estimators (int): The number of trees in the forest. Default is 100.
    max_depth (int, optional): The maximum depth of the tree. Default is None.
    class_weight (str, optional): Weights associated with classes. Default is 'balanced'.

    Returns:
    model (Pipeline): the trained Random Forest model encapsulated in a pipeline for scaling
    """

    # Sample a fraction of the dataset
    n_rows = X_train.shape[0]
    sample_size = int(n_rows * sample_fraction)
    indices = np.random.choice(n_rows, sample_size, replace=False)
    X_train_sampled = X_train[indices]
    y_train_sampled = y_train.iloc[indices]

    # Create a Random Forest classifier model
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight, random_state=random_state,n_jobs=-1)

    # We'll use a pipeline to scale the data and then train the model.
    # Note: Random Forest doesn't strictly require feature scaling, but we'll keep it for consistency with the SVM pipeline.
    # model = Pipeline([
    #     ("scaler", StandardScaler(with_mean=False)),
    #     ("rf_clf", rf_clf)
    # ])
    rf_clf.fit(X_train_sampled, y_train_sampled)

    # Fit the model to the training data
    #model.fit(X_train_sampled, y_train_sampled)

    return rf_clf

# You can call this function similar to how you called the SVM function
# model_rf = train_rf_model(X_train, y_train)