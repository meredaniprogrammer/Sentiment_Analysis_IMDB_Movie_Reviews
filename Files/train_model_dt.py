
# from sklearn.tree import DecisionTreeClassifier
# import numpy as np
#
#
# def model_decision_tree(X_train, y_train, sample_size=None, max_depth=None, min_samples_split=2, min_samples_leaf=1,
#                 random_state=42):
#     """
#     Function to train a Decision Tree classifier.
#
#     Parameters:
#     X_train (DataFrame or sparse matrix): the training data
#     y_train (Series): the training labels
#     sample_size (int, optional): number of samples to pick for training. If None, use all data.
#     max_depth (int, optional): The maximum depth of the tree.
#     min_samples_split (int, optional): The minimum number of samples required to split an internal node.
#     min_samples_leaf (int, optional): The minimum number of samples required to be at a leaf node.
#
#     Returns:
#     model (DecisionTreeClassifier): the trained Decision Tree classifier
#     """
#
#     # Sample the dataset if sample_size is provided
#     if sample_size:
#         n_rows = X_train.shape[0]  # Get the number of rows/records
#         indices = np.random.choice(n_rows, sample_size, replace=False)
#         X_train = X_train[indices]
#         y_train = y_train.iloc[indices]
#
#     # Initialize a Decision Tree classifier with specified parameters
#     model = DecisionTreeClassifier(max_depth=max_depth,
#                                    min_samples_split=min_samples_split,
#                                    min_samples_leaf=min_samples_leaf,
#                                    random_state=random_state)
#
#     # Fit the model to the training data
#     model.fit(X_train, y_train)
#
#     return model

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def model_decision_tree(X_train, y_train, sample_size=None, max_depth=None,
                        min_samples_split=2, min_samples_leaf=1, random_state=42):
    """
    Function to train a Decision Tree classifier.

    Parameters:
    - X_train (DataFrame or sparse matrix): the training data
    - y_train (Series): the training labels
    - sample_size (int, optional): number of samples to pick for training. If None, use all data.
    - max_depth (int, optional): The maximum depth of the tree.
    - min_samples_split (int, optional): The minimum number of samples required to split an internal node.
    - min_samples_leaf (int, optional): The minimum number of samples required to be at a leaf node.

    Returns:
    - model (DecisionTreeClassifier): the trained Decision Tree classifier
    """

    # Sample the dataset if sample_size is provided using stratified sampling
    if sample_size:
        X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=sample_size,
                                                    stratify=y_train, random_state=random_state)
    else:
        X_sample, y_sample = X_train, y_train

    # Initialize a Decision Tree classifier with specified parameters
    model = DecisionTreeClassifier(max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=random_state)

    # Fit the model to the training data
    model.fit(X_sample, y_sample)

    return model
