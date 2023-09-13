# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# import numpy as np
#
#
# def train_model_logistic(X_train, y_train, sample_size=None):
#     """
#     Function to train a Logistic Regression model.
#
#     Parameters:
#     X_train (DataFrame or sparse matrix): the training data
#     y_train (Series): the training labels
#     sample_size (int, optional): number of samples to pick for training. If None, use all data.
#
#     Returns:
#     model (LogisticRegression): the trained Logistic Regression model
#     """
#
#     # Sample the dataset if sample_size is provided

#     if sample_size:
#         n_rows = X_train.shape[0]  # Get the number of rows/records
#         indices = np.random.choice(n_rows, sample_size, replace=False)
#         X_train = X_train[indices]
#         y_train = y_train.iloc[indices]
#
#     # Initialize a Logistic Regression model
#     log_reg = LogisticRegression(random_state=42, max_iter=10000)  # Increase max_iter if necessary for convergence
#
#     # We'll use a pipeline to scale the data and then train the model
#     model = Pipeline([
#         ("scaler", StandardScaler(with_mean=False)),  # Logistic Regression benefits from feature scaling
#         ("log_reg", log_reg)
#     ])
#
#     # Fit the model to the training data
#     model.fit(X_train, y_train)
#
#     return model

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


def train_model_logistic(X_train, y_train, sample_size=None, C=0.1, penalty='l2', max_iter=10000, class_weight='balanced', random_state=42):
    """
    Function to train a Logistic Regression model.

    Parameters:
    X_train (DataFrame or sparse matrix): the training data
    y_train (Series): the training labels
    sample_size (int, optional): number of samples to pick for training. If None, use all data.
    C (float, optional): Inverse of regularization strength; smaller values specify stronger regularization.
    penalty (str, optional): Used to specify the norm used in the penalization ('l1', 'l2', 'elasticnet').

    Returns:
    model (Pipeline): the trained Logistic Regression model inside a pipeline with a scaler
    """

    # Sample the dataset if sample_size is provided
    if sample_size:
        n_rows = X_train.shape[0]  # Get the number of rows/records
        indices = np.random.choice(n_rows, sample_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train.iloc[indices]

    # Initialize a Logistic Regression model
    log_reg = LogisticRegression(C=C, penalty=penalty, random_state=random_state, max_iter=max_iter, class_weight=class_weight)

    # We'll use a pipeline to scale the data and then train the model
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # Logistic Regression benefits from feature scaling
        ("log_reg", log_reg)
    ])

    # Fit the model to the training data
    model.fit(X_train, y_train)

    return model
