import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def encode_sentiment(sentiment):
    """
    Convert string sentiment to numeric.

    Parameters:
        sentiment (str): The sentiment value.

    Returns:
        int: Encoded sentiment value.
    """
    sentiment_mapping = {
        'Positive': 1,
        'Negative': 0,
        # Add other sentiments here if required
    }

    if sentiment not in sentiment_mapping:
        raise ValueError(f"Unexpected sentiment value: {sentiment}")

    return sentiment_mapping[sentiment]


def prepare_data(X, y):
    """
    Prepares the data for LightGBM training.

    Parameters:
        X : Input data or features
        y : Target labels

    Returns:
        X : Prepared input data or features
        y : Prepared target labels
    """
    # Convert X to float32
    X = X.astype(np.float32)

    # Convert y to integer encoding if it's not already numeric
    if y.dtype == 'object':
        y = np.array([encode_sentiment(s) for s in y])

    return X, y


def train_model_lgbm(X_train, y_train, params=None):
    """
    Train a LightGBM model on the provided training data.

    Parameters:
        X_train : training data
        y_train : training labels
        params : LightGBM parameters (optional)

    Returns:
        model : trained LightGBM model
    """
    # Prepare the data
    X_train, y_train = prepare_data(X_train, y_train)

    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
        }

    d_train = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, d_train, 100)

    return model


def predict_evaluate_lgbm(model, X_test, y_test):
    """
    Predict using the trained model and evaluate its performance.

    Parameters:
        model : trained LightGBM model
        X_test : test data
        y_test : test labels

    Returns:
        y_pred : predicted labels
        accuracy, precision, recall, f1 : performance metrics
    """
    # Prepare the data (Note: We only convert X_test to float32 here as y_test will be used for evaluation)
    X_test = X_test.astype(np.float32)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

    # Ensure y_test is of type int before evaluation
    if y_test.dtype == 'object':
        y_test = np.array([encode_sentiment(s) for s in y_test])

    accuracy = round(accuracy_score(y_test, y_pred_binary),2)
    precision = round(precision_score(y_test, y_pred_binary),2)
    recall = round(recall_score(y_test, y_pred_binary),2)
    f1 = round(f1_score(y_test, y_pred_binary),2)

    return y_pred_binary, accuracy, precision, recall, f1