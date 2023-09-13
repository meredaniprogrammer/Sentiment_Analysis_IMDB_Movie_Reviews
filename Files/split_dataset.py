import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.30, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
