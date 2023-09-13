from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_model_svm(X_train, y_train):
    """
    Function to train a Support Vector Machine (SVM) model

    Parameters:
    X_train (DataFrame): the training data
    y_train (Series): the training labels

    Returns:
    model (SVC): the trained SVM model
    """
    # Initialize an SVM model
    svm_clf = svm.SVC(kernel='linear', C=1.0, random_state=42)

    # We'll use a pipeline to scale the data and then train the model
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # SVM requires feature scaling
        ("svm_clf", svm_clf)
    ])

    # Fit the model to the training data
    model.fit(X_train, y_train)

    return model
