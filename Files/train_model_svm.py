
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import numpy as np


def train_svm_model(X_train, y_train, sample_fraction=0.7, C=1.0, max_iter=1000, class_weight='balanced',
                    random_state=42):
    n_rows = X_train.shape[0]
    sample_size = int(n_rows * sample_fraction)
    indices = np.random.choice(n_rows, sample_size, replace=False)
    X_train_sampled = X_train[indices]
    y_train_sampled = y_train.iloc[indices]
    # Create an SVM model
    svm_clf = LinearSVC(C=C, max_iter=max_iter, class_weight=class_weight, random_state=random_state,
                        dual=False)
    # Calibrate the SVM model to produce probability estimates
    calibrated_svm = CalibratedClassifierCV(svm_clf, method="sigmoid")
    # I'll use a pipeline to scale the data and then train the model
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # SVM benefits from feature scaling
        ("svm_calibrated", calibrated_svm)
    ])
    model.fit(X_train_sampled, y_train_sampled)
    return model
