from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def predict_evaluate(model, X_test, y_test):
    # Predict the labels of the test set
    y_pred = model.predict(X_test)

    # Compute various performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return y_pred, accuracy, precision, recall, f1
