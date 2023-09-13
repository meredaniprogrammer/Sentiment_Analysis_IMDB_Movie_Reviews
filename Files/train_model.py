from sklearn.naive_bayes import MultinomialNB

def train_model(X_train, y_train):
    # Create a Multinomial Naive Bayes model
    model = MultinomialNB()
    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model
