import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from merge_datasets import merge_datasets
from text_preprocessing import clean_text
from save_predictions import save_predictions
from reduce_dataset import reduce_dataset
from predict_reviews import predict_reviews
from sklearn.feature_extraction.text import CountVectorizer
from split_dataset import split_data
from train_model import train_model
from predict_evaluate import predict_evaluate
from train_model_svm import train_svm_model
from train_rf_model import train_rf_model
from train_lg_model import train_model_logistic
from train_model_dt import model_decision_tree
from plot_metrics import plot_metrics
from pie_chart_metrics import plot_pie_metrics
from evaluate_bert import encode_reviews, train_bert, evaluate_bert
from generate_wordcloud import generate_wordcloud
from positive_wordcloud import generate_positive_wordcloud
from roc_auc_evaluation import plot_roc_auc
from plot_accuracies import plot_accuracy
from plot_precision import plot_precision
from plot_recall import plot_recall
from f1_line_plot import plot_f1_line
from plot_confusion_matrices import plot_confusion_matrices
from pie_chart_sentiment import plot_sentiment_distribution
from lightgbm_model import train_model_lgbm, predict_evaluate_lgbm
from lightgbm_model import encode_sentiment
from visualize_performance import plot_performance_pie_chart
from plot_pre import plot_precisions
from plot_f1_score import plot_f1score
from afinn_module import compute_afinn_scores
from sentiwordnet_scoring import compute_sentiwordnet_score
from bingliu_opinion_lexicon_scoring import compute_bingliu_score
from negative_wordcloud import generate_negative_wordcloud


def create_features_labels(X_train, X_test, y_train, y_test):
    # Initialize a CountVectorizer object
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

    # Fit the vectorizer to the training data and transform the training data
    X_train_bow = vectorizer.fit_transform(X_train)

    # Transform the test data using the fitted vectorizer
    X_test_bow = vectorizer.transform(X_test)

    # Return the transformed training and test data, and labels
    return X_train_bow, X_test_bow, y_train, y_test
    # Return the transformed training and test data, and labels


def main():
    dataset1_path = 'C:\\Main_Research\\Original_Datatset\\rotten_tomatoes_movie_reviews.csv'
    dataset2_path = 'C:\\Main_Research\\Original_Datatset\\rotten_tomatoes_movies.csv'
    output_path = 'C:\\Main_Research\\Original_Datatset\\moviereviews.csv'
    reduced_output_path = 'C:\\Main_Research\\Original_Datatset\\Reviews.csv'
    predicted_output_path = 'C:\\Main_Research\\Original_Datatset\\predictreviews.csv'
    # Load the reduced dataset
    reduced_data = pd.read_csv(reduced_output_path)

    # Assuming that the ratings column is named "Rating"
    average_rating = reduced_data['Rating'].mean()

    print(f"The average rating is: {average_rating}")
    # Merge datasets and create the new dataset
    merge_datasets(dataset1_path, dataset2_path, output_path)

    # Load the new dataset
    new_data = pd.read_csv(output_path)

    # Preprocess the text data
    new_data['Review'] = new_data['Review'].apply(clean_text)
    # function to plot positive vs negative sentiments in a pie chart
    # Show the negative word cloud

    plot_sentiment_distribution(new_data)
    # Reduce the dataset and save to new file
    reduce_dataset(output_path, reduced_output_path)

    # Prepare the dataset for prediction and save to new file
    predict_reviews(output_path, predicted_output_path)

    new_data['Review'] = new_data['Review'].apply(' '.join)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = split_data(new_data['Review'], new_data['Sentiment'])

    # Create features and labels for training and test sets
    X_train_bow, X_test_bow, y_train, y_test = create_features_labels(X_train, X_test, y_train, y_test)

    # Train the Multinomial Naive Bayes model
    print("Starting NBM model training...")
    model_nbm = train_model(X_train_bow, y_train)
    print("NBM model training complete!")
    y_pred_nbm, accuracy_nbm, precision_nbm, recall_nbm, f1_nbm = predict_evaluate(model_nbm, X_test_bow, y_test)
    print(f"NBM Metrics - Accuracy: {accuracy_nbm}, Precision: {precision_nbm}, Recall: {recall_nbm}, F1: {f1_nbm}")

    # # Train the SVM model
    print("Starting SVM model training...")
    model_svm = train_svm_model(X_train_bow, y_train)
    print("SVM model training complete!")
    y_pred_svm, accuracy_svm, precision_svm, recall_svm, f1_svm = predict_evaluate(model_svm, X_test_bow, y_test)
    print(f"SVM Metrics - Accuracy: {accuracy_svm}, Precision: {precision_svm}, Recall: {recall_svm}, F1: {f1_svm}")

    # # Train the Random Forest model
    print("Starting RF model training...")
    model_rf = train_rf_model(X_train_bow, y_train)
    print("RF model training complete!")
    y_pred_rf, accuracy_rf, precision_rf, recall_rf, f1_rf = predict_evaluate(model_rf, X_test_bow, y_test)
    print(f"RF Metrics - Accuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}, F1: {f1_rf}")

    # # Train the Logistic Regression model
    print("Starting LR model training...")
    model_lr = train_model_logistic(X_train_bow, y_train)
    print("LR model training complete!")
    y_pred_lr, accuracy_lr, precision_lr, recall_lr, f1_lr = predict_evaluate(model_lr, X_test_bow, y_test)
    print(f"LR Metrics - Accuracy: {accuracy_lr}, Precision: {precision_lr}, Recall: {recall_lr}, F1: {f1_lr}")

    # # Train the Decision Tree model
    print("Starting DT model training...")
    model_dt = model_decision_tree(X_train_bow, y_train)
    print("DT model training complete!")
    y_pred_dt, accuracy_dt, precision_dt, recall_dt, f1_dt = predict_evaluate(model_dt, X_test_bow, y_test)
    print(f"DT Metrics - Accuracy: {accuracy_dt},"
          f"Precision: {precision_dt}, Recall: {recall_dt}, F1: {f1_dt}")
    # Train and evaluate LGM model
    # Convert the data to float32 dtype (Correction made here)
    # Train LightGBM model
    model_lgbm = train_model_lgbm(X_train_bow, y_train)
    # Predict and evaluate the model on the test set
    y_pred, accuracy, precision, recall, f1 = predict_evaluate_lgbm(model_lgbm, X_test_bow, y_test)

    # Printing the results (you can format this as per your requirements)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    scipy.sparse.save_npz('C:\\Main_Research\\Original_Datatset\\bow_reviews.npz', X_train_bow)
    # Where you want to generate and display the word cloud:
    all_reviews = ' '.join(new_data['Review'])
    generate_wordcloud(all_reviews)

    # Show the positive word cloud
    dataset_path = 'C:\\Main_Research\\Original_Datatset\\moviereviews.csv'
    generate_positive_wordcloud(dataset_path)
    #generate_negative_wordcloud(dataset_path)
    # Names of all models
    models = ["NBM", "SVM", "RF", "LR", "DT"]

    # Their respective performance metrics
    accs = [accuracy_nbm, accuracy_svm, accuracy_rf, accuracy_lr, accuracy_dt]
    precs = [precision_nbm, precision_svm, precision_rf, precision_lr, precision_dt]
    recs = [recall_nbm, recall_svm, recall_rf, recall_lr, recall_dt]
    f1s = [f1_nbm, f1_svm, f1_rf, f1_lr, f1_dt]

    # Plot the metrics
    plot_metrics(models, accs, precs, recs, f1s)
    plot_pie_metrics(models, accs, precs, recs, f1s)
    #AUC
    # For the NBM model:
    y_pred_proba_nbm = model_nbm.predict_proba(X_test_bow)[:, 1]
    plot_roc_auc(y_test, y_pred_proba_nbm, 'NBM')

    # For the SVM model:
    # Note: Getting predicted probabilities from SVM requires setting `probability=True` during training.
    #y_pred_proba_svm = model_svm.predict_proba(X_test_bow)[:, 1]
    y_pred_proba_svm = model_svm.predict_proba(X_test_bow)[:, 1]
    plot_roc_auc(y_test, y_pred_proba_svm, 'SVM')
    # For the RF model:
    y_pred_proba_rf = model_rf.predict_proba(X_test_bow)[:, 1]
    plot_roc_auc(y_test, y_pred_proba_rf, 'RF')

    # For the LR model:
    y_pred_proba_lr = model_lr.predict_proba(X_test_bow)[:, 1]
    plot_roc_auc(y_test, y_pred_proba_lr, 'LR')

    # For the DT model:
    y_pred_proba_dt = model_dt.predict_proba(X_test_bow)[:, 1]
    plot_roc_auc(y_test, y_pred_proba_dt, 'DT')
    # For the RF model:
    y_pred_proba_rf = model_rf.predict_proba(X_test_bow)[:, 1]
    plot_roc_auc(y_test, y_pred_proba_rf, 'RF')

    # For the LR model:
    y_pred_proba_lr = model_lr.predict_proba(X_test_bow)[:, 1]
    plot_roc_auc(y_test, y_pred_proba_lr, 'LR')

    # For the DT model:
    y_pred_proba_dt = model_dt.predict_proba(X_test_bow)[:, 1]
    plot_roc_auc(y_test, y_pred_proba_dt, 'DT')
    #plot accuracy
    plot_accuracy(models, accs)
    # Names of all models
    precision_models = ["NBM", "SVM", "RF", "LR", "DT"]
    precs = [precision_nbm, precision_svm, precision_rf, precision_lr, precision_dt]

    # Plot the precision
    plot_precision(precision_models, precs)
    # Plot the precision
    plot_precisions(precision_models, precs)
    #plot recall
    plot_recall(models, recs)
    #plot F1-Score
    plot_f1_line(models, f1s)
    plot_f1score(models, f1s)
    # List of all model predictions
    all_preds = [y_pred_nbm, y_pred_svm, y_pred_rf, y_pred_lr, y_pred_dt]
    # Plot confusion matrices
    plot_confusion_matrices(y_test, all_preds, models)
    # Call the function to plot performance pie chart
    plot_performance_pie_chart(accuracy, precision, recall, f1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by the user. Exiting...")
