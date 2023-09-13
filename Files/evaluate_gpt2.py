import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load a trained model and vocabulary
MODEL_PATH = 'C:\\Main_Research\\GPT2Model'  # change this path accordingly
TOKENIZER_PATH = "gpt2-medium"  # or any other size you used (small, medium, large, etc.)

#model = GPT2ForSequenceClassification.from_pretrained(MODEL_PATH)
model = GPT2ForSequenceClassification.from_pretrained(MODEL_PATH, use_auth_token=True)

tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_gpt2(texts):
    """
    Predict class labels using the fine-tuned GPT-2 model.

    Args:
    - texts (list): List of text instances.

    Returns:
    - predictions (list): Predicted class labels.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    return predictions.cpu().tolist()


def evaluate_gpt2(X_test, y_test):
    """
    Evaluate the fine-tuned GPT-2 model on test data.

    Args:
    - X_test (list): List of test texts.
    - y_test (list): True class labels for test data.

    Returns:
    - metrics (dict): Dictionary of performance metrics.
    """
    y_pred = predict_gpt2(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted'),
        "classification_report": classification_report(y_test, y_pred)
    }
    return metrics
