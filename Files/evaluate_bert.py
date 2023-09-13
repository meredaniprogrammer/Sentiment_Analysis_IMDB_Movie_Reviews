from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

MODEL_NAME = "bert-base-uncased"
TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_reviews(reviews, labels):
    """Encode reviews using BERT tokenizer and return PyTorch Datasets."""
    encoded = TOKENIZER(reviews, truncation=True, padding=True, max_length=512)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(encoded['input_ids']),
        torch.tensor(encoded['attention_mask']),
        torch.tensor(labels)
    )
    return dataset


def train_bert(train_dataset):
    """Train the BERT model."""
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    return model


def evaluate_bert(model, test_dataset):
    """Evaluate the BERT model and return performance metrics."""
    outputs = model(test_dataset.tensors[0].to(DEVICE), attention_mask=test_dataset.tensors[1].to(DEVICE)).logits
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = test_dataset.tensors[2].numpy()

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    return preds, accuracy, precision, recall, f1
