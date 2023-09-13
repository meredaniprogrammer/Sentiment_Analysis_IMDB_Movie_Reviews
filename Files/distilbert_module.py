# distilbert_module.py

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


def train_evaluate_distilbert(X_train, y_train, X_test, y_test):
    # 1. Load DistilBERT Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # 2. Tokenize the data and get required input format
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=512)

    # 3. Convert to PyTorch DataLoaders
    class MovieReviewDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = MovieReviewDataset(train_encodings, y_train.tolist())
    test_dataset = MovieReviewDataset(test_encodings, y_test.tolist())

    # 4. Initialize model, and set up training args
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
    )

    # 5. Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()

    # 6. Evaluate the model
    predictions = trainer.predict(test_dataset).predictions.argmax(-1)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')

    return accuracy, precision, recall, f1
