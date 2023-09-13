import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
TOKENIZER_PATH = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)


def encode_data(texts, labels, max_length=512):
    """Encode text data into token ids for training."""
    encoded_texts = tokenizer.batch_encode_plus(texts, truncation=True, padding='max_length', max_length=max_length)
    return torch.tensor(encoded_texts['input_ids']), torch.tensor(labels)


def fine_tune_gpt2(data_path, model_save_path, epochs=4, batch_size=8, lr=5e-5, max_length=512):
    """Fine-tune GPT-2 model for classification."""

    # Load the data
    data = pd.read_csv(data_path)
    X = data['Review']
    y = data['Sentiment']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode the training data
    input_ids, train_labels = encode_data(X_train, y_train, max_length)

    # Prepare the DataLoader
    dataset = TensorDataset(input_ids, train_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create a GPT-2 classification model instance
    num_labels = len(set(y_train))
    model = GPT2ForSequenceClassification.from_pretrained(TOKENIZER_PATH, num_labels=num_labels)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            model.zero_grad()
            outputs = model(b_input_ids, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}")

    # Save the fine-tuned model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


if __name__ == "__main__":
    DATA_PATH = 'C:\\Main_Research\\Original_Datatset\\Reviews.csv'
    MODEL_PATH = "C:\\Main_Research\\Fine_Tuned_GPT2_Model"
    fine_tune_gpt2(DATA_PATH, MODEL_PATH)
