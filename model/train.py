#!/usr/bin/env python3
"""
Module: train.py
Description: Content Moderator training script.
Author: alexdjulin
Date: 03.2025
License: MIT
"""
import argparse
import torch
import pandas as pd
import logging
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from moderator import ContentModerator
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# Configure logging to display info and save logs to a file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ToxicCommentsDataset(Dataset):
    """
    Custom PyTorch Dataset for handling toxic comment classification.

    Args:
        texts (list): List of comment texts.
        labels (list): Corresponding labels for toxicity classification.
        tokenizer (transformers.BertTokenizer): Tokenizer to process text inputs.
        max_len (int, optional): Maximum tokenized length of the text. Defaults to 512.
    """

    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves a single data point from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing input tensors for the model.
        """
        text = str(self.texts[idx])  # Convert to string in case of missing values
        label = self.labels[idx]  # Get the corresponding label

        # Tokenize and encode the text using BERT tokenizer
        # returns a tensor of shape [1, max_len], where 1 is the batch size. Will need to flatten it later
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
            max_length=self.max_len,  # Truncate if exceeding max length
            padding='max_length',  # Pad to max length
            truncation=True,  # Ensure the text is not too long
            return_tensors='pt'  # Return PyTorch tensors
        )

        # return a dictionary with the input ids, attention mask, and labels
        return {
            'input_ids': encoding['input_ids'].flatten(),  # Tokenized input IDs (tensor of token ids)
            'attention_mask': encoding['attention_mask'].flatten(),  # Attention mask (tensor of 1s and 0s)
            'labels': torch.FloatTensor(label)  # Convert labels to float tensor (required to compute Binary Cross-Entropy Loss)
        }


def train_model(data_dir, model_dir, num_epochs=3):
    """
    Trains the Content Moderator model using the toxic comments dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        model_dir (str): Directory to save the trained model.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 1.
    """

    # Load dataset from S3 (SageMaker automatically downloads the data to /opt/ml/input/data/training)
    logging.info("Loading dataset from S3...")
    data_path = os.path.join(data_dir, "processed_train.csv")
    df = pd.read_csv(data_path)

    # Extract text and corresponding labels
    texts = df['text'].values
    labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    logging.info("Splitting dataset into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

    # Load BERT tokenizer for text processing
    logging.info("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create dataset objects for training
    train_dataset = ToxicCommentsDataset(X_train, y_train, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # After creating train_loader
    total_batches = len(train_loader)
    print(f"Total batches per epoch: {total_batches}")

    # Initialize the Content Moderator model
    logging.info("Initializing Content Moderator model...")
    model = ContentModerator()

    # Define optimizer (AdamW) and loss function (Binary Cross-Entropy Loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCELoss()  # BCELoss for multi-label classification

    # Move model to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logging.info(f"Using device: {device}")

    # START TRAINING
    logging.info(f"Starting training for {num_epochs} epochs...")

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(model_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0  # Track total loss for the epoch
        batch_count = 0  # Track number of batches processed

        for batch in train_loader:
            optimizer.zero_grad()  # Reset gradients before each batch

            # Move batch tensors to the same device as the model
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass: compute model predictions
            outputs = model(input_ids, attention_mask)

            # Compute loss
            loss = criterion(outputs, labels)

            # Log batch loss
            writer.add_scalar('Loss/batch', loss.item(), epoch * len(train_loader) + batch_count)

            epoch_loss += loss.item()
            batch_count += 1

            # Backpropagation: compute gradients
            loss.backward()
            optimizer.step()  # Update model parameters

            # Log batch loss every 10 batches
            if batch_count % 10 == 0:
                logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Batch {batch_count}/{total_batches} ({(batch_count/total_batches)*100:.1f}%), Loss: {loss.item():.4f}")

        # Compute and log average epoch loss
        avg_epoch_loss = epoch_loss / batch_count
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

    # Close the TensorBoard writer
    writer.close()

    # Save model weights
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Training complete. Model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    # Training specific arguments
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=2e-5)

    args = parser.parse_args()

    # Start training
    train_model(args.data_dir, args.model_dir)
