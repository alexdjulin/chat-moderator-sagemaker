#!/usr/bin/env python3
"""
Module: moderator.py
Description: Content Moderator Model Definition.
This module defines a BERT-based model for toxic comment classification.
The model uses BERT's pre-trained weights and adds a classification head
for multi-label toxicity detection.
Author: alexdjulin
Date: 03.2025
License: MIT
"""
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class ContentModerator(nn.Module):
    """
    Neural network model for toxic comment classification.

    Architecture:
        1. BERT base model for text encoding
        2. Dropout layer for regularization
        3. Linear layer for classification
        4. Sigmoid activation for multi-label output

    The model outputs probabilities for 6 toxicity types:
        - toxic
        - severe_toxic
        - obscene
        - threat
        - insult
        - identity_hate

    Example output:
        {
            'toxic': 0.1,
            'severe_toxic': 0.02,
            'obscene': 0.8,
            'threat': 0.0,
            'insult': 0.3,
            'identity_hate': 0.9
        }
    """

    def __init__(self):
        """
        Initialize the model components.
        Uses the 'bert-base-uncased' pre-trained model as the base encoder.
        """
        super().__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Classification head with 768->6 dimensions (BERT's output dim -> number of labels)
        self.classifier = nn.Linear(768, 6)

        # Sigmoid for multi-label classification (each label is independent)
        self.sigmoid = nn.Sigmoid()

        # Load BERT tokenizer for inference
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tokenized input sequences
            attention_mask (torch.Tensor): Attention mask for BERT

        Returns:
            torch.Tensor: Probabilities for each toxicity type (shape: batch_size x 6)
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token representation (first token)
        pooled_output = outputs.pooler_output

        # Pass through classification head
        logits = self.classifier(pooled_output)

        # Convert to probabilities between 0 and 1 (floats, in-between values are possible)
        return self.sigmoid(logits)  # Ex: [0.5, 0.2, 0.0, 0.0, 0.1, 0.9]

    def predict(self, text):
        """
        Predict toxicity probabilities for a given text.

        Args:
            text (str): Input text to classify

        Returns:
            dict: Dictionary mapping toxicity types to their probabilities
        """
        # Set model to evaluation mode
        self.eval()

        # Determine the device of the model (CPU or GPU)
        device = next(self.parameters()).device

        # Deactivate gradient calculation
        with torch.no_grad():
            # Tokenize input text
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            # Move tokenized inputs to the model's device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Get model predictions (this calls the forward method)
            outputs = self(input_ids, attention_mask)

            # Remove extra dimensions, move to CPU, and convert to numpy array
            predictions = outputs.squeeze().detach().cpu().numpy()

        # Map predictions to toxicity categories
        categories = [
            'toxic',
            'severe_toxic',
            'obscene',
            'threat',
            'insult',
            'identity_hate'
        ]

        # Create a dictionary mapping each category to its predicted probability
        results = {
            cat: float(pred)
            for cat, pred in zip(categories, predictions)
        }

        return results
