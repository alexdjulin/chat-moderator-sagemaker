#!/usr/bin/env python3
"""
Module: download_dataset.py
Description: Script to download the Toxic Comments dataset from Kaggle.
Author: alexdjulin
Date: 03.2025
License: MIT
"""
import os
import kaggle
import zipfile
import pandas as pd
from pathlib import Path


def download_toxic_dataset():
    """
    Downloads and prepares the Toxic Comments dataset from Kaggle.
    Requires Kaggle API credentials to be set up (~/.kaggle/kaggle.json)
    """
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    # Check if dataset already exists
    if (data_dir / 'train.csv').exists():
        print("Dataset already downloaded and extracted.")
        return

    print("Downloading Toxic Comments dataset from Kaggle...")
    try:
        # Download the dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'julian3833/jigsaw-toxic-comment-classification-challenge',
            path=data_dir,
            quiet=False
        )

        # Extract the zip file
        zip_path = data_dir / 'jigsaw-toxic-comment-classification-challenge.zip'
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Clean up the zip file
        os.remove(zip_path)

        # Verify the data
        train_df = pd.read_csv(data_dir / 'train.csv')
        print(f"Dataset downloaded successfully. Shape: {train_df.shape}")
        print("\nColumn names:", train_df.columns.tolist())
        print("\nSample toxic comments distribution:")
        print(train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum())

    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed kaggle package: pip install kaggle")
        print("2. Created a Kaggle account")
        print("3. Generated an API token from https://www.kaggle.com/account")
        print("4. Placed kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        print("5. Set appropriate permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)")


if __name__ == '__main__':
    download_toxic_dataset()
