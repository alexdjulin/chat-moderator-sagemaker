#!/usr/bin/env python3
"""
Module: inference.py
Description: SageMaker inference script for ContentModerator model.
Author: alexdjulin
Date: 03.2025
License: MIT
"""
import os
import json
import torch
import logging
from moderator import ContentModerator

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def model_fn(model_dir):
    """
    Load your trained ContentModerator model from 'model.pth'
    (which is placed in /opt/ml/model/ by default in SageMaker).

    Args:
        model_dir (str): Directory where the model files are stored.

    Returns:
        torch.nn.Module: The trained model
    """
    logger.info("Loading ContentModerator model...")

    # Instantiate the same class as in your 'moderator.py'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContentModerator().to(device)

    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # set to eval mode
    model.to(device)

    logger.info("Model loaded and ready.")
    return model


def input_fn(request_body, content_type):
    """
    Parse incoming JSON, e.g. {"text": "some user input"}

    Args:
        request_body (str): The incoming request body.
        content_type (str): The incoming request content type.

    Returns:
        dict: The parsed JSON data.

    Raises:
        ValueError: If the content type is not supported.
    """
    if content_type == "application/json":
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Run model inference using the 'predict' method in your ContentModerator class.

    Args:
        input_data (dict): The parsed JSON data.
        model (torch.nn.Module): The trained model.

    Returns:
        dict: The model's prediction results.
    """
    try:
        text = input_data["text"]
        with torch.no_grad():
            results = model.predict(text)
        return results

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return {"error": str(e)}


def output_fn(prediction, accept):
    """
    Format the dictionary output as JSON.

    Args:
        prediction (dict): The model's prediction results.
        accept (str): The incoming request accept type.

    Returns:
        str: The JSON-formatted prediction results.

    Raises:
        ValueError: If the accept type is not supported.
    """
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
