#!/usr/bin/env python3
"""
Module: infer_on_sagemaker_endpoint.py
Description: Script to send a text sample to a SageMaker endpoint for toxicity prediction.
Author: alexdjulin
Date: 03.2025
License: MIT
"""
import boto3
import json

REGION = "eu-central-1"
ENDPOINT_NAME = "chat-moderator-endpoint"


def predict_toxicity_sagemaker(endpoint_name, text, region=REGION):
    """
    Send a single text string to the specified SageMaker endpoint
    and return the toxicity classification scores.

    Args:
        endpoint_name (str): The name of your deployed SageMaker endpoint.
        text (str): The input text to classify.
        region (str): AWS region where your endpoint is deployed.

    Returns:
        dict: Dictionary of toxicity categories with their predicted scores.
    """
    # Create a SageMaker Runtime client
    runtime_client = boto3.client("sagemaker-runtime", region_name=region)

    # Prepare JSON payload
    payload = json.dumps({"text": text})

    # Invoke endpoint
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload
    )

    # Decode JSON response
    body = response["Body"].read().decode("utf-8")
    results = json.loads(body)

    return results


if __name__ == "__main__":

    # create a loop to send messages and get predictions
    while True:
        sample_text = input("--\nEnter a text sample to check for toxicity (or 'exit' to quit):\n")
        if sample_text.lower() == "exit":
            break

        scores = predict_toxicity_sagemaker(ENDPOINT_NAME, sample_text)
        print("Toxicity Scores:", scores)
