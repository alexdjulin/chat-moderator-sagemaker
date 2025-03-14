#!/usr/bin/env python3
"""
Module: app_local.py
Description: Flask app for a chatbot with content moderation using a local model.
Author: alexdjulin
Date: 03.2025
License: MIT
"""
import boto3
import json
import torch
import argparse
from flask import Flask, render_template, request, jsonify
from model.moderator import ContentModerator
from openai import OpenAI
import os
import getpass
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

app = Flask(__name__)
TOXICITY_THRESHOLD = 0.5  # Threshold for flagging inappropriate content

# =============================================================================
# 1) CONFIGURE OPENAI
# =============================================================================
client = OpenAI()
if not os.environ.get("OPENAI_API_KEY", ""):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key: ")

# Store a simple conversation context
messages = [
    {
        "role": "system",
        "content": """You are a helpful assistant. Always respond politely and concisely.
        If the user inputs any toxic, obscene or insulting content, ask them politely to
        respect the chat's regulations or they will be banned.
        """
    }
]

# =============================================================================
# 2) LOCAL MODEL MODERATION SETUP
# =============================================================================
model = ContentModerator()
model_path = "model/model_output/model.pth"

device = torch.device('cpu')  # CPU inference
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def call_local_moderation(text: str) -> dict:
    return model.predict(text)


# =============================================================================
# 3) SAGEMAKER ENDPOINT MODERATION SETUP
# =============================================================================
ENDPOINT_NAME = "chat-moderator-endpoint"
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='eu-central-1')


def call_sagemaker_moderation(text: str) -> dict:
    """ Infer on SageMaker endpoint for content moderation.

    Args:
        text (str): The input text to classify.

    Returns:
        dict: Dictionary of toxicity categories with their predicted scores.
    """
    payload = json.dumps({"text": text})
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload
    )
    result = response["Body"].read().decode("utf-8")
    return json.loads(result)


# We'll define a global variable `moderation_mode`, set via argparse
moderation_mode = "local"  # default


@app.route('/')
def home():
    """ Render the home page. """
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """ Handle the chat conversation. """
    user_message = request.json["message"]

    # -------------------------------------------------------------------------
    # Step A) Run Content Moderation
    # -------------------------------------------------------------------------
    if moderation_mode == "sagemaker":
        moderation_results = call_sagemaker_moderation(user_message)
    else:
        # Default local
        moderation_results = call_local_moderation(user_message)

    # Define a threshold for flagging inappropriate content
    is_inappropriate = any(score > TOXICITY_THRESHOLD for score in moderation_results.values())

    flagged_details = {}
    for category, score in moderation_results.items():
        if score > TOXICITY_THRESHOLD:
            flagged_details[category] = score

    # -------------------------------------------------------------------------
    # Step B) Append the user's message to conversation context
    # -------------------------------------------------------------------------
    messages.append({"role": "user", "content": user_message})

    # -------------------------------------------------------------------------
    # Step C) Get AI Response from GPT
    # -------------------------------------------------------------------------
    ai_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    ).choices[0].message.content

    # -------------------------------------------------------------------------
    # Step D) Append assistant response to conversation
    # -------------------------------------------------------------------------
    messages.append({"role": "assistant", "content": ai_response})

    # -------------------------------------------------------------------------
    # Step E) Return JSON to the front-end
    # -------------------------------------------------------------------------
    return jsonify({
        'user_message': user_message,
        'ai_response': ai_response,
        'is_inappropriate': is_inappropriate,
        'flagged_categories': flagged_details
    })


if __name__ == '__main__':
    # 1. Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', choices=['local', 'sagemaker'],
                        default='local', help="Choose 'local' or 'sagemaker'")
    args = parser.parse_args()

    # 2. Set the global variable for moderation mode
    moderation_mode = args.model

    # 3. Run the Flask app
    print(f"Running app with moderation mode: {moderation_mode}")
    app.run(debug=False)
