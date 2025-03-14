#!/usr/bin/env python3
"""
Module: app_deployed.py
Description: Flask app for a chatbot with content moderation using a model deployed on SageMaker.
Author: alexdjulin
Date: 03.2025
License: MIT
"""
import boto3
import json
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
import getpass
import logging
from dotenv import load_dotenv
load_dotenv('/home/alexdjulin/chat_moderator/.env')

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
# 2) SAGEMAKER ENDPOINT MODERATION SETUP
# =============================================================================
ENDPOINT_NAME = "chat-moderator-endpoint"
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='eu-central-1')


def call_sagemaker_moderation(text: str) -> dict:
    payload = json.dumps({"text": text})
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload
    )
    # Check if the HTTP status code is 200; if not, raise an exception.
    status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode", None)
    if status_code != 200:
        raise Exception(f"Received non-200 status code: {status_code}")
    result = response["Body"].read().decode("utf-8")
    return json.loads(result)


# =============================================================================
# 3) FLASK ROUTES
# =============================================================================
@app.route('/')
def home():
    """ Home route for the chatbot app. """
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """ Route for the chatbot conversation. """
    user_message = request.json["message"]
    message_censored = False

    # -------------------------------------------------------------------------
    # Step A) Run Content Moderation
    # -------------------------------------------------------------------------
    try:
        moderation_results = call_sagemaker_moderation(user_message)
        is_inappropriate = any(score > TOXICITY_THRESHOLD for score in moderation_results.values())

        flagged_details = {}
        for category, score in moderation_results.items():
            if score > TOXICITY_THRESHOLD:
                flagged_details[category] = score
        message_censored = True

    except Exception as e:
        logging.exception(f"Error inferencing on SageMaker endpoint; continuing without moderation: {e}")
        is_inappropriate = False
        flagged_details = {}

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
        # If message_censored is False, we add the "[uncensored]" tag
        'user_message': user_message if message_censored else f"[uncensored] {user_message}",
        'ai_response': ai_response,
        'is_inappropriate': is_inappropriate,
        'flagged_categories': flagged_details
    })


if __name__ == '__main__':
    app.run(debug=False)
