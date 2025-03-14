#!/usr/bin/env python3
"""
Module: deploy_sagemaker_endpoint.py
Description: Script to deploy a SageMaker endpoint for the Content Moderator model.
Author: alexdjulin
Date: 03.2025
License: MIT
"""
import os
import boto3
import tarfile
import time

# Initialize SageMaker and S3 clients
region = "eu-central-1"
sagemaker_client = boto3.client("sagemaker", region_name=region)
s3_client = boto3.client("s3", region_name=region)

# AWS Config
s3_bucket = "chat-moderator"  # Match the bucket from start_training.py
s3_prefix = "models"
s3_model_path = f"s3://{s3_bucket}/{s3_prefix}/model.tar.gz"

# Define variables
MODEL_DIR = "model"
CODE_DIR = "code"
TAR_FILENAME = "model.tar.gz"
MODEL_NAME = "chat-moderator"
ENDPOINT_CONFIG_NAME = "chat-moderator-endpoint-config"
ENDPOINT_NAME = "chat-moderator-endpoint"
ROLE_ARN = "arn:aws:iam::897722677287:role/chat-moderator-sagemaker"  # Match the role from start_training.py
IMAGE_URI = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:1.10.0-gpu-py38"  # Match PyTorch version
INSTANCE_TYPE = "ml.g5.xlarge"  # Match the instance type from start_training.py


def create_model_tar():
    """Creates model.tar.gz file with the correct structure."""

    # create requirements.txt file
    with open("model/requirements-infer.txt", "r") as f:
        with open("model/requirements.txt", "w") as f2:
            f2.write(f.read())

    # list source files, make sure they exist
    source_files = ['model/moderator.py', 'model/inference.py', 'model/requirements.txt']
    for file in source_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Source file not found: {file}")

    with tarfile.open(TAR_FILENAME, "w:gz") as tar:
        # Add the PyTorch model file from the output location
        tar.add("model/model_output/model.pth", arcname="model.pth")

        # Add the source files
        for file in source_files:
            tar.add(file, arcname=f"code/{file.split('/')[-1]}")


def upload_to_s3():
    """Uploads model.tar.gz to S3."""
    try:
        s3_client.head_bucket(Bucket=s3_bucket)
    except s3_client.exceptions.ClientError:
        s3_client.create_bucket(
            Bucket=s3_bucket,
            CreateBucketConfiguration={'LocationConstraint': region}
        )

    s3_client.upload_file(TAR_FILENAME, s3_bucket, f"{s3_prefix}/model.tar.gz")
    print(f"Model uploaded to S3: {s3_model_path}")


def delete_existing_model():
    """Deletes existing SageMaker model if it exists."""
    try:
        sagemaker_client.describe_model(ModelName=MODEL_NAME)
        print(f"Model {MODEL_NAME} exists. Deleting it...")
        sagemaker_client.delete_model(ModelName=MODEL_NAME)
        time.sleep(5)
    except sagemaker_client.exceptions.ClientError:
        print(f"No existing model found: {MODEL_NAME}")


def delete_existing_endpoint_config():
    """Deletes existing endpoint configuration if it exists."""
    try:
        sagemaker_client.describe_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
        print(f"Endpoint config {ENDPOINT_CONFIG_NAME} exists. Deleting it...")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
        time.sleep(5)
    except sagemaker_client.exceptions.ClientError:
        print(f"No existing endpoint config found: {ENDPOINT_CONFIG_NAME}")


def delete_existing_endpoint():
    """Deletes existing endpoint if it exists."""
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response["EndpointStatus"]
        print(f"Endpoint {ENDPOINT_NAME} exists (Status: {status}). Deleting it...")

        if status in ["InService", "Creating", "Updating"]:
            sagemaker_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
            print("Waiting for endpoint to be deleted...")

            while True:
                try:
                    response = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
                    print(f"Still deleting... Current status: {response['EndpointStatus']}")
                    time.sleep(30)
                except sagemaker_client.exceptions.ClientError:
                    print(f"Endpoint {ENDPOINT_NAME} deleted successfully.")
                    break
        else:
            print(f"Endpoint {ENDPOINT_NAME} is already deleted.")
    except sagemaker_client.exceptions.ClientError:
        print(f"No existing endpoint found: {ENDPOINT_NAME}")


def create_model():
    """Creates the SageMaker model."""

    environment = {
        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
        'SAGEMAKER_PROGRAM': 'inference.py',
        'MODEL_PATH': '/opt/ml/model/model.pth',
        'SAGEMAKER_REGION': region
    }

    sagemaker_client.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": s3_model_path,
            "Environment": environment
        },
        ExecutionRoleArn=ROLE_ARN,
    )
    print(f"Model {MODEL_NAME} created successfully.")


def create_endpoint_config():
    """Creates an endpoint configuration."""

    sagemaker_client.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "ModelDataDownloadTimeoutInSeconds": 1200,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600
            }
        ],
    )
    print(f"Endpoint configuration {ENDPOINT_CONFIG_NAME} created successfully.")


def deploy_endpoint():
    """Deploys the endpoint."""

    sagemaker_client.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )
    print(f"Endpoint {ENDPOINT_NAME} is being created...")

    while True:
        response = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response["EndpointStatus"]
        print(f"Current endpoint status: {status}")

        if status == "InService":
            print("Endpoint is ready for inference!")
            break
        elif status == "Failed":
            print("Deployment failed. Check AWS CloudWatch logs.")
            break
        else:
            time.sleep(30)


def cleanup():
    """Cleans up deployed files."""
    if os.path.exists(TAR_FILENAME):
        os.remove(TAR_FILENAME)
    if os.path.exists("model/requirements.txt"):
        os.remove("model/requirements.txt")


if __name__ == "__main__":
    print("Cleaning up existing SageMaker resources...")
    delete_existing_endpoint()
    delete_existing_endpoint_config()
    delete_existing_model()

    print("Creating model.tar.gz...")
    create_model_tar()

    print("Uploading model.tar.gz to S3...")
    upload_to_s3()

    print("Creating SageMaker Model...")
    create_model()

    print("Creating Endpoint Configuration...")
    create_endpoint_config()

    print("Deploying Endpoint...")
    deploy_endpoint()

    print("Cleaning up...")
    cleanup()

    print("Deployment complete!")
