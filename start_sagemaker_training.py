#!/usr/bin/env python3
"""
Module: start_sagemaker_training.py
Description: Script to start a SageMaker training job.
Author: alexdjulin
Date: 03.2025
License: MIT
"""
import os
import tarfile
import boto3
from datetime import datetime
import logging

# Set up logging
log_level = logging.INFO
logging.basicConfig(level=log_level)
logging.getLogger("botocore").setLevel(log_level)
logging.getLogger("boto3").setLevel(log_level)


def make_source_tarball(source_files, tar_name="sourcedir.tar.gz"):
    """
    Create a tar.gz file containing the specified source files.

    Args:
        source_files (list): List of file paths to include in the tarball.
        tar_name (str): Name of the tarball file.

    Returns:
        str: Name of the tarball file.
    """
    with tarfile.open(tar_name, "w:gz") as tar:
        for file in source_files:
            tar.add(file, arcname=os.path.basename(file))

    return tar_name


def upload_to_s3(file_path, bucket, prefix, region="eu-central-1"):
    """
    Upload local file to S3 and return the S3 URI.

    Args:
        file_path (str): Local file path to upload.
        bucket (str): S3 bucket name.
        prefix (str): S3 key prefix.
        region (str): AWS region name.

    Returns:
        str: S3 URI of the uploaded file.
    """
    s3_client = boto3.client("s3", region_name=region)
    file_name = os.path.basename(file_path)
    s3_key = f"{prefix}/{file_name}"

    print(f"Uploading {file_path} to s3://{bucket}/{s3_key} ...")
    s3_client.upload_file(file_path, bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"


def create_training_job(
    job_name,
    bucket,
    role_arn,
    image_uri,
    code_s3_uri,
    input_data_s3_uri,
    output_s3_uri,
    region="eu-central-1",
):
    """
    Create SageMaker training job.

    Args:
        job_name (str): Name of the training job.
        bucket (str): S3 bucket name.
        role_arn (str): ARN of the IAM role.
        image_uri (str): ECR URI of the training container.
        code_s3_uri (str): S3 URI of the source code tarball.
        input_data_s3_uri (str): S3 URI of the input data.
        output_s3_uri (str): S3 URI for the output artifacts.
        region (str): AWS region name.

    Returns:
        dict: SageMaker response object.
    """
    sm_client = boto3.client("sagemaker", region_name=region)

    hyperparams = {
        "sagemaker_program": "train.py",
        "sagemaker_submit_directory": code_s3_uri,
        "sagemaker_container_log_level": "20",
        "sagemaker_region": region
    }

    input_data_config = [{
        "ChannelName": "training",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": input_data_s3_uri,
                "S3DataDistributionType": "FullyReplicated"
            }
        }
    }]

    response = sm_client.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File"
        },
        OutputDataConfig={
            "S3OutputPath": output_s3_uri
        },
        ResourceConfig={
            "InstanceType": "ml.g5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30
        },
        InputDataConfig=input_data_config,
        StoppingCondition={
            "MaxRuntimeInSeconds": 86400  # 24 hours
        },
        HyperParameters=hyperparams
    )

    print("CreateTrainingJob response:", response)
    print(f"Training job {job_name} created.")
    return response


def main():
    """
    Main function to start a SageMaker training job for the chat moderator model.
    """
    # Project configuration
    bucket = "chat-moderator"
    role_arn = "arn:aws:iam::897722677287:role/chat-moderator-sagemaker"

    # Duplicate requirements-train.txt to requirements.txt
    with open("model/requirements-train.txt", "r") as f:
        with open("model/requirements.txt", "w") as f2:
            f2.write(f.read())

    source_files = ['model/train.py', 'model/moderator.py', 'model/requirements.txt']
    region = "eu-central-1"

    # PyTorch GPU container
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:1.10.0-gpu-py38"

    # 1) Create source code tarball
    tar_name = make_source_tarball(source_files=source_files)

    # 2) Generate unique job name
    job_name = f"chat-moderator-training-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    try:
        # 3) Upload code tarball to S3
        code_s3_uri = upload_to_s3(
            file_path=tar_name,
            bucket=bucket,
            prefix=f"{job_name}/source",
            region=region
        )

        # 4) Create training job
        create_training_job(
            job_name=job_name,
            bucket=bucket,
            role_arn=role_arn,
            image_uri=image_uri,
            code_s3_uri=code_s3_uri,
            input_data_s3_uri=f"s3://{bucket}/data",
            output_s3_uri=f"s3://{bucket}/output",
            region=region
        )

        print("Done! Check the SageMaker console for the job status.")

    except Exception as e:
        print(f"Failed to create training job: {e}")

    finally:
        # 5) Clean up local tarball and requirements.txt
        if os.path.exists(tar_name):
            os.remove(tar_name)
        if os.path.exists("model/requirements.txt"):
            os.remove("model/requirements.txt")


if __name__ == "__main__":
    main()
