# AWS MLOps — S3 Data Pipeline

## Overview

A Python-based AWS MLOps mini-project demonstrating secure cloud authentication, object storage, and dataset integrity verification.

## Technologies

- Python
- boto3
- AWS IAM Identity Center
- Amazon S3
- Git

## Features

- Reproducible synthetic customer churn dataset
- AWS SSO authentication
- Upload and download using boto3
- SHA-256 integrity verification
- Private S3 storage with SSE-S3 encryption

## Running Locally

1. Create and activate a Python virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Configure an AWS CLI SSO profile named `mlops-dev`.
4. Provision the required private S3 bucket with appropriate IAM permissions.
5. Run `python create_dataset.py`.
6. Run `python s3_pipeline.py`.

## Future Improvements

- ML model training
- Docker containerization
- Amazon ECR and ECS deployment
- CI/CD
- Model monitoring and retraining

## AWS Cost Management

AWS resources are created for temporary learning exercises and cleaned up afterward to minimize costs.