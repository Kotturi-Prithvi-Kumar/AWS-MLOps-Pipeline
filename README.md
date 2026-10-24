## AWS MLOps — Customer Churn Prediction Pipeline

An end-to-end machine learning project covering data ingestion, validation, model training, evaluation, containerization, and AWS deployment.

### Technology Stack

Python, Pandas, scikit-learn, FastAPI, Docker, pytest, AWS S3, Amazon ECR, Amazon EC2, IAM, Git, GitHub.

### Architecture

Synthetic Customer Dataset → Data Validation → Model Training → Model Evaluation → FastAPI → Docker → Amazon ECR → Amazon EC2

### Features Implemented

- Generated a reproducible synthetic customer churn dataset with 100 records.
- Built an S3 pipeline for uploading and downloading datasets with SHA-256 integrity verification.
- Implemented schema validation, data quality checks, and automated tests using pytest.
- Trained and evaluated Logistic Regression models using accuracy, recall, and F1-score.
- Implemented model versioning and experiment comparison.
- Built a FastAPI inference service with input validation and a health-check endpoint.
- Containerized the application using Docker and published the versioned image to Amazon ECR.
- Deployed the Docker container on an Amazon EC2 t2.micro instance running Amazon Linux 2023.
- Configured an IAM instance role, security group rules, SSH access, and Docker networking.
- Verified application startup, HTTP health checks, Swagger UI access, and container resource utilization.

### Deployment Details

| Component | Configuration |
|---|---|
| AWS Region | ap-south-1 (Mumbai) |
| Compute | Amazon EC2 t2.micro |
| Operating System | Amazon Linux 2023 |
| Container Registry | Amazon ECR |
| Image Version | churn-prediction-api:v1 |
| Application | FastAPI |
| Container Port | 8000 |
| IAM | ECR read-only instance role |

### Deployment Verification

The application started successfully and responded to HTTP health checks.

Observed container resource usage during testing:

- CPU: 0.14%
- Memory: 162.4 MiB / 700 MiB
- Memory utilization: 23.20%

These measurements represent a single low-traffic observation, not a load test.

### Cost Management

The EC2 instance was stopped after deployment testing to avoid continued instance compute usage. EBS and ECR resources may continue to incur storage charges.

### Future Improvements

- Automated build and deployment with GitHub Actions
- Model and application monitoring
- CI/CD deployment workflow
- Model retraining and deployment automation

### Project Status

**Completed:** Data pipeline, model training, evaluation, testing, containerization, ECR publishing, EC2 deployment, and API health verification.

**Planned:** CI/CD, monitoring, and automated retraining.