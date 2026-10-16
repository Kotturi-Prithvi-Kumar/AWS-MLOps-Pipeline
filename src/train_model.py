from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_validator import validate_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import json
import joblib
from datetime import datetime, timezone


FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
]

TARGET_COLUMN = "churn"

RANDOM_STATE = 69


def load_training_data(dataset_path: str | Path):
    """Load and validate the customer churn dataset."""

    dataset_path = Path(dataset_path)

    validate_dataset(dataset_path)

    dataframe = pd.read_csv(dataset_path)

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    return X, y


def split_training_data(X, y):
    """Create reproducible training and testing datasets."""

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_model(X_train, y_train):
    """Train a baseline churn classification model."""

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using classification metrics."""

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    return metrics

def save_model(model, metrics, output_dir="models"):
    """Save a trained model and its evaluation metadata."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "churn_model_v1.joblib"
    metrics_path = output_path / "churn_model_v1_metrics.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": "customer_churn_classifier",
        "model_version": "v1",
        "algorithm": type(model).__name__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "metrics": metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    return model_path, metrics_path

def load_model(model_path="models/churn_model_v1.joblib"):
    """Load a previously trained model artifact."""

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return joblib.load(model_path)