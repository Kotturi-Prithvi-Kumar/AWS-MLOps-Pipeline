
import json
from pathlib import Path

import joblib

from src.train_model import calculate_file_hash


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_VERSION = "v2"

MODEL_PATH = (
    PROJECT_ROOT / "models" / f"churn_model_{MODEL_VERSION}.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT / "models"
    / f"churn_model_{MODEL_VERSION}_metrics.json"
)


class ModelService:
    """Load and serve a verified ML model."""

    def __init__(self):
        self.model = None
        self.metadata = None

    def load(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        if not METADATA_PATH.is_file():
            raise FileNotFoundError(
                f"Metadata not found: {METADATA_PATH}"
            )

        with open(METADATA_PATH, encoding="utf-8") as file:
            metadata = json.load(file)

        actual_hash = calculate_file_hash(MODEL_PATH)
        expected_hash = metadata["model_sha256"]

        if actual_hash != expected_hash:
            raise ValueError(
                "Model integrity verification failed"
            )

        if metadata["model_version"] != MODEL_VERSION:
            raise ValueError("Model version mismatch")

        model = joblib.load(MODEL_PATH)

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "Model does not support probability prediction"
            )

        expected_features = metadata["features"]

        if list(model.feature_names_in_) != expected_features:
            raise ValueError("Model feature schema mismatch")

        self.model = model
        self.metadata = metadata

        return self

    def predict(self, customer_data):
        if self.model is None:
            raise RuntimeError("Model has not been loaded")

        import pandas as pd

        features = self.metadata["features"]

        dataframe = pd.DataFrame(
            [customer_data],
            columns=features,
        )

        prediction = int(self.model.predict(dataframe)[0])

        probability = float(
            self.model.predict_proba(dataframe)[0][1]
        )

        return {
            "prediction": prediction,
            "churn_probability": round(probability, 4),
            "model_version": self.metadata["model_version"],
        }
