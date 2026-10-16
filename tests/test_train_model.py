import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.train_model import (
    FEATURE_COLUMNS,
    load_training_data,
    split_training_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


@pytest.fixture
def training_data():
    return load_training_data("data/customers.csv")


@pytest.fixture
def split_data(training_data):
    X, y = training_data
    return split_training_data(X, y)


def test_training_features(training_data):
    X, y = training_data

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
    assert "customer_id" not in X.columns
    assert "churn" not in X.columns


def test_train_test_split(split_data):
    X_train, X_test, y_train, y_test = split_data

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_reproducibility(training_data):
    X, y = training_data

    first = split_training_data(X, y)
    second = split_training_data(X, y)

    for a, b in zip(first, second):
        assert a.equals(b)


def test_model_training(split_data):
    X_train, _, y_train, _ = split_data

    model = train_model(X_train, y_train)

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert hasattr(model, "classes_")


def test_model_evaluation(split_data):
    X_train, X_test, y_train, y_test = split_data

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    for metric in ("accuracy", "precision", "recall", "f1_score"):
        assert 0 <= metrics[metric] <= 1

    matrix = metrics["confusion_matrix"]

    assert len(matrix) == 2
    assert all(len(row) == 2 for row in matrix)
    assert sum(map(sum, matrix)) == len(y_test)


def test_model_save_and_load(split_data, tmp_path):
    X_train, X_test, y_train, y_test = split_data

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    model_path, metrics_path = save_model(
        model, metrics, output_dir=tmp_path
    )

    assert model_path.is_file()
    assert metrics_path.is_file()

    with metrics_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["model_version"] == "v1"
    assert metadata["features"] == FEATURE_COLUMNS
    assert metadata["metrics"] == metrics

    loaded_model = load_model(model_path)

    assert np.array_equal(
        model.predict(X_test),
        loaded_model.predict(X_test),
    )


def test_missing_model(tmp_path):
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError):
        load_model(missing_path)