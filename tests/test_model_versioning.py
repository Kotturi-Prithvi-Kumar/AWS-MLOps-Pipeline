import hashlib
import json

from sklearn.linear_model import LogisticRegression

from src.train_model import (
    calculate_file_hash,
    get_next_model_version,
    save_model,
)


def test_first_model_version(tmp_path):
    assert get_next_model_version(tmp_path) == 1


def test_next_model_version(tmp_path):
    (tmp_path / "churn_model_v1.joblib").write_bytes(b"model1")
    (tmp_path / "churn_model_v3.joblib").write_bytes(b"model3")

    assert get_next_model_version(tmp_path) == 4


def test_ignore_invalid_model_names(tmp_path):
    (tmp_path / "churn_model_vabc.joblib").write_bytes(b"invalid")
    (tmp_path / "other_model_v10.joblib").write_bytes(b"other")

    assert get_next_model_version(tmp_path) == 1


def test_file_hash(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_bytes(b"hello mlops")

    expected = hashlib.sha256(b"hello mlops").hexdigest()

    assert calculate_file_hash(file_path) == expected


def test_save_model_metadata(tmp_path):
    model = LogisticRegression(max_iter=1000, random_state=69)
    metrics = {"accuracy": 0.95, "f1_score": 0.67}

    model_path, metrics_path = save_model(
        model,
        metrics,
        output_dir=tmp_path,
    )

    assert model_path.name == "churn_model_v1.joblib"
    assert model_path.exists()
    assert metrics_path.exists()

    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metadata["model_version"] == "v1"
    assert metadata["algorithm"] == "LogisticRegression"
    assert metadata["metrics"] == metrics
    assert metadata["hyperparameters"]["max_iter"] == 1000
    assert metadata["model_sha256"] == calculate_file_hash(model_path)


def test_save_model_creates_new_version(tmp_path):
    model = LogisticRegression()
    metrics = {"accuracy": 0.9}

    first_model, _ = save_model(model, metrics, tmp_path)
    first_hash = calculate_file_hash(first_model)

    second_model, _ = save_model(model, metrics, tmp_path)

    assert first_model.name == "churn_model_v1.joblib"
    assert second_model.name == "churn_model_v2.joblib"
    assert first_model.exists()
    assert second_model.exists()
    assert calculate_file_hash(first_model) == first_hash