import json

import pytest

from src.experiment_tracker import (
    get_best_experiment,
    list_experiments,
)


def make_experiment(directory, version, f1):
    metadata = {
        "model_version": version,
        "algorithm": "LogisticRegression",
        "created_at": "2026-10-17T00:00:00+00:00",
        "metrics": {
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1_score": f1,
        },
    }

    path = directory / f"churn_model_{version}_metrics.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_list_experiments(tmp_path):
    make_experiment(tmp_path, "v2", 0.8)
    make_experiment(tmp_path, "v1", 0.6)

    experiments = list_experiments(tmp_path)

    assert len(experiments) == 2
    assert [e["version"] for e in experiments] == ["v1", "v2"]


def test_best_experiment(tmp_path):
    make_experiment(tmp_path, "v1", 0.6)
    make_experiment(tmp_path, "v2", 0.8)

    experiments = list_experiments(tmp_path)
    best = get_best_experiment(experiments)

    assert best["version"] == "v2"


def test_empty_experiments(tmp_path):
    assert list_experiments(tmp_path) == []


def test_best_experiment_empty():
    with pytest.raises(ValueError):
        get_best_experiment([])


def test_invalid_metric():
    experiments = [{"f1_score": 0.8}]

    with pytest.raises(ValueError):
        get_best_experiment(experiments, metric="invalid")