import json
from pathlib import Path


def list_experiments(models_dir="models"):
    """Collect all versioned model experiment metadata."""

    models_path = Path(models_dir)
    experiments = []

    for metrics_path in models_path.glob(
        "churn_model_v*_metrics.json"
    ):
        with open(metrics_path, encoding="utf-8") as file:
            metadata = json.load(file)

        experiments.append({
            "version": metadata["model_version"],
            "algorithm": metadata["algorithm"],
            "created_at": metadata["created_at"],
            "accuracy": metadata["metrics"]["accuracy"],
            "precision": metadata["metrics"]["precision"],
            "recall": metadata["metrics"]["recall"],
            "f1_score": metadata["metrics"]["f1_score"],
        })

    return sorted(
        experiments,
        key=lambda item: int(item["version"][1:]),
    )


def get_best_experiment(
    experiments,
    metric="f1_score",
):
    """Return the experiment with the highest selected metric."""

    allowed_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    if metric not in allowed_metrics:
        raise ValueError(f"Unsupported metric: {metric}")

    if not experiments:
        raise ValueError("No experiments available")

    return max(
        experiments,
        key=lambda item: item[metric],
    )


def main():
    experiments = list_experiments()

    if not experiments:
        print("No experiments found.")
        return

    print("\nMODEL EXPERIMENT HISTORY")
    print("-" * 65)

    for experiment in experiments:
        print(
            f"{experiment['version']:5} "
            f"{experiment['algorithm']:20} "
            f"Accuracy={experiment['accuracy']:.4f} "
            f"F1={experiment['f1_score']:.4f}"
        )

    best = get_best_experiment(experiments)

    print("\nBest experiment by F1:")
    print(f"Version: {best['version']}")
    print(f"F1 Score: {best['f1_score']:.4f}")


if __name__ == "__main__":
    main()