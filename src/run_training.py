from pathlib import Path

from src.train_model import (
    load_training_data,
    split_training_data,
    train_model,
    evaluate_model,
    save_model,
)


def main():
    dataset_path = Path("data/customers.csv")

    X, y = load_training_data(dataset_path)

    X_train, X_test, y_train, y_test = split_training_data(X, y)

    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    model_path, metrics_path = save_model(model, metrics)

    print("\nTraining completed successfully!")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")

    for metric in ("accuracy", "precision", "recall", "f1_score"):
        print(f"{metric}: {metrics[metric]:.4f}")


if __name__ == "__main__":
    main()
