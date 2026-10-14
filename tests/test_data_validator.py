import csv

import pytest

from src.data_validator import (
    DataValidationError,
    REQUIRED_COLUMNS,
    validate_dataset,
)


def create_test_csv(tmp_path, rows):
    file_path = tmp_path / "customers.csv"

    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=REQUIRED_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)

    return file_path


@pytest.fixture
def valid_customer():
    return {
        "customer_id": 1,
        "tenure_months": 12,
        "monthly_spend": 99.50,
        "support_tickets": 2,
        "churn": 0,
    }


def test_valid_dataset(tmp_path, valid_customer):
    file_path = create_test_csv(tmp_path, [valid_customer])

    report = validate_dataset(file_path)

    assert report["valid"] is True
    assert report["total_rows"] == 1
    assert report["error_count"] == 0


def test_negative_monthly_spend(tmp_path, valid_customer):
    valid_customer["monthly_spend"] = -100

    file_path = create_test_csv(tmp_path, [valid_customer])
    report = validate_dataset(file_path)

    assert report["valid"] is False
    assert report["error_count"] == 1


def test_invalid_churn(tmp_path, valid_customer):
    valid_customer["churn"] = 3

    file_path = create_test_csv(tmp_path, [valid_customer])
    report = validate_dataset(file_path)

    assert report["valid"] is False
    assert "Churn must be 0 or 1" in report["errors"][0]


def test_duplicate_customer_id(tmp_path, valid_customer):
    rows = [valid_customer.copy(), valid_customer.copy()]

    file_path = create_test_csv(tmp_path, rows)
    report = validate_dataset(file_path)

    assert report["valid"] is False
    assert "Duplicate customer_id" in report["errors"][0]


def test_missing_value(tmp_path, valid_customer):
    valid_customer["tenure_months"] = ""

    file_path = create_test_csv(tmp_path, [valid_customer])
    report = validate_dataset(file_path)

    assert report["valid"] is False
    assert report["error_count"] == 1


def test_empty_dataset(tmp_path):
    file_path = create_test_csv(tmp_path, [])

    report = validate_dataset(file_path)

    assert report["valid"] is False
    assert "Dataset is empty" in report["errors"]


def test_missing_file(tmp_path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(DataValidationError):
        validate_dataset(file_path)


def test_invalid_schema(tmp_path):
    file_path = tmp_path / "wrong_schema.csv"
    file_path.write_text(
        "customer_id,monthly_spend\n1,100\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError):
        validate_dataset(file_path)