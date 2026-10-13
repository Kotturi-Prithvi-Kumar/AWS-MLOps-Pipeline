
import csv

import pytest

from src.create_dataset import generate_dataset
from src.s3_pipeline import sha256, verify_integrity


def test_dataset_row_count(tmp_path):
    output = tmp_path / "customers.csv"

    generate_dataset(output_path=output, num_records=100)

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 100


def test_dataset_columns(tmp_path):
    output = tmp_path / "customers.csv"

    generate_dataset(output_path=output)

    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

    assert columns == [
        "customer_id",
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "churn",
    ]


def test_reproducibility(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    generate_dataset(output_path=first, seed=69)
    generate_dataset(output_path=second, seed=69)

    assert sha256(first) == sha256(second)


def test_corrupted_file_detection(tmp_path):
    original = tmp_path / "original.csv"
    corrupted = tmp_path / "corrupted.csv"

    original.write_text("customer_id,churn\n1,0\n")
    corrupted.write_text("customer_id,churn\n1,1\n")

    with pytest.raises(ValueError):
        verify_integrity(original, corrupted)


def test_invalid_record_count(tmp_path):
    output = tmp_path / "invalid.csv"

    with pytest.raises(ValueError):
        generate_dataset(output_path=output, num_records=0)
