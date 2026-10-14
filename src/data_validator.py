import csv
import math
from pathlib import Path


REQUIRED_COLUMNS = [
    "customer_id",
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "churn",
]


class DataValidationError(ValueError):
    """Raised when a dataset fails validation."""


def validate_dataset(file_path: str | Path) -> dict:
    """Validate customer data before further processing."""

    file_path = Path(file_path)

    if not file_path.is_file():
        raise DataValidationError(
            f"Dataset not found: {file_path}"
        )

    errors = []
    customer_ids = set()
    row_count = 0

    try:
        with file_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != REQUIRED_COLUMNS:
                raise DataValidationError(
                    f"Invalid schema. Expected: {REQUIRED_COLUMNS}. "
                    f"Found: {reader.fieldnames}"
                )

            for line_number, row in enumerate(reader, start=2):
                row_count += 1

                if None in row or any(
                    value is None or not value.strip()
                    for value in row.values()
                ):
                    errors.append(
                        f"Line {line_number}: missing or extra values"
                    )
                    continue

                try:
                    customer_id = int(row["customer_id"])
                    tenure = int(row["tenure_months"])
                    spend = float(row["monthly_spend"])
                    tickets = int(row["support_tickets"])
                    churn = int(row["churn"])

                    if customer_id <= 0:
                        raise ValueError("Invalid customer_id")

                    if customer_id in customer_ids:
                        raise ValueError("Duplicate customer_id")

                    customer_ids.add(customer_id)

                    if tenure < 0:
                        raise ValueError("Negative tenure")

                    if not math.isfinite(spend) or spend < 0:
                        raise ValueError("Invalid monthly_spend")

                    if tickets < 0:
                        raise ValueError("Negative support_tickets")

                    if churn not in (0, 1):
                        raise ValueError("Churn must be 0 or 1")

                except ValueError as exc:
                    errors.append(
                        f"Line {line_number}: {exc}"
                    )

    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataValidationError(
            f"Unable to read dataset: {exc}"
        ) from exc

    if row_count == 0:
        errors.append("Dataset is empty")

    report = {
        "file": str(file_path),
        "total_rows": row_count,
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
    }

    return report