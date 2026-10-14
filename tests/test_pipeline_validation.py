from unittest.mock import patch

import pytest

from src.main import run_pipeline
from src.data_validator import DataValidationError


def test_invalid_data_blocks_s3_upload():
    invalid_report = {
        "valid": False,
        "total_rows": 100,
        "error_count": 1,
        "errors": ["Line 5: Negative tenure"],
    }

    with (
        patch("src.main.generate_dataset") as mock_generate,
        patch("src.main.validate_dataset") as mock_validate,
        patch("src.main.create_s3_client") as mock_client,
        patch("src.main.upload_to_s3") as mock_upload,
    ):
        mock_generate.return_value = "data/customers.csv"
        mock_validate.return_value = invalid_report

        with pytest.raises(
            DataValidationError,
            match="Dataset validation failed",
        ):
            run_pipeline(use_aws=True)

        mock_validate.assert_called_once_with(
            "data/customers.csv"
        )
        mock_client.assert_not_called()
        mock_upload.assert_not_called()


def test_valid_data_completes_local_pipeline():
    valid_report = {
        "valid": True,
        "total_rows": 100,
        "error_count": 0,
        "errors": [],
    }

    with (
        patch("src.main.generate_dataset") as mock_generate,
        patch("src.main.validate_dataset") as mock_validate,
        patch("src.main.create_s3_client") as mock_client,
    ):
        mock_generate.return_value = "data/customers.csv"
        mock_validate.return_value = valid_report

        run_pipeline(use_aws=False)

        mock_validate.assert_called_once()
        mock_client.assert_not_called()