import logging
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.s3_pipeline import (
    upload_to_s3,
    download_from_s3,
    verify_integrity,
)


def make_s3_error():
    return ClientError(
        {
            "Error": {
                "Code": "AccessDenied",
                "Message": "Access Denied",
            }
        },
        "PutObject",
    )


def test_upload_failure(tmp_path, caplog):
    source = tmp_path / "customers.csv"
    source.write_text("customer_id\n1\n")

    s3 = MagicMock()
    s3.upload_file.side_effect = make_s3_error()

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ClientError):
            upload_to_s3(
                s3,
                source,
                "test-bucket",
                "datasets/customers.csv",
            )

    s3.upload_file.assert_called_once()
    assert "S3 upload failed" in caplog.text


def test_download_failure(tmp_path, caplog):
    s3 = MagicMock()
    s3.download_file.side_effect = make_s3_error()

    destination = tmp_path / "downloaded.csv"

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ClientError):
            download_from_s3(
                s3,
                "test-bucket",
                "datasets/customers.csv",
                destination,
            )

    s3.download_file.assert_called_once()
    assert "S3 download failed" in caplog.text


def test_corrupted_download(tmp_path):
    original = tmp_path / "original.csv"
    downloaded = tmp_path / "downloaded.csv"

    original.write_text("customer_id\n1\n")
    downloaded.write_text("customer_id\n999\n")

    with pytest.raises(
        ValueError,
        match="Dataset integrity verification failed",
    ):
        verify_integrity(original, downloaded)