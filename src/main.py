
import argparse
import logging
import os
from pathlib import Path

from src.create_dataset import generate_dataset
from src.data_validator import (
    DataValidationError,
    validate_dataset,
)
from src.s3_pipeline import (
    create_s3_client,
    upload_to_s3,
    download_from_s3,
    verify_integrity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_pipeline(use_aws=False):
    """Run the customer churn data pipeline."""

    logger.info("Starting MLOps pipeline")

    source = generate_dataset(
        output_path="data/customers.csv",
        num_records=100,
        seed=69,
    )

    logger.info("Starting dataset validation")

    report = validate_dataset(source)

    if not report["valid"]:
        for error in report["errors"]:
            logger.error("Data validation error: %s", error)

        raise DataValidationError(
            f"Dataset validation failed: "
            f"{report['error_count']} error(s) detected"
        )

    logger.info(
        "Dataset validation passed: %s records checked",
        report["total_rows"],
    )

    if not use_aws:
        logger.info("Local mode: skipping AWS operations")
        logger.info("Pipeline completed successfully")
        return

    profile = os.getenv("AWS_PROFILE", "mlops-dev")
    region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
    bucket = os.getenv("S3_BUCKET")

    if not bucket:
        raise ValueError(
            "S3_BUCKET environment variable is required for AWS mode"
        )

    s3 = create_s3_client(profile=profile, region=region)

    object_key = "datasets/customers.csv"
    destination = Path("data/customers_downloaded.csv")

    upload_to_s3(s3, source, bucket, object_key)

    download_from_s3(
        s3,
        bucket,
        object_key,
        destination,
    )

    verify_integrity(source, destination)

    logger.info("AWS pipeline completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AWS MLOps customer churn pipeline"
    )

    parser.add_argument(
        "--aws",
        action="store_true",
        help="Enable AWS S3 upload/download",
    )

    args = parser.parse_args()
    run_pipeline(use_aws=args.aws)
