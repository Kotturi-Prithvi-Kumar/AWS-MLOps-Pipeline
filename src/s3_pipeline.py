
import hashlib
import logging
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


def create_s3_client(profile="mlops-dev", region="ap-south-1"):
    """Create an authenticated AWS S3 client."""
    session = boto3.Session(
        profile_name=profile,
        region_name=region,
    )
    return session.client("s3")


def upload_to_s3(s3, source_path, bucket, object_key):
    """Upload a local file to S3."""
    source = Path(source_path)

    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    try:
        s3.upload_file(str(source), bucket, object_key)
        logger.info("Uploaded %s to s3://%s/%s", source, bucket, object_key)
    except (BotoCoreError, ClientError, OSError):
        logger.exception("S3 upload failed")
        raise


def download_from_s3(s3, bucket, object_key, destination_path):
    """Download an S3 object to a local file."""
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        s3.download_file(bucket, object_key, str(destination))
        logger.info("Downloaded s3://%s/%s", bucket, object_key)
    except (BotoCoreError, ClientError, OSError):
        logger.exception("S3 download failed")
        raise

    return destination


def sha256(path):
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_integrity(original_path, downloaded_path):
    """Verify that two files have identical SHA-256 hashes."""
    original_hash = sha256(original_path)
    downloaded_hash = sha256(downloaded_path)

    if original_hash != downloaded_hash:
        raise ValueError("Dataset integrity verification failed")

    logger.info("SUCCESS: Dataset integrity verified")
    return True
