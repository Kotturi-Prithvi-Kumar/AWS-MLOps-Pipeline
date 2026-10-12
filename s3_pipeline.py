import boto3
import hashlib
from pathlib import Path

REGION = "ap-south-1"
PROFILE = "mlops-dev"

session = boto3.Session(
    profile_name=PROFILE,
    region_name=REGION)

sts = session.client("sts")
s3 = session.client("s3")

account_id = sts.get_caller_identity()["Account"]
bucket = f"mlops-day1-lab-{account_id}"

source = Path("data/customers.csv")
downloaded = Path("data/customers_downloaded.csv")
object_key = "datasets/customers.csv"


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


if not source.is_file():
    raise FileNotFoundError(
        "Run create_dataset.py first."
    )

# Upload the dataset to S3
print("Uploading dataset...")

s3.upload_file(
    str(source),
    bucket,
    object_key
)

print(f"Uploaded to s3://{bucket}/{object_key}")

# Download the dataset from S3
print("Downloading dataset...")

s3.download_file(
    bucket,
    object_key,
    str(downloaded)
)

# Compare file contents
original_hash = sha256(source)
downloaded_hash = sha256(downloaded)

print("Original SHA-256:", original_hash)
print("Downloaded SHA-256:", downloaded_hash)

assert original_hash == downloaded_hash, (
    "Dataset integrity verification failed!"
)

print("SUCCESS: Dataset integrity verified.")
