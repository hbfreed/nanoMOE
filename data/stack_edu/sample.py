import gzip

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import load_dataset

# S3 setup for Software Heritage (no auth required)
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
BUCKET_NAME = "softwareheritage"


def download_contents(blob_id):
    """Download code content from Software Heritage S3 bucket."""
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        with gzip.GzipFile(fileobj=obj["Body"]) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return content
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


# Load streaming dataset
ds = load_dataset("HuggingFaceTB/stack-edu", "Python", split="train", streaming=True)

# Loop until we find a meaty file
MIN_LINES = 50  # Adjust to taste
MIN_CHARS = 1000

for i, sample in enumerate(ds):
    content = download_contents(sample["blob_id"])

    if content is None:
        print(f"[{i}] Download failed, skipping...")
        continue

    lines = content.count('\n')
    chars = len(content)

    if lines >= MIN_LINES and chars >= MIN_CHARS:
        print(f"[{i}] Found meaty file: {lines} lines, {chars} chars")
        print(f"\nMetadata: {sample}\n")
        print("=" * 60)
        print(content)
        break
    else:
        print(f"[{i}] Too small: {lines} lines, {chars} chars - skipping")
