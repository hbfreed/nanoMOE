# saves the stack-edu dataset to a binary file for training
# stack-edu requires downloading content from Software Heritage S3

import gzip
import os

import boto3
import numpy as np
import tiktoken
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import Dataset, load_dataset
from tqdm import tqdm

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
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return {"text": "", "download_success": False}
        else:
            raise


def process_language(language, num_samples=5000, num_proc=24):
    """Process a single programming language from stack-edu."""
    print(f"\n{'=' * 50}")
    print(f"Processing {language}...")
    print(f"{'=' * 50}")

    # Load dataset in streaming mode
    streaming_ds = load_dataset(
        "HuggingFaceTB/stack-edu",
        language,
        split="train",
        streaming=True,
    )
    streaming_ds = streaming_ds.shuffle(seed=2357, buffer_size=10_000)
    streaming_ds = streaming_ds.take(num_samples)

    # Convert to regular dataset
    print(f"Loading {num_samples} samples for {language}...")
    dataset = Dataset.from_generator(lambda: (yield from streaming_ds))

    # Download content from S3
    print(f"Downloading content from Software Heritage S3...")
    dataset = dataset.map(
        download_contents,
        input_columns="blob_id",
        num_proc=num_proc,
        desc=f"downloading {language}",
    )

    # Filter failed downloads
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: x["download_success"])
    print(f"Downloaded {len(dataset)}/{original_len} files successfully")

    if len(dataset) == 0:
        print(f"No data for {language}, skipping...")
        return 0

    # Tokenize with gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = dataset.map(
        process,
        remove_columns=dataset.column_names,
        desc=f"tokenizing {language}",
        num_proc=num_proc,
    )

    # Calculate total tokens
    arr_len = np.sum(tokenized["len"], dtype=np.uint64)
    print(f"Total tokens for {language}: {arr_len:,}")

    # Write to bin file
    filename = os.path.join(os.path.dirname(__file__), f"{language.lower()}_train.bin")
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    total_batches = min(1024, len(tokenized))

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        batch = tokenized.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

    print(f"Saved {filename}")
    return arr_len


if __name__ == "__main__":
    # Target ~4M tokens total
    # Python is the most educational, so we'll focus on it
    # Adjust num_samples to hit token target (code files avg ~100-300 tokens)

    languages_config = {
        "Python": 15000,  # ~2M tokens (largest share)
        "JavaScript": 5000,  # ~500k tokens
        "Java": 5000,  # ~500k tokens
        "Cpp": 5000,  # ~500k tokens
        "C": 5000,  # ~500k tokens
    }

    total_tokens = 0
    for language, num_samples in languages_config.items():
        tokens = process_language(language, num_samples=num_samples, num_proc=24)
        total_tokens += tokens

    print(f"\n{'=' * 50}")
    print(f"Total tokens across all languages: {total_tokens:,}")
    print(f"{'=' * 50}")

    # Combine all language files and trim to exactly 4,000,768 tokens
    TARGET_TOKENS = 4_000_768
    print(f"\nCombining and trimming to {TARGET_TOKENS:,} tokens...")

    all_tokens = []
    for language in languages_config.keys():
        bin_file = os.path.join(
            os.path.dirname(__file__), f"{language.lower()}_train.bin"
        )
        if os.path.exists(bin_file):
            data = np.memmap(bin_file, dtype=np.uint16, mode="r")
            all_tokens.append(np.array(data))
            print(f"  Loaded {language}: {len(data):,} tokens")

    combined = np.concatenate(all_tokens)
    print(f"Combined total: {len(combined):,} tokens")

    # Trim to target
    if len(combined) >= TARGET_TOKENS:
        combined = combined[:TARGET_TOKENS]
    else:
        print(
            f"Warning: only {len(combined):,} tokens available, less than target {TARGET_TOKENS:,}"
        )

    # Write final combined file
    final_file = os.path.join(os.path.dirname(__file__), "val.bin")
    final_arr = np.memmap(
        final_file, dtype=np.uint16, mode="w+", shape=(len(combined),)
    )
    final_arr[:] = combined
    final_arr.flush()

    print(f"\nSaved {final_file} with {len(combined):,} tokens")
