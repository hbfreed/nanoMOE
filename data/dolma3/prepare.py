# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os

import numpy as np
import tiktoken
from datasets import Dataset, load_dataset  # huggingface datasets
from tqdm import tqdm


def process_data(domain, num_samples=):
    # stream the dataset and take only what we need
    streaming_ds = load_dataset(
        "allenai/dolma3_pool",
        data_files=f"data/common_crawl-{domain}*/*.jsonl.zst",
        split="train",
        streaming=True,
    )
    streaming_ds = streaming_ds.shuffle(seed=2357, buffer_size=10_000)
    streaming_ds = streaming_ds.select_columns(
        ["text"]
    )  # only need text, avoid schema issues
    streaming_ds = streaming_ds.take(num_samples)

    # convert streaming dataset to regular dataset
    print(f"Loading {num_samples} samples for {domain}...")
    dataset = Dataset.from_generator(lambda: (yield from streaming_ds))

    # tokenize with gpt2 bpe
    def process(example):
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    num_proc = 12
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc=f"tokenizing {domain}",
        num_proc=num_proc,
    )

    # write to bin file
    arr_len = np.sum(tokenized["len"], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f"{domain}_val.bin")
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        batch = tokenized.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')


if __name__ == "__main__":
    all_domains = [
        "crime_and_law",
        "science_math_and_technology",
        "finance_and_business",
        "health",
        "food_and_dining",
        "games",
        "literature",
        "religion",
        "politics",
    ]
    for domain in all_domains:
        process_data(domain)
