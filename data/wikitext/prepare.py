# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, pre_tokenizers, models, decoders, processors
import pickle
import random
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 12

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc


dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", num_proc=num_proc_load_dataset)

# Rename validation to val for shorter filenames
if "validation" in dataset:
    dataset["val"] = dataset["validation"]
    del dataset["validation"]

tokenizer_path = os.path.join(os.path.dirname(__file__), 'wikitext_tokenizer_8k.json')

if os.path.exists(tokenizer_path):
    print(f"Loading existing tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    print(f"No tokenizer found, training new one...")

    # Define special tokens
    special_tokens = [
        "<|endoftext|>",
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|unk|>"
    ]

    data = dataset['train']['text'] #get the train data as a list

    # Create BPE model with byte_fallback enabled
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>", byte_fallback=True))

    # Set pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Set post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Train the tokenizer
    tokenizer.train_from_iterator(data, trainer=BpeTrainer(
        vocab_size=8192,
        special_tokens=special_tokens,
        show_progress=True
    ))

    tokenizer.save(tokenizer_path)

# Add the EOT token if it doesn't exist (for backward compatibility)
eot_token = "<|endoftext|>"
if eot_token not in tokenizer.get_vocab():
    tokenizer.add_tokens([eot_token])
    tokenizer.save(tokenizer_path)

eot_token_id = tokenizer.token_to_id(eot_token)
print(f"EOT token ID: {eot_token_id}")

# Get IDs for all special tokens
special_token_ids = {}
for token in ["<|endoftext|>", "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]:
    token_id = tokenizer.token_to_id(token)
    if token_id is not None:
        special_token_ids[token] = token_id
        print(f"{token} ID: {token_id}")

enc = tokenizer.encode
   


if __name__ == '__main__':
    
    # we now want to tokenize the dataset. first define the encoding function (8192 bpe)
    def process(example):
        ids = enc(example['text']).ids # encode_ordinary ignores any special tokens
        ids.append(eot_token_id) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': 8192,
        'eot_token_id': eot_token_id,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)