"""
Based on https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py
For our tiny little models, Hellaswag seems like it should be fine for evaluating between them.
Probably won't perform as well as any of Karpathy's because we use chinchilla 2.5B tokens (125*20) vs his 600B tokens
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, GPTConfig

def get_config_from_checkpoint_dir(checkpoint_dir):
    """Infer model config from checkpoint directory name."""
    checkpoint_dir_lower = checkpoint_dir.lower()

    # Base config
    base_config = {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "bias": False,
        "vocab_size": 50304,
    }

    print(f"checkpoint_dir_lower: {checkpoint_dir_lower}")
    if "moe" not in checkpoint_dir_lower:
        # Dense model
        base_config.update({
            "use_moe": False,
        })
    else:
        # MoE model
        moe_config = {
            "use_moe": True,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "block_size": 128,
            "block_k": 64,
        }

        if "variable" in checkpoint_dir_lower:
            # Variable-size experts
            moe_config["expert_sizes"] = [(4, 2944), (4, 128)]
        else:
            # Same-sized experts
            moe_config["expert_sizes"] = [(8, 1536)]

        base_config.update(moe_config)

    return GPTConfig(**base_config)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


@torch.no_grad()
def evaluate(checkpoint_dir, device, split="val"):
    torch.set_float32_matmul_precision("high")

    # Load config from checkpoint if available, otherwise infer
    checkpoint = torch.load(f"{checkpoint_dir}/ckpt.pt", map_location="cpu")
    if 'model_args' in checkpoint:
        print("Loading config from checkpoint...")
        config = GPTConfig(**checkpoint['model_args'])
    else:
        print("Inferring config from checkpoint directory name...")
        config = get_config_from_checkpoint_dir(checkpoint_dir)

    print(f"Using config: use_moe={config.use_moe}", end="")
    if config.use_moe:
        print(f", expert_sizes={config.expert_sizes}")
    else:
        print(" (dense model)")

    model = GPT(config)
    state_dict = checkpoint["model"]
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    # Note: torch.compile disabled for hellaswag eval to avoid issues with custom CUDA ops
    # model = torch.compile(model)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    # Get total count for progress bar
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        total_examples = sum(1 for _ in f)

    pbar = tqdm(iterate_examples(split), desc="HellaSwag", total=total_examples)
    for example in pbar:
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        # Create dummy targets to get full logits (not just last position)
        dummy_targets = tokens.clone()
        logits = model(tokens, targets=dummy_targets)[0]
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        # Update progress bar with running accuracy
        pbar.set_postfix({'acc_norm': f'{num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}'})

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

if __name__ == "__main__":
    # Example usage:
    # python hellaswag.py -c out-openwebtext/moe-8x2-variable-4x2944-4x128-seed1337 -d cuda:0
    # python hellaswag.py -c out-openwebtext/moe-8x2-8x1536_lbl_0.2 -d cuda:1
    # python hellaswag.py -c out-openwebtext/dense-baseline -d cuda:2
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint_dir", type=str, required=True, help="path to checkpoint directory")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.checkpoint_dir, args.device)