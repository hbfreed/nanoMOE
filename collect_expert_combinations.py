#!/usr/bin/env python3
"""
Collect expert routing statistics and save to CSV.
This script runs the analysis from variable_expert_analysis.ipynb up to CSV export.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from tqdm import tqdm
import tiktoken
import pandas as pd
from pathlib import Path
import trio

from model import GPTConfig, MoeMLPWithTracking, GPTWithTracking


def get_config(dataset, expert_config, seed):
    """Build configuration based on dataset, expert_config, and seed."""
    CONFIGS = {
        "gpt2": {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "vocab_size": 50304,
            "val_data_path": "data/openwebtext/val.bin",
            "expert_configs": {
                "5to1": {
                    "expert_sizes": [(4, 2560), (4, 512)],
                    "base_dir": "gpt2_experiments/multiseed_5to1",
                    "run_name_pattern": "ratio5_lbl0.01_compute0.004_seed{seed}",
                    "model_name": "GPT-2 120M 5:1 (4x2560 + 4x512)",
                    "available_seeds": [42, 1223, 1337]
                },
                "23to1": {
                    "expert_sizes": [(4, 2944), (4, 128)],
                    "base_dir": "gpt2_experiments/multiseed_23to1",
                    "run_name_pattern": "ratio23_lbl0.01_compute0.004_seed{seed}",
                    "model_name": "GPT-2 120M 23:1 (4x2944 + 4x128)",
                    "available_seeds": [42, 1223, 1337]
                }
            }
        },
        "gpt2_250m": {
            "n_layer": 16,
            "n_head": 16,
            "n_embd": 1024,
            "vocab_size": 50304,
            "val_data_path": "data/openwebtext/val.bin",
            "expert_configs": {
                "5to1": {
                    "expert_sizes": [(4, 3456), (4, 640)],
                    "base_dir": "gpt2_250m_experiments/expert_sizes_final_weights",
                    "run_name_pattern": "sizes_5to1_lbl0.01_compute0.004_seed{seed}",
                    "model_name": "GPT-2 250M 5:1 (4x3456 + 4x640)",
                    "available_seeds": [1337]
                },
                "31to1": {
                    "expert_sizes": [(4, 3968), (4, 128)],
                    "base_dir": "gpt2_250m_experiments/expert_sizes_final_weights",
                    "run_name_pattern": "sizes_31to1_lbl0.01_compute0.004_seed{seed}",
                    "model_name": "GPT-2 250M 31:1 (4x3968 + 4x128)",
                    "available_seeds": [1337]
                },
                "uniform": {
                    "expert_sizes": [(8, 2048)],
                    "base_dir": "gpt2_250m_experiments/expert_sizes_final_weights",
                    "run_name_pattern": "sizes_uniform_lbl0.01_compute0.004_seed{seed}",
                    "model_name": "GPT-2 250M Uniform (8x2048)",
                    "available_seeds": [1337]
                }
            }
        },
        "wikitext": {
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 640,
            "vocab_size": 8192,
            "expert_sizes": [(4, 2432), (4, 128)],
            "checkpoint_dir": f"out-wikitext/moe-8x2-variable-4x2432-4x128-seed{seed}",
            "val_data_path": "data/wikitext/val.bin",
            "model_name": f"WikiText (4x2432 + 4x128) seed{seed}"
        }
    }

    if dataset in ["gpt2", "gpt2_250m"]:
        base_cfg = CONFIGS[dataset]
        expert_cfg = base_cfg["expert_configs"][expert_config]

        # Check if seed is available
        if seed not in expert_cfg["available_seeds"]:
            raise ValueError(
                f"Seed {seed} not available for {dataset} {expert_config}. "
                f"Available seeds: {expert_cfg['available_seeds']}"
            )

        # Build run name with seed
        if "{seed}" in expert_cfg["run_name_pattern"]:
            run_name = expert_cfg["run_name_pattern"].replace("{seed}", str(seed))
        else:
            run_name = expert_cfg["run_name_pattern"]

        cfg = {
            "n_layer": base_cfg["n_layer"],
            "n_head": base_cfg["n_head"],
            "n_embd": base_cfg["n_embd"],
            "vocab_size": base_cfg["vocab_size"],
            "expert_sizes": expert_cfg["expert_sizes"],
            "checkpoint_dir": f"{expert_cfg['base_dir']}/{run_name}",
            "val_data_path": base_cfg["val_data_path"],
            "model_name": f"{expert_cfg['model_name']} seed{seed}"
        }
    else:
        # Flat config (e.g., wikitext)
        cfg = CONFIGS[dataset]

    return cfg


def load_model(cfg, device='cuda'):
    """Load model and checkpoint."""
    config = GPTConfig(
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_embd=cfg['n_embd'],
        bias=False,
        vocab_size=cfg['vocab_size'],
        use_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        block_size=128,
        block_k=64,
        expert_sizes=cfg["expert_sizes"]
    )

    checkpoint_path = f"{cfg['checkpoint_dir']}/ckpt.pt"

    model = GPTWithTracking(config).to(torch.bfloat16)

    for block in model.transformer.h:
        if hasattr(block.mlp, 'expert_sizes'):
            old_mlp = block.mlp
            block.mlp = MoeMLPWithTracking(config).to(torch.bfloat16)
            block.mlp.load_state_dict(old_mlp.state_dict())

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    return model, config


def collect_routing_statistics(model, config, val_data_path, device='cuda'):
    """Collect routing statistics in a single pass through validation data."""
    tokenizer = tiktoken.get_encoding('gpt2')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

    num_layers = config.n_layer
    expert_sizes = model.transformer.h[0].mlp.expert_sizes

    # Initialize tracking structures
    token_stats_per_layer = {}
    token_combinations_per_layer = {}

    for layer_idx in range(num_layers):
        layer_name = f'layer_{layer_idx}'
        token_stats_per_layer[layer_name] = defaultdict(lambda: {
            'expert_counts': np.zeros(config.num_experts, dtype=np.int64),
            'total_occurrences': 0,
            'total_entropy': 0.0,
            'total_layer_entropy': 0.0,
            'expert_size_sum': 0.0,
        })
        token_combinations_per_layer[layer_name] = defaultdict(Counter)

    batch_size = 1
    seq_len = 1024
    total_tokens = len(val_data)
    num_batches = total_tokens // seq_len

    print(f"Running single pass through {num_batches} batches to collect all statistics...")

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * seq_len
        end_idx = start_idx + seq_len
        batch_tokens = torch.from_numpy(val_data[start_idx:end_idx].astype(np.int64)).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits, loss, aux_loss = model(batch_tokens, targets=batch_tokens)

        output_probs = F.softmax(logits[0], dim=-1)
        epsilon = 1e-10
        output_entropy = -(output_probs * torch.log(output_probs + epsilon)).sum(dim=-1).float().cpu().numpy()

        for layer_idx in range(num_layers):
            layer_name = f'layer_{layer_idx}'
            layer_assignments = aux_loss['expert_assignments'][layer_name][0].cpu().numpy()
            layer_entropies = aux_loss['layer_entropies'][layer_name][0].cpu().numpy()
            token_stats = token_stats_per_layer[layer_name]

            for pos in range(seq_len):
                token_id = int(batch_tokens[0, pos].item())
                expert_ids = layer_assignments[pos]

                # Update individual expert statistics
                token_stats[token_id]['total_occurrences'] += 1
                token_stats[token_id]['total_entropy'] += output_entropy[pos]
                token_stats[token_id]['total_layer_entropy'] += layer_entropies[pos]

                for expert_id in expert_ids:
                    token_stats[token_id]['expert_counts'][expert_id] += 1
                    token_stats[token_id]['expert_size_sum'] += expert_sizes[expert_id]

                # Track expert combinations
                expert_combination = tuple(sorted(expert_ids))
                token_combinations_per_layer[layer_name][token_id][expert_combination] += 1

    print(f"✓ Collected all statistics in a single pass!")
    return token_stats_per_layer, token_combinations_per_layer, tokenizer, expert_sizes


def build_dataframe(token_stats_per_layer, token_combinations_per_layer, config, tokenizer, expert_sizes):
    """Build DataFrame from collected statistics."""
    print("Building dataframe from collected statistics...")

    num_layers = config.n_layer

    # Get unique tokens
    all_token_ids = set()
    for layer_combos in token_combinations_per_layer.values():
        all_token_ids.update(layer_combos.keys())

    # Build data for dataframe
    data = []
    for token_id in all_token_ids:
        # Decode token
        try:
            token_text = tokenizer.decode([token_id])
            token_text = token_text.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
            if '�' in token_text or not token_text.isprintable():
                token_text = f"<{token_id}>"
            if len(token_text) > 18:
                token_text = token_text[:17] + '…'
        except:
            token_text = f"<{token_id}>"

        # Calculate average expert SIZE and ENTROPY across all layers
        total_size = 0
        total_entropy = 0.0
        layer_count = 0
        layer_data = {}
        layer_entropy_data = {}
        token_count = 0

        for layer_idx in range(num_layers):
            layer_name = f'layer_{layer_idx}'
            combos = token_combinations_per_layer[layer_name][token_id]
            stats = token_stats_per_layer[layer_name][token_id]

            # Get token count from the first layer
            if layer_idx == 0 and stats['total_occurrences'] > 0:
                token_count = stats['total_occurrences']

            if combos:
                most_common = combos.most_common(1)[0][0]
                layer_size = sum(expert_sizes[e] for e in most_common)
                total_size += layer_size

                # Calculate average entropy for this token in this layer
                if stats['total_occurrences'] > 0:
                    layer_entropy = stats['total_entropy'] / stats['total_occurrences']
                    total_entropy += layer_entropy

                    # Store layer-wise intermediate entropy
                    layer_wise_entropy = stats['total_layer_entropy'] / stats['total_occurrences']
                    layer_entropy_data[f'layer_{layer_idx}_entropy'] = layer_wise_entropy

                layer_count += 1
                # Format with expert sizes: (5(128),7(128))
                formatted = "(" + ",".join([f"{e}({expert_sizes[e]})" for e in most_common]) + ")"
                layer_data[f'layer_{layer_idx}'] = formatted
            else:
                layer_data[f'layer_{layer_idx}'] = 'N/A'
                layer_entropy_data[f'layer_{layer_idx}_entropy'] = np.nan

        avg_size = total_size / layer_count if layer_count > 0 else 0
        avg_entropy = total_entropy / layer_count if layer_count > 0 else 0
        flops = 4 * config.n_embd * total_size

        row = {
            'token_id': token_id,
            'token': token_text,
            'count': token_count,
            'avg_size': avg_size,
            'avg_entropy': avg_entropy,
            'flops': flops,
            **layer_data,
            **layer_entropy_data
        }
        data.append(row)

    # Create DataFrame and sort by FLOPs
    df = pd.DataFrame(data)
    df = df.sort_values('flops', ascending=True).reset_index(drop=True)

    return df


def print_summary(df, config):
    """Print summary statistics."""
    total_dataset_flops = (df['flops'] * df['count']).sum()
    total_tokens = df['count'].sum()
    weighted_avg_flops = total_dataset_flops / total_tokens

    print(f"\nDataFrame created with {len(df)} tokens, sorted by FLOPs (low to high)")

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"Summary Statistics:")
    print(f"{'='*80}")
    print(f"Total unique tokens: {len(df)}")
    print(f"Total token occurrences: {total_tokens:,}")

    print(f"\nToken Count distribution:")
    print(f"  Min:    {df['count'].min():,}")
    print(f"  25%:    {df['count'].quantile(0.25):,.0f}")
    print(f"  Median: {df['count'].median():,.0f}")
    print(f"  75%:    {df['count'].quantile(0.75):,.0f}")
    print(f"  Max:    {df['count'].max():,}")
    print(f"  Mean:   {df['count'].mean():,.0f}")

    print(f"\nFLOPs distribution:")
    print(f"  Min:    {df['flops'].min():,.0f}")
    print(f"  25%:    {df['flops'].quantile(0.25):,.0f}")
    print(f"  Median: {df['flops'].median():,.0f}")
    print(f"  75%:    {df['flops'].quantile(0.75):,.0f}")
    print(f"  Max:    {df['flops'].max():,.0f}")
    print(f"  Mean:   {df['flops'].mean():,.0f}")

    print(f"\nAverage Size distribution:")
    print(f"  Tokens with avg_size >= {4*config.n_embd}: {(df['avg_size'] >= 4*config.n_embd).sum()} ({100*(df['avg_size'] >= config.n_embd).sum()/len(df):.2f}%)")
    print(f"  Tokens with avg_size < {4*config.n_embd}:  {(df['avg_size'] < 4*config.n_embd).sum()} ({100*(df['avg_size'] < 4*config.n_embd).sum()/len(df):.2f}%)")

    print(f"\nAverage Output Entropy distribution:")
    print(f"  Min:    {df['avg_entropy'].min():.4f}")
    print(f"  25%:    {df['avg_entropy'].quantile(0.25):.4f}")
    print(f"  Median: {df['avg_entropy'].median():.4f}")
    print(f"  75%:    {df['avg_entropy'].quantile(0.75):.4f}")
    print(f"  Max:    {df['avg_entropy'].max():.4f}")
    print(f"  Mean:   {df['avg_entropy'].mean():.4f}")

    baseline_flops = config.n_layer * 4 * config.n_embd * (config.n_embd*4)
    print(f"\n{'='*80}")
    print(f"FLOPs Analysis:")
    print(f"{'='*80}")
    print(f"Average FLOPs per unique token: {df['flops'].mean():.0f} ({100*df['flops'].mean()/baseline_flops:.2f}% of baseline)")
    print(f"Weighted average FLOPs per token occurrence: {weighted_avg_flops:.0f} ({100*weighted_avg_flops/baseline_flops:.2f}% of baseline)")
    print(f"Total dataset FLOPs: {total_dataset_flops:,.0f}")
    print(f"Baseline total FLOPs: {baseline_flops * total_tokens:,.0f}")
    print(f"FLOPs savings: {100 * (1 - total_dataset_flops / (baseline_flops * total_tokens)):.2f}%")


def run_single_seed(dataset, expert_config, seed, device, output_dir):
    """Run analysis for a single seed."""
    # Setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # Get configuration
    print(f"\n{'='*60}")
    print(f"Configuration: {dataset} {expert_config} seed{seed}")
    cfg = get_config(dataset, expert_config, seed)
    print(f"Expert sizes: {cfg['expert_sizes']}")
    print(f"Checkpoint: {cfg['checkpoint_dir']}/ckpt.pt")
    print(f"Val data: {cfg['val_data_path']}")
    print(f"{'='*60}\n")

    # Load model
    model, config = load_model(cfg, device=device)

    # Collect statistics
    token_stats_per_layer, token_combinations_per_layer, tokenizer, expert_sizes = collect_routing_statistics(
        model, config, cfg['val_data_path'], device=device
    )

    # Build DataFrame
    df = build_dataframe(token_stats_per_layer, token_combinations_per_layer, config, tokenizer, expert_sizes)

    # Print summary
    print_summary(df, config)

    # Save to CSV
    Path(output_dir).mkdir(exist_ok=True)
    sweep_value = cfg['checkpoint_dir'].split('/')[-1]
    output_path = f"{output_dir}/{sweep_value}_expert_combinations.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved to {output_path}")
    print(f"  Columns: {', '.join(df.columns.tolist()[:10])}... (and {len(df.columns)-10} more)")


async def run_seed_subprocess(dataset, expert_config, seed, device, output_dir):
    """Run a single seed in a subprocess using trio."""
    import sys

    cmd = [
        sys.executable,
        __file__,
        '--dataset', dataset,
        '--expert_config', expert_config,
        '--seed', str(seed),
        '--device', device,
        '--output_dir', output_dir,
    ]

    print(f"[Seed {seed}] Starting...")
    result = await trio.run_process(cmd, capture_stdout=True, capture_stderr=True)

    if result.returncode == 0:
        print(f"[Seed {seed}] ✓ Completed successfully")
    else:
        print(f"[Seed {seed}] ✗ Failed with return code {result.returncode}")
        print(f"[Seed {seed}] STDERR: {result.stderr.decode()}")

    return result.returncode


async def run_all_seeds(dataset, expert_config, device, output_dir):
    """Run all available seeds in parallel using trio."""
    # Get available seeds for this configuration
    cfg_test = get_config(dataset, expert_config, 42)  # dummy seed to get config structure

    # Determine available seeds based on dataset and config
    if dataset in ["gpt2", "gpt2_250m"]:
        base_cfg = {
            "gpt2": {
                "expert_configs": {
                    "5to1": {"available_seeds": [42, 1223, 1337]},
                    "23to1": {"available_seeds": [42, 1223, 1337]}
                }
            },
            "gpt2_250m": {
                "expert_configs": {
                    "5to1": {"available_seeds": [1337]},
                    "31to1": {"available_seeds": [1337]},
                    "uniform": {"available_seeds": [1337]}
                }
            }
        }
        available_seeds = base_cfg[dataset]["expert_configs"][expert_config]["available_seeds"]
    else:
        available_seeds = [42, 1223, 1337]  # default for wikitext

    print(f"\n{'='*80}")
    print(f"Running all seeds for {dataset} {expert_config}")
    print(f"Available seeds: {available_seeds}")
    print(f"{'='*80}\n")

    async with trio.open_nursery() as nursery:
        for seed in available_seeds:
            nursery.start_soon(run_seed_subprocess, dataset, expert_config, seed, device, output_dir)

    print(f"\n{'='*80}")
    print(f"All seeds completed!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Collect expert routing statistics')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['gpt2', 'gpt2_250m', 'wikitext'],
                        help='Dataset/model size to use')
    parser.add_argument('--expert_config', type=str, required=True,
                        help='Expert configuration (e.g., 5to1, 23to1, 31to1, uniform)')
    parser.add_argument('--seed', type=int,
                        help='Random seed (required if not using --all-seeds)')
    parser.add_argument('--all-seeds', action='store_true',
                        help='Run all available seeds in parallel')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--output_dir', type=str, default='analysis_csvs',
                        help='Output directory for CSV files')

    args = parser.parse_args()

    if args.all_seeds:
        # Run all seeds in parallel using trio
        trio.run(run_all_seeds, args.dataset, args.expert_config, args.device, args.output_dir)
    else:
        if args.seed is None:
            parser.error("--seed is required when not using --all-seeds")
        # Run single seed
        run_single_seed(args.dataset, args.expert_config, args.seed, args.device, args.output_dir)


if __name__ == "__main__":
    main()
