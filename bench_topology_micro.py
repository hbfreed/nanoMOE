"""
Micro-benchmark for topology_var.py - isolates the topology computation.
Tests the specific overhead of Python loop vs tensor ops.
"""
import time
import torch

NUM_RUNS = 1000
WARMUP_RUNS = 100


def benchmark_reference(padded_bins, expert_block_counts, block_size):
    """Original Python loop implementation."""
    total_nnz = 0
    output_offsets = [0]

    for expert_id in range(expert_block_counts.numel()):
        if expert_id == 0:
            expert_token_blocks = padded_bins[0].item() // block_size
        else:
            expert_token_blocks = (
                padded_bins[expert_id].item() - padded_bins[expert_id - 1].item()
            ) // block_size
        expert_output_size = (
            expert_token_blocks * expert_block_counts[expert_id].item()
        )
        total_nnz += expert_output_size
        output_offsets.append(output_offsets[-1] + expert_output_size)

    output_offsets_tensor = torch.tensor(
        output_offsets[:-1], dtype=torch.int32, device=padded_bins.device
    )
    return total_nnz, output_offsets_tensor


def benchmark_optimized(padded_bins, expert_block_counts, block_size):
    """Optimized tensor-based implementation."""
    prepend = padded_bins.new_zeros(1)
    bin_sizes = torch.diff(padded_bins, prepend=prepend)
    expert_token_blocks = (bin_sizes // block_size).to(torch.int32)

    expert_output_sizes = expert_token_blocks * expert_block_counts.to(torch.int32)
    total_nnz = expert_output_sizes.sum().item()

    cumsum = expert_output_sizes.cumsum(0)
    output_offsets_tensor = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=padded_bins.device),
        cumsum[:-1]
    ]).to(torch.int32)

    return total_nnz, output_offsets_tensor


def run_benchmark(func, padded_bins, expert_block_counts, block_size, name):
    """Run benchmark for a function."""
    # Warmup
    for _ in range(WARMUP_RUNS):
        func(padded_bins, expert_block_counts, block_size)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(NUM_RUNS):
        func(padded_bins, expert_block_counts, block_size)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    per_call_us = elapsed_ms / NUM_RUNS * 1000
    return per_call_us


def main():
    device = 'cuda'
    block_size = 128

    print("=" * 70)
    print("Micro-benchmark: topology_var Python loop vs tensor ops")
    print(f"Config: {NUM_RUNS} runs, {WARMUP_RUNS} warmup, block_size={block_size}")
    print("=" * 70)
    print()

    # Test with different numbers of experts
    expert_configs = [
        (8, [16, 32, 16, 32, 16, 32, 16, 32]),   # Variable 8 experts
        (16, [16]*8 + [32]*8),                   # Variable 16 experts
        (32, [8]*16 + [32]*16),                  # Variable 32 experts
        (64, [8]*32 + [32]*32),                  # Variable 64 experts
        (128, [4]*64 + [16]*64),                 # Variable 128 experts
    ]

    results = []
    for num_experts, block_counts in expert_configs:
        # Generate test data
        tokens_per_expert = torch.randint(128, 1024, (num_experts,), device=device, dtype=torch.int64)
        tokens_per_expert = ((tokens_per_expert + block_size - 1) // block_size) * block_size
        padded_bins = tokens_per_expert.cumsum(0)
        expert_block_counts = torch.tensor(block_counts, device=device, dtype=torch.int32)

        # Run benchmarks
        ref_time = run_benchmark(benchmark_reference, padded_bins, expert_block_counts, block_size, "reference")
        opt_time = run_benchmark(benchmark_optimized, padded_bins, expert_block_counts, block_size, "optimized")

        speedup = ref_time / opt_time
        results.append({
            'num_experts': num_experts,
            'ref_time_us': ref_time,
            'opt_time_us': opt_time,
            'speedup': speedup,
        })

        print(f"Experts: {num_experts:3d} | Reference: {ref_time:8.2f}µs | Optimized: {opt_time:8.2f}µs | Speedup: {speedup:.2f}x")

    print()
    print("=" * 70)
    print("Summary:")
    print(f"  - The optimized version eliminates {3}N GPU syncs (N=num_experts)")
    print(f"  - With more experts, the speedup is more significant")
    print("=" * 70)


if __name__ == "__main__":
    main()
