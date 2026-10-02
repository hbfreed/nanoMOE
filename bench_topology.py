"""
Benchmark script for topology_var.py performance testing.

Tests the TopologyVarOp with variable-size MoE experts across different batch sizes.
Measures: time per iteration, peak GPU memory, MFU.
"""
import time
import torch
from model import GPTConfig, GPT

# Configuration
BATCH_SIZES = [4, 8]  # Limited by GPU memory (~24GB)
N_CTX = 1024
WARMUP_ITERS = 5
BENCH_ITERS = 10
DEVICE = 'cuda'
DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
COMPILE = False  # Disable compilation for consistent timing of topology ops

# Expert configurations to test: (num_experts, num_experts_per_tok, expert_sizes, name)
# Note: expert sizes must be multiples of block_size (128)
EXPERT_CONFIGS = [
    (8, 2, [(4, 2944), (4, 128)], "8 experts, top-2"),
    (64, 8, [(32, 512), (32, 128)], "64 experts, top-8"),
    (128, 8, [(64, 256), (64, 128)], "128 experts, top-8"),
]

# Set seeds for reproducibility
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.fp32_precision = 'tf32'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DTYPE]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)


def get_batch(batch_size: int):
    """Generate synthetic data (random tokens)."""
    x = torch.randint(50304, (batch_size, N_CTX), device=DEVICE)
    y = torch.randint(50304, (batch_size, N_CTX), device=DEVICE)
    return x, y


def benchmark_config(batch_size: int, num_experts: int, num_experts_per_tok: int, expert_sizes: list) -> dict:
    """Run benchmark for a specific batch size and expert config."""
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create model with MoE and variable-size experts
    gptconf = GPTConfig(
        n_ctx=N_CTX,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0,
        bias=False,
        # MoE configuration
        use_moe=True,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=True,
        block_size=128,
        expert_sizes=expert_sizes,
        load_balance_loss_weight=0.08,
        compute_loss_weight=0.004,
    )
    model = GPT(gptconf)
    model.to(DEVICE)

    optimizer = model.configure_optimizers(
        weight_decay=1e-2,
        learning_rate=1e-4,
        betas=(0.9, 0.95),
        device_type='cuda'
    )

    if COMPILE:
        model = torch.compile(model)

    # Warmup
    X, Y = get_batch(batch_size)
    for _ in range(WARMUP_ITERS):
        with ctx:
            logits, loss, aux_loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        X, Y = get_batch(batch_size)

    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(BENCH_ITERS):
        with ctx:
            logits, loss, aux_loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        X, Y = get_batch(batch_size)

    torch.cuda.synchronize()
    t1 = time.time()

    dt = t1 - t0
    time_per_iter_ms = (dt / BENCH_ITERS) * 1000
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    mfu = model.estimate_mfu(batch_size * BENCH_ITERS, dt)

    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()

    return {
        'batch_size': batch_size,
        'num_experts': num_experts,
        'time_per_iter_ms': time_per_iter_ms,
        'peak_memory_gb': peak_memory_gb,
        'mfu': mfu,
    }


def main():
    print("=" * 70)
    print("Topology Variable Expert Benchmark")
    print("=" * 70)
    print(f"Config: n_ctx={N_CTX}, dtype={DTYPE}, compile={COMPILE}")
    print(f"Warmup: {WARMUP_ITERS} iters, Benchmark: {BENCH_ITERS} iters")
    print("=" * 70)

    for num_experts, num_experts_per_tok, expert_sizes, config_name in EXPERT_CONFIGS:
        print()
        print(f">>> {config_name} <<<")
        print(f"    expert_sizes={expert_sizes}")
        print()

        results = []
        for bs in BATCH_SIZES:
            print(f"  Testing batch_size={bs}...", end=" ", flush=True)
            try:
                result = benchmark_config(bs, num_experts, num_experts_per_tok, expert_sizes)
                results.append(result)
                print(f"done ({result['time_per_iter_ms']:.2f}ms/iter)")
            except torch.cuda.OutOfMemoryError:
                print("OOM - skipping")
                torch.cuda.empty_cache()
                continue

        if results:
            print()
            print(f"  {'Batch':>6} | {'Time/Iter (ms)':>15} | {'Peak Mem (GB)':>14} | {'MFU (%)':>10}")
            print("  " + "-" * 64)
            for r in results:
                print(f"  {r['batch_size']:>6} | {r['time_per_iter_ms']:>15.2f} | {r['peak_memory_gb']:>14.2f} | {r['mfu']*100:>10.2f}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
