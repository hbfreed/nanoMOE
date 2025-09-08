"""
Benchmark script comparing MLP and MoeMLP (Triton kernel) performance.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

# Import the models
from model import MLP, MoeMLP, GPTConfig

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    batch_sizes: List[int] = None
    seq_lengths: List[int] = None
    n_embd: int = 768
    num_experts: int = 8
    num_experts_per_tok: int = 2
    warmup_iters: int = 10
    benchmark_iters: int = 100
    device: str = 'cuda:2'
    dtype: torch.dtype = torch.bfloat16
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16]
        if self.seq_lengths is None:
            self.seq_lengths = [128, 256, 512, 1024, 2048]

def create_gpt_config(bench_config: BenchmarkConfig) -> GPTConfig:
    """Create GPTConfig from BenchmarkConfig."""
    config = GPTConfig()
    config.n_embd = bench_config.n_embd
    config.num_experts = bench_config.num_experts
    config.num_experts_per_tok = bench_config.num_experts_per_tok
    config.dropout = 0.0  # No dropout for benchmarking
    config.bias = False
    config.norm_topk_prob = True
    return config

def benchmark_forward_pass(
    model: nn.Module,
    x: torch.Tensor,
    warmup_iters: int,
    benchmark_iters: int
) -> Dict[str, float]:
    """Benchmark forward pass of a model."""
    
    # Check if this is MoeMLP (returns tuple) or MLP (returns tensor)
    is_moe = hasattr(model, 'router')
    

    # Warmup phase
    for _ in tqdm(range(warmup_iters), desc="Warming up", leave=False):
        with torch.no_grad():
            output = model(x)
            if is_moe:
                output = output[0]  # Extract tensor from tuple

    # Synchronize before timing
    torch.cuda.synchronize()

    # Benchmark phase
    times = []
    for _ in tqdm(range(benchmark_iters), desc="Benchmarking", leave=False):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model(x)
            if is_moe:
                output = output[0]  # Extract tensor from tuple

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)

    times = np.array(times)
    
    return {
        'mean': np.mean(times) * 1000,  # Convert to ms
        'std': np.std(times) * 1000,
        'median': np.median(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
    }

def benchmark_backward_pass(
    model: nn.Module,
    x: torch.Tensor,
    warmup_iters: int,
    benchmark_iters: int
) -> Dict[str, float]:
    """Benchmark backward pass of a model."""
    
    # Check if this is MoeMLP (returns tuple) or MLP (returns tensor)
    is_moe = hasattr(model, 'router')
    
    # Enable gradients
    x = x.requires_grad_(True)
    model.train()
    
    # Warmup phase
    for _ in tqdm(range(warmup_iters), desc="Warming up (backward)", leave=False):
        output = model(x)
        if is_moe:
            output = output[0]  # Extract tensor from tuple
        loss = output.mean()
        loss.backward()
        # Clear gradients
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark phase
    times = []
    for _ in tqdm(range(benchmark_iters), desc="Benchmarking (backward)", leave=False):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = model(x)
        if is_moe:
            output = output[0]  # Extract tensor from tuple
        loss = output.mean()
        loss.backward()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        
        # Clear gradients for next iteration
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
    
    times = np.array(times)
    return {
        'mean': np.mean(times) * 1000,  # Convert to ms
        'std': np.std(times) * 1000,
        'median': np.median(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
    }

def calculate_flops(batch_size: int, seq_len: int, n_embd: int, 
                   is_moe: bool = False, num_experts_per_tok: int = 2) -> float:
    """Calculate theoretical FLOPs for MLP operations."""
    # Standard MLP: input -> 4*n_embd -> n_embd
    # Forward: 2 matrix multiplications
    # fc1: (batch * seq) x n_embd @ n_embd x (4*n_embd) = 2 * batch * seq * n_embd * 4 * n_embd
    # fc2: (batch * seq) x (4*n_embd) @ (4*n_embd) x n_embd = 2 * batch * seq * 4 * n_embd * n_embd
    
    tokens = batch_size * seq_len
    
    if not is_moe:
        # Standard MLP FLOPs
        flops = 2 * tokens * n_embd * 4 * n_embd  # fc1
        flops += 2 * tokens * 4 * n_embd * n_embd  # fc2
    else:
        # MoE MLP FLOPs (only counting active experts)
        # Each token activates num_experts_per_tok experts
        # Each expert processes with reduced dimension: 4*n_embd/num_experts_per_tok
        d_ffn = 4 * n_embd // num_experts_per_tok
        
        # Router computation
        router_flops = 2 * tokens * n_embd * 8  # Assuming 8 experts total
        
        # Expert computation (per token, num_experts_per_tok experts)
        expert_flops = 2 * tokens * num_experts_per_tok * n_embd * d_ffn  # w1
        expert_flops += 2 * tokens * num_experts_per_tok * d_ffn * n_embd  # w2
        
        flops = router_flops + expert_flops
    
    return flops

def run_comparison_benchmark(config: BenchmarkConfig):
    """Run comprehensive benchmark comparing MLP and MoeMLP."""
    
    print("=" * 80)
    print("MLP vs MoeMLP (Triton) Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Hidden dimension: {config.n_embd}")
    print(f"  Number of experts: {config.num_experts}")
    print(f"  Experts per token: {config.num_experts_per_tok}")
    print(f"  Warmup iterations: {config.warmup_iters}")
    print(f"  Benchmark iterations: {config.benchmark_iters}")
    print(f"  Device: {config.device}")
    print(f"  Dtype: {config.dtype}")
    print("=" * 80)
    
    # Create models
    gpt_config = create_gpt_config(config)
    mlp = MLP(gpt_config).to(config.device).to(config.dtype)
    moe_mlp = MoeMLP(gpt_config).to(config.device).to(config.dtype)
    
    # Results storage
    results = []
    
    for batch_size in config.batch_sizes:
        for seq_len in config.seq_lengths:
            print(f"\nBatch size: {batch_size}, Sequence length: {seq_len}")
            print("-" * 60)
            
            # Create input tensor
            x = torch.randn(
                batch_size, seq_len, config.n_embd,
                device=config.device, dtype=config.dtype
            )
            
            # Benchmark MLP forward
            print("  Benchmarking MLP forward...")
            mlp_forward = benchmark_forward_pass(mlp, x, config.warmup_iters, config.benchmark_iters)
            
            # Benchmark MoeMLP forward
            print("  Benchmarking MoeMLP forward...")
            moe_forward = benchmark_forward_pass(moe_mlp, x, config.warmup_iters, config.benchmark_iters)
            
            # Benchmark MLP backward
            print("  Benchmarking MLP backward...")
            mlp_backward = benchmark_backward_pass(mlp, x.clone(), config.warmup_iters, config.benchmark_iters)
            
            # Benchmark MoeMLP backward
            print("  Benchmarking MoeMLP backward...")
            moe_backward = benchmark_backward_pass(moe_mlp, x.clone(), config.warmup_iters, config.benchmark_iters)
            
            # Calculate FLOPs
            mlp_flops = calculate_flops(batch_size, seq_len, config.n_embd, is_moe=False)
            moe_flops = calculate_flops(batch_size, seq_len, config.n_embd, is_moe=True, 
                                       num_experts_per_tok=config.num_experts_per_tok)
            
            # Calculate TFLOPS
            mlp_tflops_forward = (mlp_flops / mlp_forward['mean']) / 1e9  # ms to s, then to TFLOPS
            moe_tflops_forward = (moe_flops / moe_forward['mean']) / 1e9
            
            # Store results
            result = {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'mlp_forward_ms': mlp_forward['mean'],
                'mlp_forward_std': mlp_forward['std'],
                'moe_forward_ms': moe_forward['mean'],
                'moe_forward_std': moe_forward['std'],
                'mlp_backward_ms': mlp_backward['mean'],
                'mlp_backward_std': mlp_backward['std'],
                'moe_backward_ms': moe_backward['mean'],
                'moe_backward_std': moe_backward['std'],
                'mlp_tflops': mlp_tflops_forward,
                'moe_tflops': moe_tflops_forward,
                'speedup_forward': mlp_forward['mean'] / moe_forward['mean'],
                'speedup_backward': mlp_backward['mean'] / moe_backward['mean'],
            }
            results.append(result)
            
            # Print results
            print(f"\n  Results:")
            print(f"    Forward pass:")
            print(f"      MLP:    {mlp_forward['mean']:.3f} � {mlp_forward['std']:.3f} ms ({mlp_tflops_forward:.2f} TFLOPS)")
            print(f"      MoeMLP: {moe_forward['mean']:.3f} � {moe_forward['std']:.3f} ms ({moe_tflops_forward:.2f} TFLOPS)")
            print(f"      Speedup: {result['speedup_forward']:.2f}x")
            print(f"    Backward pass:")
            print(f"      MLP:    {mlp_backward['mean']:.3f} � {mlp_backward['std']:.3f} ms")
            print(f"      MoeMLP: {moe_backward['mean']:.3f} � {moe_backward['std']:.3f} ms")
            print(f"      Speedup: {result['speedup_backward']:.2f}x")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        # Calculate summary statistics
        forward_speedups = [r['speedup_forward'] for r in results]
        backward_speedups = [r['speedup_backward'] for r in results]
        
        # Average speedups
        avg_forward_speedup = np.mean(forward_speedups)
        avg_backward_speedup = np.mean(backward_speedups)
        
        # Best and worst cases
        best_forward_idx = np.argmax(forward_speedups)
        worst_forward_idx = np.argmin(forward_speedups)
        best_backward_idx = np.argmax(backward_speedups)
        worst_backward_idx = np.argmin(backward_speedups)
        
        print(f"\nAverage Speedups:")
        print(f"  Forward:  {avg_forward_speedup:.2f}x")
        print(f"  Backward: {avg_backward_speedup:.2f}x")
        
        print(f"\nBest Forward Speedup: {forward_speedups[best_forward_idx]:.2f}x")
        print(f"  Config: batch={results[best_forward_idx]['batch_size']}, "
              f"seq_len={results[best_forward_idx]['seq_len']}")
        
        print(f"\nWorst Forward Speedup: {forward_speedups[worst_forward_idx]:.2f}x")
        print(f"  Config: batch={results[worst_forward_idx]['batch_size']}, "
              f"seq_len={results[worst_forward_idx]['seq_len']}")
        
        print(f"\nBest Backward Speedup: {backward_speedups[best_backward_idx]:.2f}x")
        print(f"  Config: batch={results[best_backward_idx]['batch_size']}, "
              f"seq_len={results[best_backward_idx]['seq_len']}")
        
        print(f"\nWorst Backward Speedup: {backward_speedups[worst_backward_idx]:.2f}x")
        print(f"  Config: batch={results[worst_backward_idx]['batch_size']}, "
              f"seq_len={results[worst_backward_idx]['seq_len']}")
        
        # Analyze patterns
        print(f"\nPerformance Patterns:")
        
        # Group by batch size
        batch_sizes = sorted(set(r['batch_size'] for r in results))
        for bs in batch_sizes:
            bs_results = [r for r in results if r['batch_size'] == bs]
            if bs_results:
                avg_speedup = np.mean([r['speedup_forward'] for r in bs_results])
                print(f"  Batch size {bs:2d}: avg forward speedup {avg_speedup:.2f}x")
        
        # Calculate efficiency metrics
        print(f"\nEfficiency Metrics:")
        avg_mlp_tflops = np.mean([r['mlp_tflops'] for r in results])
        avg_moe_tflops = np.mean([r['moe_tflops'] for r in results])
        print(f"  Average MLP TFLOPS:    {avg_mlp_tflops:.2f}")
        print(f"  Average MoeMLP TFLOPS: {avg_moe_tflops:.2f}")
        
        # Check if MoeMLP is consistently faster or slower
        if all(s > 1.0 for s in forward_speedups):
            print(f"\n✓ MoeMLP is consistently faster than MLP in forward pass!")
        elif all(s < 1.0 for s in forward_speedups):
            print(f"\n✗ MLP is consistently faster than MoeMLP in forward pass.")
        else:
            print(f"\n~ Performance varies by configuration.")
    
    return results

def cprofile_models(config: BenchmarkConfig):
    """Profile MLP and MoeMLP using cProfile."""
    import cProfile
    import pstats
    from io import StringIO
    
    print("\n" + "=" * 80)
    print("CPROFILING")
    print("=" * 80)
    
    # Use fixed size for profiling
    batch_size = 4
    seq_len = 512
    
    print(f"cProfile with batch_size={batch_size}, seq_len={seq_len}")
    
    # Create models and input
    gpt_config = create_gpt_config(config)
    mlp = MLP(gpt_config).to(config.device).to(config.dtype)
    moe_mlp = MoeMLP(gpt_config).to(config.device).to(config.dtype)
    
    x = torch.randn(
        batch_size, seq_len, config.n_embd,
        device=config.device, dtype=config.dtype, requires_grad=True
    )
    
    # Profile MLP
    print("\n" + "-" * 40)
    print("cProfile MLP (10 forward + backward passes):")
    print("-" * 40)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(10):
        output = mlp(x)
        loss = output.mean()
        loss.backward()
        mlp.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    # Save to file
    profiler.dump_stats('mlp_cprofile.stats')
    print("MLP cProfile stats saved to mlp_cprofile.stats")
    
    # Profile MoeMLP
    print("\n" + "-" * 40)
    print("cProfile MoeMLP (10 forward + backward passes):")
    print("-" * 40)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(10):
        output = moe_mlp(x)
        if isinstance(output, tuple):
            output = output[0]
        loss = output.mean()
        loss.backward()
        moe_mlp.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    # Save to file
    profiler.dump_stats('moe_mlp_cprofile.stats')
    print("MoeMLP cProfile stats saved to moe_mlp_cprofile.stats")
    
    print("\nTip: You can analyze the .stats files with:")
    print("  python3 -m pstats mlp_cprofile.stats")
    print("  python3 -m pstats moe_mlp_cprofile.stats")

def profile_models(config: BenchmarkConfig):
    """Profile MLP and MoeMLP using PyTorch profiler."""
    import torch.profiler as profiler
    
    print("\n" + "=" * 80)
    print("PROFILING (PyTorch)")
    print("=" * 80)
    
    # Use fixed size for profiling
    batch_size = 4
    seq_len = 512
    
    print(f"Profiling with batch_size={batch_size}, seq_len={seq_len}")
    
    # Create models and input
    gpt_config = create_gpt_config(config)
    mlp = MLP(gpt_config).to(config.device).to(config.dtype)
    moe_mlp = MoeMLP(gpt_config).to(config.device).to(config.dtype)
    
    x = torch.randn(
        batch_size, seq_len, config.n_embd,
        device=config.device, dtype=config.dtype, requires_grad=True
    )
    
    # Profile MLP
    print("\nProfiling MLP...")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with profiler.record_function("MLP_forward"):
            for _ in range(10):
                output = mlp(x)
                loss = output.mean()
                loss.backward()
    
    print("\nMLP Profile Results:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Save trace
    prof.export_chrome_trace("mlp_trace.json")
    print("MLP trace saved to mlp_trace.json")
    
    # Clear gradients
    mlp.zero_grad()
    x.grad.zero_()
    
    # Profile MoeMLP
    print("\nProfiling MoeMLP...")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with profiler.record_function("MoeMLP_forward"):
            for _ in range(10):
                output = moe_mlp(x)
                output = output[0] if isinstance(output, tuple) else output  # Handle MoeMLP tuple output
                loss = output.mean()
                loss.backward()
    
    print("\nMoeMLP Profile Results:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Save trace
    prof.export_chrome_trace("moe_mlp_trace.json")
    print("MoeMLP trace saved to moe_mlp_trace.json")

def main():
    parser = argparse.ArgumentParser(description='Benchmark MLP vs MoeMLP')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8],
                       help='Batch sizes to benchmark')
    parser.add_argument('--seq-lengths', nargs='+', type=int, default=[128, 256, 512, 1024],
                       help='Sequence lengths to benchmark')
    parser.add_argument('--n-embd', type=int, default=768,
                       help='Hidden dimension')
    parser.add_argument('--num-experts', type=int, default=8,
                       help='Number of experts in MoE')
    parser.add_argument('--num-experts-per-tok', type=int, default=2,
                       help='Number of experts per token')
    parser.add_argument('--warmup-iters', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--benchmark-iters', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--profile', action='store_true',
                       help='Run PyTorch profiling after benchmarking')
    parser.add_argument('--cprofile', action='store_true',
                       help='Run cProfile after benchmarking')
    parser.add_argument('--dtype', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'bf16'],
                       help='Data type to use (default: bf16)')
    parser.add_argument('--device', type=str, default='cuda:2',
                       help='Device to use (default: cuda:2)')
    
    args = parser.parse_args()
    
    # Map dtype string to torch dtype
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    
    # Create config
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        n_embd=args.n_embd,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        dtype=dtype_map[args.dtype],
        device=args.device
    )
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        return
    
    # Extract device index if specified
    device_idx = 0
    if ':' in config.device:
        device_idx = int(config.device.split(':')[1])
    
    # Set CUDA device to ensure proper context for Triton kernels
    torch.cuda.set_device(device_idx)
    
    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(device_idx)}")
    print(f"Using device: {config.device}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Run benchmarks
    results = run_comparison_benchmark(config)
    
    # Run PyTorch profiling if requested
    if args.profile:
        profile_models(config)
    
    # Run cProfile if requested
    if args.cprofile:
        cprofile_models(config)

if __name__ == "__main__":
    main()