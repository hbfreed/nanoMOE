import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange, einsum

# Consistent configuration parameters
BATCH_SIZE = 4
SEQ_LEN = 128
HIDDEN_DIM = 512
NUM_EXPERTS = 16
NUM_EXPERTS_PER_TOK = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16

torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.allow_tf32        = True
torch.set_float32_matmul_precision('high')

# Helper MLP class for the for-loop version
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# For-loop implementation (based on OLMoE)
class MoeMLP_ForLoop(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts # e
        self.num_experts_per_tok = config.num_experts_per_tok # k
        self.norm_topk_prob = config.norm_topk_prob # bool
        
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            MLP(config) for _ in range(self.num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        # Router logits and weights
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # One-hot expert mask
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = rearrange(expert_mask, 'num_tokens num_experts_per_tok num_experts -> num_experts num_experts_per_tok num_tokens') #n = num_tokens (batch_size * seq_length)
        # Dispatch to experts (FOR LOOP 🤮)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if len(top_x) > 0:
                current_state = x_flat[top_x]
                current_output = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
                output.index_add_(0, top_x, current_output.to(x.dtype))
        
        return output.view(batch_size, seq_len, hidden_dim), router_logits

class MoeMLP_Batched(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts  # e
        self.num_experts_per_tok = config.num_experts_per_tok  # k
        self.norm_topk_prob = config.norm_topk_prob  # bool
        
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
        # Store expert weights as tensors for batched processing
        self.expert_w1 = nn.Parameter(torch.empty(self.num_experts, config.n_embd, 4 * config.n_embd))
        self.expert_w2 = nn.Parameter(torch.empty(self.num_experts, 4 * config.n_embd, config.n_embd))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.expert_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.expert_w2, a=math.sqrt(5))
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [num_tokens, hidden_dim]
        
        # Router logits and weights
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        
        # Create one-hot encoding for selected experts
        # This avoids the memory explosion from weight duplication
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).to(x.dtype)
        # expert_mask: [num_tokens, num_experts_per_tok, num_experts]
        
        # Use einsum for memory-efficient batched processing
        # First layer: compute x @ w1 for all experts, then select
        # x_flat: [num_tokens, hidden_dim]
        # expert_w1: [num_experts, hidden_dim, 4*hidden_dim]
        all_h1 = torch.einsum('th,ehd->ted', x_flat, self.expert_w1)  # [num_tokens, num_experts, 4*hidden_dim]
        
        # Select only the chosen experts using the mask
        # expert_mask: [num_tokens, num_experts_per_tok, num_experts]
        # all_h1: [num_tokens, num_experts, 4*hidden_dim]
        h1 = torch.einsum('tke,ted->tkd', expert_mask, all_h1)  # [num_tokens, num_experts_per_tok, 4*hidden_dim]
        
        # Apply activation
        h1 = F.gelu(h1)
        
        # Second layer: h1 @ w2 for selected experts
        # We need to map back to full expert space, apply w2, then select again
        # First, expand h1 back to full expert space
        h1_expanded = torch.einsum('tkd,tke->ted', h1, expert_mask)  # [num_tokens, num_experts, 4*hidden_dim]
        
        # Apply second weight matrix
        # h1_expanded: [num_tokens, num_experts, 4*hidden_dim]
        # expert_w2: [num_experts, 4*hidden_dim, hidden_dim]
        all_h2 = torch.einsum('ted,edh->teh', h1_expanded, self.expert_w2)  # [num_tokens, num_experts, hidden_dim]
        
        # Select and weight the final outputs
        # expert_mask: [num_tokens, num_experts_per_tok, num_experts]
        # all_h2: [num_tokens, num_experts, hidden_dim]
        h2 = torch.einsum('tke,teh->tkh', expert_mask, all_h2)  # [num_tokens, num_experts_per_tok, hidden_dim]
        
        # Apply routing weights and sum
        # routing_weights: [num_tokens, num_experts_per_tok]
        # h2: [num_tokens, num_experts_per_tok, hidden_dim]
        output = torch.einsum('tk,tkh->th', routing_weights, h2)  # [num_tokens, hidden_dim]
        
        return output.view(batch_size, seq_len, hidden_dim), router_logits

def benchmark_moe(moe_module, x, num_runs=1000, warmup=100):
    """Benchmark MoE forward pass"""
    # Warmup
    for _ in range(warmup):
        _, _ = moe_module(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_runs):
        output, router_logits = moe_module(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time

def run_benchmarks():
    # Config
    batch_size = BATCH_SIZE
    seq_len = SEQ_LEN
    hidden_dim = HIDDEN_DIM
    num_experts = NUM_EXPERTS
    num_experts_per_tok = NUM_EXPERTS_PER_TOK
    device = DEVICE
    
    # Create config object
    class Config:
        n_embd = hidden_dim
        num_experts = NUM_EXPERTS
        num_experts_per_tok = NUM_EXPERTS_PER_TOK
        norm_topk_prob = True
        bias = False
        dropout = 0.0
    
    config = Config()
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=DTYPE)
    
    print(f"Benchmarking MoE implementations:")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}, Hidden dim: {hidden_dim}")
    print(f"Num experts: {num_experts}, Experts per token: {num_experts_per_tok}")
    print(f"Device: {device}\n")
    
    # Test for-loop version
    moe_loop = MoeMLP_ForLoop(config).to(device).to(DTYPE)
    moe_loop = torch.compile(moe_loop)
    with torch.no_grad():
        time_loop = benchmark_moe(moe_loop, x)
    print(f"For-loop version: {time_loop:.2f} ms")
    
    # Test batched version
    moe_batched = MoeMLP_Batched(config).to(device).to(DTYPE)
    moe_batched = torch.compile(moe_batched)
    with torch.no_grad():
        time_batched = benchmark_moe(moe_batched, x)
    print(f"Batched version: {time_batched:.2f} ms")

    print(f"\nSpeedup:")
    print(f"For-loop vs Batched: {time_loop/time_batched:.2f}x")

    
    # Memory usage (only on CUDA)
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = moe_loop(x)
        memory_loop = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = moe_batched(x)
        memory_batched = torch.cuda.max_memory_allocated() / 1024**2  # MB

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print(f"\nMemory usage:")
        print(f"For-loop: {memory_loop:.2f} MB")
        print(f"Batched: {memory_batched:.2f} MB")

def benchmark_regular_mlp():
    """Benchmark a regular MLP for comparison"""
    if not torch.cuda.is_available():
        print("\nSkipping regular MLP benchmark (requires CUDA)")
        return
        
    class SimpleMLP(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
            self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim)
        
        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))
    
    hidden_dim = HIDDEN_DIM
    mlp = SimpleMLP(hidden_dim).cuda().to(DTYPE)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, hidden_dim, device='cuda', dtype=DTYPE)
    
    with torch.no_grad():
        avg_time = benchmark_moe(lambda x: (mlp(x), None), x)
    
    print(f"\nRegular MLP (baseline): {avg_time:.2f} ms")

def benchmark_scaling():
    """Test how performance scales with number of experts"""
    if not torch.cuda.is_available():
        print("\nSkipping scaling benchmark (requires CUDA)")
        return
        
    print("\n" + "="*50)
    print("Scaling with number of experts:")
    print("="*50)
    
    expert_counts = [4, 8, 16, 32, 64]
    hidden_dim = HIDDEN_DIM
    batch_size = BATCH_SIZE
    seq_len = SEQ_LEN
    
    for num_experts in expert_counts:
        current_num_experts = num_experts
        class Config:
            n_embd = hidden_dim
            num_experts = current_num_experts
            num_experts_per_tok = NUM_EXPERTS_PER_TOK
            norm_topk_prob = True
            bias = False
            dropout = 0.0
        
        config = Config()
        x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=DTYPE)
        
        # Only test batched version for scaling
        moe = MoeMLP_Batched(config).cuda().to(DTYPE)
        
        with torch.no_grad():
            avg_time = benchmark_moe(moe, x, num_runs=50)
        
        print(f"Experts: {num_experts:3d} | Time: {avg_time:6.2f} ms")

def benchmark_routing_overhead():
    """Measure just the routing computation overhead"""
    if not torch.cuda.is_available():
        print("\nSkipping routing overhead benchmark (requires CUDA)")
        return
        
    print("\n" + "="*50)
    print("Routing overhead breakdown:")
    print("="*50)
    
    batch_size = BATCH_SIZE
    seq_len = SEQ_LEN
    hidden_dim = HIDDEN_DIM
    num_experts = NUM_EXPERTS
    num_experts_per_tok = NUM_EXPERTS_PER_TOK
    num_runs = 1_000_000
    
    x = torch.randn(batch_size * seq_len, hidden_dim, device='cuda', dtype=DTYPE)
    router = nn.Linear(hidden_dim, num_experts, bias=False).cuda().to(DTYPE)
    
    # Benchmark router forward pass
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        router_logits = router(x)
    torch.cuda.synchronize()
    router_time = (time.time() - start) / num_runs * 1000
    
    # Benchmark softmax + topk
    router_logits = router(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
    torch.cuda.synchronize()
    routing_time = (time.time() - start) / num_runs * 1000
    
    # Benchmark sorting
    selected_experts_flat = selected_experts.view(-1)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        sorted_experts, sort_indices = selected_experts_flat.sort(stable=True)
    torch.cuda.synchronize()
    sort_time = (time.time() - start) / num_runs * 1000
    
    print(f"Router forward: {router_time:.3f} ms")
    print(f"Softmax + TopK: {routing_time:.3f} ms")
    print(f"Sorting: {sort_time:.3f} ms")
    print(f"Total routing overhead: {router_time + routing_time + sort_time:.3f} ms")

def verify_implementations():
    """Verify that for-loop and batched implementations produce identical results"""
    print("="*60)
    print("VERIFICATION: Testing implementation correctness")
    print("="*60)
    
    # Use smaller, deterministic setup for verification
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    n_experts = 4
    n_experts_per_tok = 2
    
    class Config:
        n_embd = hidden_dim
        num_experts = n_experts
        num_experts_per_tok = n_experts_per_tok
        norm_topk_prob = True
        bias = False
        dropout = 0.0
    
    config = Config()
    
    # Create models
    moe_loop = MoeMLP_ForLoop(config).to(DEVICE).to(DTYPE)
    moe_batched = MoeMLP_Batched(config).to(DEVICE).to(DTYPE)
    
    # Important: Make sure both models have the same weights!
    with torch.no_grad():
        # Copy router weights
        moe_batched.router.weight.data = moe_loop.router.weight.data.clone()
        
        # Copy expert weights
        for i in range(n_experts):
            mlp = moe_loop.experts[i]
            
            # DEBUG: Print shapes and check weight copying
            print(f"Expert {i}:")
            print(f"  Original c_fc.weight shape: {mlp.c_fc.weight.shape}")
            print(f"  Original c_proj.weight shape: {mlp.c_proj.weight.shape}")
            
            # Copy weights
            moe_batched.expert_w1[i] = mlp.c_fc.weight.data.T.clone()
            moe_batched.expert_w2[i] = mlp.c_proj.weight.data.T.clone()
            
            print(f"  Copied expert_w1[{i}] shape: {moe_batched.expert_w1[i].shape}")
            print(f"  Copied expert_w2[{i}] shape: {moe_batched.expert_w2[i].shape}")
    
    # Test with one fixed seed
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=DEVICE, dtype=DTYPE)
    
    # Run both implementations with detailed debugging
    with torch.no_grad():
        print("\n=== FOR-LOOP VERSION ===")
        output_loop, router_logits_loop = moe_loop(x)
        print(f"Output shape: {output_loop.shape}")
        print(f"Output sample: {output_loop[0,0,:4]}")
        
        print("\n=== BATCHED VERSION ===")
        output_batched, router_logits_batched = moe_batched(x)
        print(f"Output shape: {output_batched.shape}")
        print(f"Output sample: {output_batched[0,0,:4]}")
    
    # Compare outputs
    output_diff = torch.abs(output_loop - output_batched).max().item()
    router_diff = torch.abs(router_logits_loop - router_logits_batched).max().item()
    
    print(f"\n=== COMPARISON ===")
    print(f"Output max diff: {output_diff:.2e}")
    print(f"Router max diff: {router_diff:.2e}")
    print(f"Output diff sample: {torch.abs(output_loop - output_batched)[0,0,:4]}")
    
    # Check if they're close enough (more realistic tolerance for bfloat16)
    output_close = torch.allclose(output_loop, output_batched, atol=1e-2, rtol=1e-3)  # Much more relaxed
    router_close = torch.allclose(router_logits_loop, router_logits_batched, atol=1e-5, rtol=1e-4)
    
    print(f"Output match: {'✅' if output_close else '❌'}")
    print(f"Router match: {'✅' if router_close else '❌'}")
    
    return output_close and router_close

def comprehensive_moe_testing():
    """Test MoE implementations across different configurations"""
    print("="*80)
    print("COMPREHENSIVE MoE TESTING")
    print("="*80)
    
    # Test configurations
    test_configs = [
        # Small configurations
        {"experts": 4, "experts_per_tok": 2, "hidden": 256, "batch": 2, "seq": 64},
        {"experts": 8, "experts_per_tok": 2, "hidden": 512, "batch": 4, "seq": 128},
        
        # Medium configurations  
        {"experts": 16, "experts_per_tok": 4, "hidden": 512, "batch": 8, "seq": 256},
        {"experts": 32, "experts_per_tok": 4, "hidden": 1024, "batch": 4, "seq": 512},
        
        # Large configurations
        {"experts": 64, "experts_per_tok": 8, "hidden": 1024, "batch": 2, "seq": 1024},
        {"experts": 128, "experts_per_tok": 8, "hidden": 2048, "batch": 1, "seq": 2048},
        
        # Edge cases
        {"experts": 16, "experts_per_tok": 1, "hidden": 512, "batch": 4, "seq": 128},  # Low k
        {"experts": 8, "experts_per_tok": 8, "hidden": 512, "batch": 4, "seq": 128},   # High k (all experts)
        {"experts": 256, "experts_per_tok": 2, "hidden": 512, "batch": 2, "seq": 128}, # Many experts
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Test {i+1}/{len(test_configs)} ---")
        print(f"Experts: {config['experts']}, K: {config['experts_per_tok']}, "
              f"Hidden: {config['hidden']}, Batch: {config['batch']}, Seq: {config['seq']}")
        
        try:
            result = run_single_moe_test(config)
            results.append(result)
            
            # Print key metrics
            print(f"✅ SUCCESS")
            print(f"   Speedup: {result['speedup']:.2f}x")
            print(f"   Memory ratio: {result['memory_ratio']:.2f}x")
            print(f"   Verification: {'✅' if result['verification_passed'] else '❌'}")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            results.append({"config": config, "error": str(e)})
    
    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    successful_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]
    
    print(f"Successful tests: {len(successful_results)}/{len(results)}")
    print(f"Failed tests: {len(failed_results)}/{len(results)}")
    
    if successful_results:
        speedups = [r['speedup'] for r in successful_results]
        memory_ratios = [r['memory_ratio'] for r in successful_results]
        
        print(f"\nSpeedup statistics:")
        print(f"  Mean: {sum(speedups)/len(speedups):.2f}x")
        print(f"  Min: {min(speedups):.2f}x")
        print(f"  Max: {max(speedups):.2f}x")
        
        print(f"\nMemory ratio statistics:")
        print(f"  Mean: {sum(memory_ratios)/len(memory_ratios):.2f}x")
        print(f"  Min: {min(memory_ratios):.2f}x")
        print(f"  Max: {max(memory_ratios):.2f}x")
    
    if failed_results:
        print(f"\nFailed configurations:")
        for r in failed_results:
            config = r['config']
            print(f"  Experts: {config['experts']}, K: {config['experts_per_tok']}, "
                  f"Hidden: {config['hidden']} -> {r['error']}")

def run_single_moe_test(config):
    """Run a single MoE test configuration"""
    # Create config object
    class Config:
        n_embd = config['hidden']
        num_experts = config['experts']
        num_experts_per_tok = config['experts_per_tok']
        norm_topk_prob = True
        bias = False
        dropout = 0.0
    
    moe_config = Config()
    
    # Create input
    x = torch.randn(config['batch'], config['seq'], config['hidden'], 
                   device=DEVICE, dtype=DTYPE)
    
    # Create models
    moe_loop = MoeMLP_ForLoop(moe_config).to(DEVICE).to(DTYPE)
    moe_batched = MoeMLP_Batched(moe_config).to(DEVICE).to(DTYPE)
    
    # Copy weights for verification
    with torch.no_grad():
        moe_batched.router.weight.data = moe_loop.router.weight.data.clone()
        for i in range(config['experts']):
            mlp = moe_loop.experts[i]
            moe_batched.expert_w1[i] = mlp.c_fc.weight.data.T.clone()
            moe_batched.expert_w2[i] = mlp.c_proj.weight.data.T.clone()
    
    # Quick verification
    torch.manual_seed(42)
    x_test = torch.randn(2, 4, config['hidden'], device=DEVICE, dtype=DTYPE)
    
    with torch.no_grad():
        out_loop, _ = moe_loop(x_test)
        out_batched, _ = moe_batched(x_test)
    
    verification_passed = torch.allclose(out_loop, out_batched, atol=1e-2, rtol=1e-3)
    
    # Performance benchmarking
    num_runs = min(100, max(10, 1000 // config['experts']))  # Adaptive run count
    
    with torch.no_grad():
        time_loop = benchmark_moe(moe_loop, x, num_runs=num_runs, warmup=10)
        time_batched = benchmark_moe(moe_batched, x, num_runs=num_runs, warmup=10)
    
    speedup = time_loop / time_batched
    
    # Memory usage
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = moe_loop(x)
        memory_loop = torch.cuda.max_memory_allocated() / 1024**2
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = moe_batched(x)
        memory_batched = torch.cuda.max_memory_allocated() / 1024**2
        
        memory_ratio = memory_batched / memory_loop
    else:
        memory_loop = memory_batched = memory_ratio = 0.0
    
    return {
        "config": config,
        "speedup": speedup,
        "time_loop": time_loop,
        "time_batched": time_batched,
        "memory_loop": memory_loop,
        "memory_batched": memory_batched,
        "memory_ratio": memory_ratio,
        "verification_passed": verification_passed
    }

def validate_timing():
    """Quick sanity check on timing methodology"""

    # Super simple baseline - just matrix multiplication
    x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(512, 2048, device='cuda', dtype=torch.bfloat16)

    # Compile the matmul function
    @torch.compile
    def matmul_fn(x, w):
        return torch.mm(x, w)

    # Time a simple matmul
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        result = matmul_fn(x, w)
    torch.cuda.synchronize()
    simple_time = (time.time() - start) / 1000 * 1000

    print(f"Simple 512x512 @ 512x2048 matmul: {simple_time:.3f} ms")

    # Now time our einsum
    expert_w1 = torch.randn(16, 512, 2048, device='cuda', dtype=torch.bfloat16)

    # Compile the einsum function
    @torch.compile
    def einsum_fn(x, expert_w1):
        return torch.einsum('th,ehd->ted', x, expert_w1)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        result = einsum_fn(x, expert_w1)
    torch.cuda.synchronize()
    einsum_time = (time.time() - start) / 1000 * 1000

    print(f"Einsum 512x512 with 16 experts: {einsum_time:.3f} ms")
    print(f"Ratio: {einsum_time/simple_time:.1f}x slower")

if __name__ == "__main__":
    # First verify correctness
    verify_implementations()
    
    # Comprehensive testing
    comprehensive_moe_testing()
    
    # Original benchmarks
    run_benchmarks()
    benchmark_regular_mlp()
    benchmark_scaling()
    # benchmark_routing_overhead()
    validate_timing()