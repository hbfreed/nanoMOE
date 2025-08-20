import torch
import torch.nn as nn
from dataclasses import dataclass
from model import MoeMLP, MLP
import torch.nn.functional as F

@dataclass  
class TestConfig:
    num_experts: int = 4
    num_experts_per_tok: int = 2  
    n_embd: int = 32
    norm_topk_prob: bool = True
    block_size: int = 32
    block_k: int = 32
    bias: bool = True
    dropout: float = 0.0

def test_full_moe_pipeline():
    """
    Test the complete MoE pipeline: SDD -> GELU -> DSD
    Compare against a naive implementation that manually routes tokens to experts.
    """
    config = TestConfig()
    torch.manual_seed(42)
    
    if not torch.cuda.is_available():
        print("CUDA not available - Triton kernels require GPU. Skipping test.")
        return
    
    device = torch.device('cuda')
    
    # Create test input
    batch_size, seq_len = 2, 4
    x = torch.randn(batch_size, seq_len, config.n_embd, dtype=torch.float32, device=device)
    
    # Initialize MoeMLP with kernels
    moe_mlp = MoeMLP(
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok, 
        n_embd=config.n_embd,
        norm_topk_prob=config.norm_topk_prob,
        block_size=config.block_size,
        block_k=config.block_k
    ).to(device)
    
    # Run the full MoeMLP forward pass (SDD -> GELU -> DSD)
    with torch.no_grad():
        kernel_output = moe_mlp(x)
    
    # Now compute the naive version for comparison
    x_flat = x.view(-1, config.n_embd)
    num_tokens = x_flat.shape[0]
    
    with torch.no_grad():
        # Route tokens
        router_logits = moe_mlp.router(x_flat)
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        router_weights, selected_experts = torch.topk(router_weights, config.num_experts_per_tok, dim=-1)
        
        if config.norm_topk_prob:
            router_weights /= router_weights.sum(dim=-1, keepdim=True)
        
        # Naive implementation: process each token-expert pair
        naive_output = torch.zeros_like(x_flat)
        
        for token_idx in range(num_tokens):
            for k in range(config.num_experts_per_tok):
                expert_idx = selected_experts[token_idx, k].item()
                weight = router_weights[token_idx, k].item()
                
                # Get expert's weights
                w1_start = expert_idx * config.n_embd * 4
                w1_end = (expert_idx + 1) * config.n_embd * 4
                expert_w1 = moe_mlp.w1[:, w1_start:w1_end]
                
                w2_start = expert_idx * config.n_embd * 4
                w2_end = (expert_idx + 1) * config.n_embd * 4
                expert_w2 = moe_mlp.w2[w2_start:w2_end, :]
                
                # Forward through expert
                hidden = x_flat[token_idx:token_idx+1] @ expert_w1
                hidden = F.gelu(hidden)
                expert_out = hidden @ expert_w2
                
                # Scale and accumulate
                naive_output[token_idx] += weight * expert_out.squeeze()
    
    # Handle shape mismatch - kernel output might be padded
    kernel_output_flat = kernel_output.view(-1, config.n_embd)
    
    # Only compare the first num_tokens (ignore padding)
    if kernel_output_flat.shape[0] > num_tokens:
        print(f"Note: Kernel output is padded ({kernel_output_flat.shape[0]} tokens), comparing first {num_tokens} tokens only")
        kernel_output_to_compare = kernel_output_flat[:num_tokens]
    else:
        kernel_output_to_compare = kernel_output_flat
    
    # Compare
    max_diff = (kernel_output_to_compare - naive_output).abs().max().item()
    mean_diff = (kernel_output_to_compare - naive_output).abs().mean().item()
    
    print(f"Full MoE Pipeline Test Results:")
    print(f"  Original input shape: {x.shape}")
    print(f"  Kernel output shape: {kernel_output.shape}")
    print(f"  Comparing {num_tokens} tokens")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✅ Test PASSED - kernels match naive implementation!")
    else:
        print("❌ Test FAILED - significant difference detected")
        # Debug info
        print(f"\nDebug info:")
        print(f"  Kernel output range: [{kernel_output_to_compare.min():.4f}, {kernel_output_to_compare.max():.4f}]")
        print(f"  Naive output range: [{naive_output.min():.4f}, {naive_output.max():.4f}]")

def test_full_moe_pipeline_debug():
    """
    Debug version with extensive prints to identify issues
    """
    config = TestConfig()
    torch.manual_seed(42)
    
    if not torch.cuda.is_available():
        print("CUDA not available - Triton kernels require GPU. Skipping test.")
        return
    
    device = torch.device('cuda')
    
    # Create test input
    batch_size, seq_len = 2, 4
    x = torch.randn(batch_size, seq_len, config.n_embd, dtype=torch.float32, device=device)
    
    # Initialize MoeMLP with kernels
    moe_mlp = MoeMLP(
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok, 
        n_embd=config.n_embd,
        norm_topk_prob=config.norm_topk_prob,
        block_size=config.block_size,
        block_k=config.block_k
    ).to(device)
    
    # Run the full MoeMLP forward pass
    with torch.no_grad():
        kernel_output = moe_mlp(x)
    
    print(f"Kernel output shape: {kernel_output.shape}")
    print(f"First few kernel outputs:\n{kernel_output[:3, :5]}")
    
    # Now compute the naive version
    x_flat = x.view(-1, config.n_embd)
    num_tokens = x_flat.shape[0]
    
    with torch.no_grad():
        # Route tokens
        router_logits = moe_mlp.router(x_flat)
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        router_weights, selected_experts = torch.topk(router_weights, config.num_experts_per_tok, dim=-1)
        
        if config.norm_topk_prob:
            router_weights /= router_weights.sum(dim=-1, keepdim=True)
        
        print(f"\nSelected experts for each token:\n{selected_experts}")
        print(f"Router weights:\n{router_weights}")
        
        # Naive implementation with debug
        naive_output = torch.zeros_like(x_flat)
        
        for token_idx in range(min(3, num_tokens)):  # Debug first 3 tokens
            print(f"\nToken {token_idx}:")
            token_out = torch.zeros(config.n_embd, device=device)
            
            for k in range(config.num_experts_per_tok):
                expert_idx = selected_experts[token_idx, k].item()
                weight = router_weights[token_idx, k].item()
                print(f"  Expert {expert_idx}, weight {weight:.4f}")
                
                # Get expert's weights
                w1_start = expert_idx * config.n_embd * 4
                w1_end = (expert_idx + 1) * config.n_embd * 4
                expert_w1 = moe_mlp.w1[:, w1_start:w1_end]
                
                w2_start = expert_idx * config.n_embd * 4
                w2_end = (expert_idx + 1) * config.n_embd * 4
                expert_w2 = moe_mlp.w2[w2_start:w2_end, :]
                
                print(f"    w1 slice norm: {expert_w1.norm().item():.4f}")
                print(f"    w2 slice norm: {expert_w2.norm().item():.4f}")
                
                # Forward through expert
                hidden = x_flat[token_idx:token_idx+1] @ expert_w1
                print(f"    After w1 norm: {hidden.norm().item():.4f}")
                hidden = F.gelu(hidden)
                print(f"    After GELU norm: {hidden.norm().item():.4f}")
                expert_out = hidden @ expert_w2
                print(f"    Expert output norm: {expert_out.norm().item():.4f}")
                
                # Scale and accumulate
                token_out += weight * expert_out.squeeze()
            
            naive_output[token_idx] = token_out
            print(f"  Final token output norm: {token_out.norm().item():.4f}")
    
    # Compare
    kernel_output_flat = kernel_output.view(-1, config.n_embd)
    kernel_output_to_compare = kernel_output_flat[:num_tokens]
    
    print(f"\nComparison:")
    for i in range(min(3, num_tokens)):
        print(f"Token {i}:")
        print(f"  Kernel norm: {kernel_output_to_compare[i].norm().item():.4f}")
        print(f"  Naive norm:  {naive_output[i].norm().item():.4f}")
        print(f"  Diff norm:   {(kernel_output_to_compare[i] - naive_output[i]).norm().item():.4f}")

def debug_weight_initialization():
    """Check if weights are initialized properly"""
    config = TestConfig()
    torch.manual_seed(42)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    device = torch.device('cuda')
    
    moe_mlp = MoeMLP(
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        n_embd=config.n_embd,
        norm_topk_prob=config.norm_topk_prob,
        block_size=config.block_size,
        block_k=config.block_k
    ).to(device)
    
    print(f"w1 shape: {moe_mlp.w1.shape}")
    print(f"w2 shape: {moe_mlp.w2.shape}")
    print(f"Expected w1: ({config.n_embd}, {config.num_experts * config.n_embd * 4})")
    print(f"Expected w2: ({config.num_experts * config.n_embd * 4}, {config.n_embd})")
    
    # Check if weights are initialized (non-zero)
    for expert_idx in range(config.num_experts):
        w1_start = expert_idx * config.n_embd * 4
        w1_end = (expert_idx + 1) * config.n_embd * 4
        expert_w1 = moe_mlp.w1[:, w1_start:w1_end]
        
        w2_start = expert_idx * config.n_embd * 4
        w2_end = (expert_idx + 1) * config.n_embd * 4
        expert_w2 = moe_mlp.w2[w2_start:w2_end, :]
        
        print(f"\nExpert {expert_idx}:")
        print(f"  w1 slice [{0}:{config.n_embd}, {w1_start}:{w1_end}] norm: {expert_w1.norm().item():.4f}")
        print(f"  w2 slice [{w2_start}:{w2_end}, {0}:{config.n_embd}] norm: {expert_w2.norm().item():.4f}")
        
        if expert_w1.norm() < 1e-6:
            print(f"  WARNING: Expert {expert_idx} w1 is near zero!")
        if expert_w2.norm() < 1e-6:
            print(f"  WARNING: Expert {expert_idx} w2 is near zero!")

def test_dsd_kernel_simple():
    """Super simple DSD test with known values"""
    config = TestConfig()
    
    if not torch.cuda.is_available():
        return
    
    device = torch.device('cuda')
    
    # Very simple case - single block
    num_tokens = 32  
    d_ffn = 32
    hidden_size = 32
    
    # Simple inputs - all ones scaled
    block_sparse = torch.ones(num_tokens, d_ffn, device=device) * 0.1
    w2 = torch.ones(d_ffn * 2, hidden_size, device=device) * 0.01  # 2 experts
    
    # All tokens to expert 0
    row_indices = torch.zeros(1, device=device, dtype=torch.int32)
    weight_row_indices = torch.zeros(1, device=device, dtype=torch.int32)
    
    from model import dsd_kernel
    import triton
    
    expert_output = torch.zeros(num_tokens, hidden_size, device=device)
    
    print(f"Input shape: {block_sparse.shape}")
    print(f"Weight shape: {w2.shape}")
    print(f"Output shape: {expert_output.shape}")
    
    dsd_kernel[(1, 1)](
        block_sparse, w2, expert_output,
        row_indices_ptr=row_indices,
        weight_row_indices_ptr=weight_row_indices,
        stride_xm=block_sparse.stride()[0], 
        stride_xk=block_sparse.stride()[1],
        stride_wk=w2.stride()[0], 
        stride_wn=w2.stride()[1],
        stride_om=expert_output.stride()[0], 
        stride_on=expert_output.stride()[1],
        d_ffn=d_ffn, 
        hidden_size=hidden_size,
        BLOCK_SIZE=32,
        BLOCK_K=32
    )
    
    # Expected: 0.1 * 32 * 0.01 = 0.032 for each element
    expected = torch.ones_like(expert_output) * 0.032
    
    max_diff = (expert_output - expected).abs().max().item()
    print(f"\nSimple DSD test (all ones):")
    print(f"  Expected value: 0.032")
    print(f"  Actual first value: {expert_output[0, 0].item():.6f}")
    print(f"  Max difference: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✅ Simple test PASSED!")
    else:
        print("❌ Simple test FAILED")
        print(f"First row of output:\n{expert_output[0, :5]}")
        print(f"Expected:\n{expected[0, :5]}")

def test_dsd_kernel_with_different_experts():
    """Test DSD with multiple experts"""
    config = TestConfig()
    
    if not torch.cuda.is_available():
        return
    
    device = torch.device('cuda')
    
    # Two blocks, different experts
    num_tokens = 64
    d_ffn = config.n_embd * 4
    
    # Create distinct patterns for each expert
    block_sparse = torch.randn(num_tokens, d_ffn, device=device) * 0.1
    
    # Create w2 with distinct patterns per expert
    w2 = torch.randn(d_ffn * config.num_experts, config.n_embd, device=device) * 0.02
    
    # First block -> expert 0, second block -> expert 1
    row_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    weight_row_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    
    from model import dsd_kernel
    import triton
    
    expert_output = torch.zeros(num_tokens, config.n_embd, device=device)
    
    num_token_blocks = len(row_indices)
    num_hidden_blocks = triton.cdiv(config.n_embd, config.block_size)
    
    print(f"Testing with {num_token_blocks} token blocks, {num_hidden_blocks} hidden blocks")
    
    dsd_kernel[(num_token_blocks, num_hidden_blocks)](
        block_sparse, w2, expert_output,
        row_indices_ptr=row_indices,
        weight_row_indices_ptr=weight_row_indices,
        stride_xm=block_sparse.stride()[0],
        stride_xk=block_sparse.stride()[1],
        stride_wk=w2.stride()[0],
        stride_wn=w2.stride()[1],
        stride_om=expert_output.stride()[0],
        stride_on=expert_output.stride()[1],
        d_ffn=d_ffn,
        hidden_size=config.n_embd,
        BLOCK_SIZE=config.block_size,
        BLOCK_K=config.block_k
    )
    
    # Compute expected outputs
    expected = torch.zeros_like(expert_output)
    # First block with expert 0
    expected[:32] = block_sparse[:32] @ w2[:d_ffn, :]
    # Second block with expert 1
    expected[32:64] = block_sparse[32:64] @ w2[d_ffn:2*d_ffn, :]
    
    max_diff = (expert_output - expected).abs().max().item()
    
    print(f"Multi-expert DSD test:")
    print(f"  Max difference: {max_diff:.6f}")
    
    # Check each block separately
    block1_diff = (expert_output[:32] - expected[:32]).abs().max().item()
    block2_diff = (expert_output[32:64] - expected[32:64]).abs().max().item()
    
    print(f"  Block 1 (expert 0) max diff: {block1_diff:.6f}")
    print(f"  Block 2 (expert 1) max diff: {block2_diff:.6f}")
    
    if max_diff < 1e-4:
        print("✅ Multi-expert test PASSED!")
    else:
        print("❌ Multi-expert test FAILED")

def run_all_tests():
    """Run all tests in sequence"""
    tests = [
        ("Weight Initialization Check", debug_weight_initialization),
        ("Simple DSD Kernel Test", test_dsd_kernel_simple),
        ("Multi-Expert DSD Test", test_dsd_kernel_with_different_experts),
        ("Full Pipeline Test", test_full_moe_pipeline),
        ("Full Pipeline Debug", test_full_moe_pipeline_debug),
    ]
    
    for name, test_func in tests:
        print("\n" + "="*60)
        print(f"{name}")
        print("="*60)
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()