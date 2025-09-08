#!/usr/bin/env python3
"""
Script to compare the outputs of MoeMLP (Triton kernel) and MoeMlpForLoop (Python loop)
to identify discrepancies in the Triton kernel implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
import sys
import os

# Add parent directory to path to import the model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import MoeMLP, MoeMLPForLoop

@dataclass
class TestConfig:
    """Configuration for testing the MoE models."""
    n_embd: int = 384
    n_ctx: int = 256  # sequence length
    num_experts: int = 8
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True
    bias: bool = False
    # Triton block config (match MoeMLP defaults)
    block_size: int = 16
    block_k: int = 32
    
    # For compatibility with MLP initialization
    dropout: float = 0.0

def create_test_input(batch_size: int, seq_len: int, hidden_dim: int, device='cuda', seed=42):
    """Create reproducible test input."""
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)

def compare_outputs(out1: torch.Tensor, out2: torch.Tensor, name: str, rtol=1e-4, atol=1e-5):
    """Compare two tensors and report differences."""
    print(f"\n{'='*60}")
    print(f"Comparing: {name}")
    print(f"{'='*60}")
    
    # Basic shape check
    if out1.shape != out2.shape:
        print(f"❌ Shape mismatch: {out1.shape} vs {out2.shape}")
        return False
    
    print(f"✓ Shape match: {out1.shape}")
    
    # Compute differences
    diff = (out1 - out2).abs()
    rel_diff = diff / (out2.abs() + 1e-8)
    
    # Statistics
    print(f"\nAbsolute difference stats:")
    print(f"  Max: {diff.max().item():.2e}")
    print(f"  Mean: {diff.mean().item():.2e}")
    print(f"  Std: {diff.std().item():.2e}")
    
    print(f"\nRelative difference stats:")
    print(f"  Max: {rel_diff.max().item():.2e}")
    print(f"  Mean: {rel_diff.mean().item():.2e}")
    print(f"  Std: {rel_diff.std().item():.2e}")
    
    # Check if outputs are close
    is_close = torch.allclose(out1, out2, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"✓ Outputs match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"❌ Outputs differ beyond tolerance (rtol={rtol}, atol={atol})")
        
        # Find worst mismatches
        n_worst = min(5, diff.numel())
        diff_flat = diff.flatten()
        worst_indices = torch.topk(diff_flat, n_worst).indices
        
        print(f"\nTop {n_worst} worst absolute differences:")
        for i, idx in enumerate(worst_indices):
            unraveled = np.unravel_index(idx.cpu().item(), out1.shape)
            val1 = out1[unraveled].item()
            val2 = out2[unraveled].item()
            print(f"  {i+1}. Index {unraveled}: {val1:.6f} vs {val2:.6f} (diff: {abs(val1-val2):.2e})")
    
    return is_close

def test_forward_pass(config: TestConfig, batch_size=2, seq_len=128, device='cuda'):
    """Test forward pass of both models."""
    print("\n" + "="*80)
    print("TESTING FORWARD PASS")
    print("="*80)
    
    # Create models
    print("\nInitializing models...")
    moe_triton = MoeMLP(config).to(device)
    moe_loop = MoeMLPForLoop(config).to(device)
    
    # Copy weights from triton model to loop model to ensure same initialization
    print("Synchronizing weights...")
    with torch.no_grad():
        # Copy router weights
        moe_loop.router.weight.copy_(moe_triton.router.weight)
        
        # Copy expert weights from Triton's combined weights to loop's individual experts
        d_ffn_per_expert = moe_triton.d_ffn
        for i in range(config.num_experts):
            # Extract this expert's weights from the combined tensors
            w1_slice = moe_triton.w1[:, i*d_ffn_per_expert:(i+1)*d_ffn_per_expert]
            w2_slice = moe_triton.w2[i*d_ffn_per_expert:(i+1)*d_ffn_per_expert, :]
            
            # The loop model uses standard MLP with c_fc and c_proj
            # MLP structure: c_fc projects n_embd -> 4*n_embd, c_proj projects 4*n_embd -> n_embd
            # But we need to handle the dimension mismatch since MoeMLP uses d_ffn
            
            # Get the actual dimensions from the loop model's experts
            expert_mlp = moe_loop.experts[i]
            fc_out_dim = expert_mlp.c_fc.weight.shape[0]
            
            # Only copy the portions that fit
            copy_dim = min(w1_slice.shape[1], fc_out_dim)
            expert_mlp.c_fc.weight[:copy_dim, :].copy_(w1_slice.T[:copy_dim, :])
            expert_mlp.c_proj.weight[:, :copy_dim].copy_(w2_slice[:copy_dim, :].T)
    
    # Create test input
    print(f"\nCreating test input: batch={batch_size}, seq={seq_len}, hidden={config.n_embd}")
    x = create_test_input(batch_size, seq_len, config.n_embd, device=device)
    
    # Forward pass
    print("\nRunning forward passes...")
    with torch.no_grad():
        # Triton model
        out_triton, aux_triton, f_i_triton = moe_triton(x)
        
        # Loop model  
        out_loop, aux_loop, f_i_loop = moe_loop(x)
    
    # Compare main outputs
    outputs_match = compare_outputs(out_triton, out_loop, "Main Output")
    
    # Compare auxiliary losses
    print("\n" + "="*60)
    print("Auxiliary Losses Comparison")
    print("="*60)
    
    print(f"\nRouter Z-Loss:")
    print(f"  Triton: {aux_triton['router_z_loss'].item():.6f}")
    print(f"  Loop:   {aux_loop['router_z_loss'].item():.6f}")
    print(f"  Diff:   {abs(aux_triton['router_z_loss'].item() - aux_loop['router_z_loss'].item()):.2e}")
    
    print(f"\nLoad Balance Loss:")
    print(f"  Triton: {aux_triton['load_balance_loss'].item():.6f}")
    print(f"  Loop:   {aux_loop['load_balance_loss'].item():.6f}")
    print(f"  Diff:   {abs(aux_triton['load_balance_loss'].item() - aux_loop['load_balance_loss'].item()):.2e}")
    
    # Compare expert utilization
    f_i_match = compare_outputs(f_i_triton, f_i_loop, "Expert Utilization (f_i)")
    
    return outputs_match and f_i_match

def test_gradient_flow(config: TestConfig, batch_size=2, seq_len=64, device='cuda'):
    """Test backward pass and gradient computation."""
    print("\n" + "="*80)
    print("TESTING GRADIENT FLOW")
    print("="*80)
    
    # Create models
    print("\nInitializing models...")
    moe_triton = MoeMLP(config).to(device)
    moe_loop = MoeMLPForLoop(config).to(device)
    
    # Synchronize weights as before
    print("Synchronizing weights...")
    with torch.no_grad():
        moe_loop.router.weight.copy_(moe_triton.router.weight)
        
        d_ffn_per_expert = moe_triton.d_ffn
        for i in range(config.num_experts):
            w1_slice = moe_triton.w1[:, i*d_ffn_per_expert:(i+1)*d_ffn_per_expert]
            w2_slice = moe_triton.w2[i*d_ffn_per_expert:(i+1)*d_ffn_per_expert, :]
            
            expert_mlp = moe_loop.experts[i]
            fc_out_dim = expert_mlp.c_fc.weight.shape[0]
            
            copy_dim = min(w1_slice.shape[1], fc_out_dim)
            expert_mlp.c_fc.weight[:copy_dim, :].copy_(w1_slice.T[:copy_dim, :])
            expert_mlp.c_proj.weight[:, :copy_dim].copy_(w2_slice[:copy_dim, :].T)
    
    # Create test input
    print(f"\nCreating test input: batch={batch_size}, seq={seq_len}, hidden={config.n_embd}")
    x = create_test_input(batch_size, seq_len, config.n_embd, device=device)
    x_triton = x.clone().requires_grad_(True)
    x_loop = x.clone().requires_grad_(True)
    
    # Forward pass
    print("\nRunning forward passes with gradient tracking...")
    out_triton, aux_triton, _ = moe_triton(x_triton)
    out_loop, aux_loop, _ = moe_loop(x_loop)
    
    # Create dummy loss
    loss_triton = out_triton.mean() + 0.01 * aux_triton['load_balance_loss']
    loss_loop = out_loop.mean() + 0.01 * aux_loop['load_balance_loss']
    
    # Backward pass
    print("Running backward passes...")
    loss_triton.backward()
    loss_loop.backward()
    
    # Compare input gradients
    grad_match = compare_outputs(x_triton.grad, x_loop.grad, "Input Gradients")
    
    # Compare router gradients
    router_grad_match = compare_outputs(
        moe_triton.router.weight.grad,
        moe_loop.router.weight.grad,
        "Router Weight Gradients"
    )
    
    print("\n" + "="*60)
    print("Weight Gradient Statistics")
    print("="*60)
    
    # Check W1 gradients
    if moe_triton.w1.grad is not None:
        print(f"\nW1 gradients (Triton):")
        print(f"  Shape: {moe_triton.w1.grad.shape}")
        print(f"  Mean: {moe_triton.w1.grad.mean().item():.2e}")
        print(f"  Std:  {moe_triton.w1.grad.std().item():.2e}")
        print(f"  Max:  {moe_triton.w1.grad.abs().max().item():.2e}")
    
    # Check W2 gradients
    if moe_triton.w2.grad is not None:
        print(f"\nW2 gradients (Triton):")
        print(f"  Shape: {moe_triton.w2.grad.shape}")
        print(f"  Mean: {moe_triton.w2.grad.mean().item():.2e}")
        print(f"  Std:  {moe_triton.w2.grad.std().item():.2e}")
        print(f"  Max:  {moe_triton.w2.grad.abs().max().item():.2e}")
    
    return grad_match and router_grad_match

def test_edge_cases(config: TestConfig, device='cuda'):
    """Test edge cases and special scenarios."""
    print("\n" + "="*80)
    print("TESTING EDGE CASES")
    print("="*80)
    
    test_cases = [
        ("Single token", 1, 1),
        ("Single batch", 1, 64),
        ("Large batch", 8, 32),
        ("Non-power-of-2 seq", 2, 97),
    ]
    
    all_pass = True
    
    for name, batch_size, seq_len in test_cases:
        print(f"\n\nTest: {name} (batch={batch_size}, seq={seq_len})")
        print("-" * 40)
        
        try:
            moe_triton = MoeMLP(config).to(device)
            moe_loop = MoeMLPForLoop(config).to(device)
            
            # Sync weights
            with torch.no_grad():
                moe_loop.router.weight.copy_(moe_triton.router.weight)
                d_ffn_per_expert = moe_triton.d_ffn
                for i in range(config.num_experts):
                    w1_slice = moe_triton.w1[:, i*d_ffn_per_expert:(i+1)*d_ffn_per_expert]
                    w2_slice = moe_triton.w2[i*d_ffn_per_expert:(i+1)*d_ffn_per_expert, :]
                    expert_mlp = moe_loop.experts[i]
                    fc_out_dim = expert_mlp.c_fc.weight.shape[0]
                    copy_dim = min(w1_slice.shape[1], fc_out_dim)
                    expert_mlp.c_fc.weight[:copy_dim, :].copy_(w1_slice.T[:copy_dim, :])
                    expert_mlp.c_proj.weight[:, :copy_dim].copy_(w2_slice[:copy_dim, :].T)
            
            x = create_test_input(batch_size, seq_len, config.n_embd, device=device)
            
            with torch.no_grad():
                out_triton, _, _ = moe_triton(x)
                out_loop, _, _ = moe_loop(x)
            
            if torch.allclose(out_triton, out_loop, rtol=1e-3, atol=1e-4):
                print(f"✓ PASS")
            else:
                print(f"❌ FAIL - Max diff: {(out_triton - out_loop).abs().max().item():.2e}")
                all_pass = False
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            all_pass = False
    
    return all_pass

def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU (will be slow).")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Using device: {torch.cuda.get_device_name()}")
    
    # Create test configuration
    config = TestConfig()
    
    print("\n" + "="*80)
    print("MOE IMPLEMENTATION COMPARISON TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Hidden dimension: {config.n_embd}")
    print(f"  Sequence length: {config.n_ctx}")
    print(f"  Number of experts: {config.num_experts}")
    print(f"  Experts per token: {config.num_experts_per_tok}")
    print(f"  Normalize top-k: {config.norm_topk_prob}")
    
    # Run tests
    results = {}
    
    # Test forward pass
    results['forward'] = test_forward_pass(config, device=device)
    
    # Test gradient flow
    results['gradient'] = test_gradient_flow(config, device=device)
    
    # Test edge cases
    results['edge_cases'] = test_edge_cases(config, device=device)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.capitalize():15} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! The implementations match.")
    else:
        print("\n⚠️  Some tests failed. There are discrepancies between implementations.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
