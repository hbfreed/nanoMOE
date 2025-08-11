import torch
import torch.nn as nn
from model import MoeMLPForLoop, MLP, MoeMLP
from moe import sdd_kernel


def test_sdd_kernel_vs_baseline():
    """Compare sparse Triton kernel with the dense for-loop baseline."""
    
    # Test config
    class TestConfig:
        n_embd = 32
        num_experts = 4
        num_experts_per_tok = 2
        norm_topk_prob = True
        bias = False
        dropout = 0.0
    
    config = TestConfig()
    
    # Create both models
    baseline = MoeMLPForLoop(config).cuda()
    sparse_moe = MoeMLP(
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        n_embd=config.n_embd,
        norm_topk_prob=config.norm_topk_prob,
        block_size=16,  # Triton dot requires M,N,K >= 16
        block_k=16
    ).cuda()
    
    # Copy weights: baseline uses separate MLPs, sparse uses concatenated weights
    with torch.no_grad():
        # Copy router weights
        sparse_moe.router.weight.data = baseline.router.weight.data.clone()
        
        # Copy expert weights (concatenate them for sparse version)
        # Note: Linear layers store weights as [out_features, in_features]
        # c_fc weight is [4*n_embd, n_embd], we need to transpose to [n_embd, 4*n_embd]
        w1_chunks = []
        w2_chunks = []
        for i in range(config.num_experts):
            full_w1_t = baseline.experts[i].c_fc.weight.T  # [n_embd, 4*n_embd]
            # Align with sparse_moe.d_ffn per expert by slicing
            w1_chunks.append(full_w1_t[:, :sparse_moe.d_ffn])  # [n_embd, d_ffn]
            full_w2_t = baseline.experts[i].c_proj.weight.T  # [4*n_embd, n_embd]
            w2_chunks.append(full_w2_t[:sparse_moe.d_ffn, :])  # [d_ffn, n_embd]
        
        # Concatenate along the FFN dimension per expert
        sparse_moe.w1.data = torch.cat(w1_chunks, dim=1).contiguous()  # [n_embd, d_ffn*num_experts]
        sparse_moe.w2.data = torch.cat(w2_chunks, dim=0).contiguous()  # [d_ffn*num_experts, n_embd]
    
    # Test input
    torch.manual_seed(42)
    x = torch.randn(2, 4, config.n_embd).cuda()
    
    with torch.no_grad():
        # Flatten tokens
        x_flat = x.view(-1, config.n_embd)
        
        # Route tokens using the shared router
        router_weights, selected_experts = sparse_moe._route_tokens(x_flat)
        
        # Sort by expert
        x_sorted, selected_experts_sorted, router_weights_sorted, inv_indices = sparse_moe._sort_by_expert(
            x_flat, router_weights, selected_experts
        )
        
        # Pad to blocks
        x_padded, router_weights_padded, tokens_per_expert_padded, cumsum_padded = sparse_moe._pad_to_blocks(
            x_sorted, router_weights_sorted, selected_experts_sorted
        )
        
        # Create indices for SDD
        row_indices, col_indices = sparse_moe._create_sparse_indices(tokens_per_expert_padded)
        
        # Output tensor for SDD
        total_padded_tokens = cumsum_padded[-1].item()
        block_sparse = torch.zeros(
            (total_padded_tokens, sparse_moe.d_ffn * config.num_experts),
            dtype=x.dtype, device=x.device
        )
        
        # Run SDD kernel if there are any blocks
        if len(row_indices) > 0:
            sdd_kernel[len(row_indices),](
                x_padded, sparse_moe.w1, block_sparse,
                row_indices, col_indices,
                *x_padded.stride(),
                *sparse_moe.w1.stride(),
                *block_sparse.stride(),
                config.n_embd,
                BLOCK_SIZE=sparse_moe.block_size,
                BLOCK_K=sparse_moe.block_k,
            )
        
        # Reference dense compute on padded inputs per expert (no activation/weights)
        block_sparse_ref = torch.zeros_like(block_sparse)
        tokens_per_expert = torch.bincount(
            selected_experts_sorted, minlength=config.num_experts
        )
        for exp_id in range(config.num_experts):
            n_tokens = tokens_per_expert[exp_id].item()
            if n_tokens == 0:
                continue
            dst_start = cumsum_padded[exp_id].item()
            dst_end = cumsum_padded[exp_id + 1].item()
            # Use all padded rows for this expert
            x_pad_slice = x_padded[dst_start:dst_end]
            col_start = exp_id * sparse_moe.d_ffn
            col_end = (exp_id + 1) * sparse_moe.d_ffn
            w_slice = sparse_moe.w1[:, col_start:col_end]
            block_sparse_ref[dst_start:dst_end, col_start:col_end] = x_pad_slice @ w_slice
        
        # Check kernel vs reference padded output first
        torch.testing.assert_close(block_sparse, block_sparse_ref, rtol=1e-3, atol=1e-3)
        
        # Extract the non-padded rows per expert and concatenate in expert order
        per_expert_slices = []
        for exp_id in range(config.num_experts):
            n_tokens = tokens_per_expert[exp_id].item()
            if n_tokens == 0:
                continue
            dst_start = cumsum_padded[exp_id].item()
            per_expert_slices.append(block_sparse[dst_start:dst_start + n_tokens])
        
        if len(per_expert_slices) > 0:
            block_sparse_sorted = torch.cat(per_expert_slices, dim=0)
        else:
            block_sparse_sorted = torch.empty(0, sparse_moe.d_ffn * config.num_experts, device=x.device, dtype=x.dtype)
        
        # Unsort back to the original repeated token order
        repeated_unsorted = block_sparse_sorted[inv_indices]
        
        # Aggregate repeats per token back to [num_tokens, d_ffn * num_experts]
        num_tokens = x_flat.shape[0]
        k = config.num_experts_per_tok
        repeated_unsorted = repeated_unsorted.view(num_tokens, k, -1)
        sparse_first_layer = repeated_unsorted.sum(dim=1)
        
        # Build baseline first layer outputs using the same expert slices used to build w1
        baseline_first_layer = torch.zeros_like(sparse_first_layer)
        for token_idx in range(num_tokens):
            for kk in range(k):
                expert_id = selected_experts[token_idx, kk].item()
                start_col = expert_id * sparse_moe.d_ffn
                end_col = (expert_id + 1) * sparse_moe.d_ffn
                weight_slice = baseline.experts[expert_id].c_fc.weight.T[:, :sparse_moe.d_ffn]
                expert_out = x_flat[token_idx] @ weight_slice
                baseline_first_layer[token_idx, start_col:end_col] = expert_out
        
        # Compare final reduced outputs
        torch.testing.assert_close(
            sparse_first_layer, baseline_first_layer, rtol=1e-4, atol=1e-4
        )
    
    print("\n✓ SDD kernel first-layer output matches baseline slices!")


def test_full_forward_pass():
    """Test complete MoE forward pass with both FFN layers."""
    # TODO: implement once DSD kernel is ready
    pass


def test_gradient_flow():
    """Verify gradients flow correctly through sparse ops."""
    # TODO: implement for training
    pass


if __name__ == "__main__":
    print("Testing SDD kernel...")
    test_sdd_kernel_vs_baseline()
    print("\n✓ All tests passed!")