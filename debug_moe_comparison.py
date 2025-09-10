#!/usr/bin/env python3
"""
Debug comparison between MoeMLP and MoeMLPForLoop implementations.
Uses small tensors for manual inspection to identify computation bugs.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from dataclasses import dataclass

# Import classes from model.py and moe.py 
from moe import SDD, DSD 

# Simple MLP class copied from model.py
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Debug version of MoeMLPForLoop with tensor inspection
class MoeMLPForLoopDebug(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)

        self.experts = nn.ModuleList([
            MLP(config) for _ in range(self.num_experts)            
        ])

    def forward(self, x):
        print(f"\n=== MoeMLPForLoop Debug ===")
        batch_size, seq_len, hidden_dim = x.shape
        print(f"Input shape: {x.shape}")
        
        x_flat = x.view(-1, hidden_dim)
        print(f"x_flat shape: {x_flat.shape}")

        router_logits = self.router(x_flat)
        print(f"router_logits shape: {router_logits.shape}")
        print(f"router_logits first few values: {router_logits[:2, :2]}")
        
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        router_weights, selected_experts = torch.topk(router_weights, self.num_experts_per_tok, dim=-1)
        print(f"selected_experts:\n{selected_experts}")

        if self.norm_topk_prob:
            router_weights /= router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(x.dtype)
        print(f"router_weights (final):\n{router_weights}")

        output = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = rearrange(expert_mask, 'n k e -> e k n')

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            print(f"\n--- Expert {expert_idx} processes tokens: {top_x} ---")

            if len(top_x) > 0:
                current_state = x_flat[top_x]
                current_output_raw = self.experts[expert_idx](current_state)
                weights_for_expert = router_weights[top_x, idx, None]
                current_output = current_output_raw * weights_for_expert
                output.index_add_(0, top_x, current_output.to(x.dtype))
                print(f"Expert {expert_idx} output sample: {current_output[0, :4]}")
        
        final_output = output.view(batch_size, seq_len, hidden_dim)
        print(f"Final output sample (first token, first 4 dims): {final_output[0, 0, :4]}")
        
        return final_output, None, None  # Simplified - skip aux losses

# Debug version of MoeMLP with tensor inspection  
class MoeMLPDebug(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.n_embd = config.n_embd
        self.seq_len = config.n_ctx
        
        d_ffn = 4 * self.n_embd // self.num_experts_per_tok
        self.block_size = config.block_size
        self.d_ffn = ((d_ffn + self.block_size - 1) // self.block_size) * self.block_size
        self.block_k = config.block_k
        
        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(self.n_embd, self.d_ffn * self.num_experts))
        self.w2 = nn.Parameter(torch.empty(self.d_ffn * self.num_experts, self.n_embd))
        
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)
    
    def _route_tokens(self, x_flat):
        print(f"\n=== MoeMLP Debug - Routing ===")
        print(f"Input to routing: {x_flat.shape}")
        
        router_logits = self.router(x_flat)
        print(f"router_logits first few values: {router_logits[:2, :2]}")
        
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        router_weights, selected_experts = torch.topk(router_weights, self.num_experts_per_tok, dim=-1)
        print(f"selected_experts:\n{selected_experts}")
        
        if self.norm_topk_prob:
            router_weights /= router_weights.sum(dim=-1, keepdim=True)
        
        router_weights = router_weights.to(x_flat.dtype)
        print(f"router_weights (final):\n{router_weights}")
        return router_weights, selected_experts, router_logits
    
    def _sort_by_expert(self, x_flat, router_weights, selected_experts):
        print(f"\n=== MoeMLP Debug - Sorting ===")
        x_rep = x_flat.repeat_interleave(self.num_experts_per_tok, dim=0)
        print(f"x_rep shape: {x_rep.shape}")
        
        selected_experts_rep = selected_experts.reshape(-1)
        router_weights_rep = router_weights.reshape(-1, 1)
        
        expert_sort_indices = torch.argsort(selected_experts_rep, stable=True)
        print(f"expert_sort_indices: {expert_sort_indices}")
        
        x_sorted = x_rep[expert_sort_indices]
        selected_experts_sorted = selected_experts_rep[expert_sort_indices]
        router_weights_sorted = router_weights_rep[expert_sort_indices]
        print(f"selected_experts_sorted: {selected_experts_sorted}")
        print(f"router_weights_sorted: {router_weights_sorted.flatten()}")
        
        inv_expert_sort_indices = torch.empty_like(expert_sort_indices)
        inv_expert_sort_indices[expert_sort_indices] = torch.arange(
            expert_sort_indices.numel(), device=x_flat.device
        )
        
        return x_sorted, selected_experts_sorted, router_weights_sorted, inv_expert_sort_indices
    
    def _pad_to_blocks(self, x_sorted, selected_experts_sorted):
        print(f"\n=== MoeMLP Debug - Padding ===")
        device = x_sorted.device
        n = x_sorted.shape[0]
        d = x_sorted.shape[-1]
        e = self.num_experts
        b = self.block_size

        # Upper-bound capacity in *blocks* based only on (n,e,b) → compile-safe
        m = min(n, e)
        max_blocks = m + (n - m) // b
        capacity_tokens = max_blocks * b
        print(f"capacity_tokens: {capacity_tokens}")

        # Per-expert counts via scatter_add (compile-safe; avoids bincount)
        counts = torch.zeros(e, dtype=torch.long, device=device)
        ones = torch.ones_like(selected_experts_sorted, dtype=torch.long)
        counts.scatter_add_(0, selected_experts_sorted, ones)
        print(f"per-expert counts:\n{counts}")

        # Round each expert up to a multiple of b
        tokens_per_expert_padded = ((counts + b - 1) // b) * b
        print(f"tokens_per_expert_padded:\n{tokens_per_expert_padded}")

        # Exclusive-prefix sums (orig vs padded) for placement
        off_orig = F.pad(counts.cumsum(0), (1, 0))              # [e+1]
        off_pad  = F.pad(tokens_per_expert_padded.cumsum(0), (1, 0))  # [e+1]
        print(f"off_orig: {off_orig}")
        print(f"off_pad: {off_pad}")

        # Allocate fixed capacity once; the tail is never indexed
        x_padded = x_sorted.new_zeros((capacity_tokens, d))

        # Map each sorted token to its padded position
        token_idx = torch.arange(n, device=device)
        idx_within_expert = token_idx - off_orig[selected_experts_sorted]
        unpad_indices = idx_within_expert + off_pad[selected_experts_sorted]
        print(f"unpad_indices:\n{unpad_indices}")

        # Scatter the actual tokens into their padded slots
        x_padded[unpad_indices] = x_sorted
        print(f"x_padded shape: {x_padded.shape}")
        print(f"x_padded:\n{x_padded}")

        return x_padded, tokens_per_expert_padded, unpad_indices

    @torch.compiler.disable
    def forward(self, x):
        print(f"\n=== MoeMLP Debug - Forward ===")
        batch_size, seq_len, n_embd = x.shape
        print(f"Input shape: {x.shape}")
        print(f"Input:\n{x}")
        
        x_flat = rearrange(x, 'batch seq hidden -> (batch seq) hidden')
        print(f"x_flat shape: {x_flat.shape}")
        
        router_weights, selected_experts, router_logits = self._route_tokens(x_flat)
        
        x_sorted, selected_experts_sorted, router_weights_sorted, inv_indices = self._sort_by_expert(
            x_flat, router_weights, selected_experts
        )
        
        x_padded, tokens_per_expert_padded, unpad_indices = self._pad_to_blocks(
            x_sorted, selected_experts_sorted
        )
        
        print(f"\n=== MoeMLP Debug - Computing Sparse Indices ===")
        # Simplified sparse indices computation for debugging
        num_ffn_blocks = (self.d_ffn + self.block_size - 1) // self.block_size
        print(f"num_ffn_blocks: {num_ffn_blocks}")
        
        # For debugging, create dummy indices that match the expected pattern
        num_blocks = tokens_per_expert_padded.sum() // self.block_size * num_ffn_blocks
        print(f"num_blocks: {num_blocks}")
        
        if num_blocks > 0:
            # Create simple indices for the first few blocks
            row_indices = torch.arange(min(4, num_blocks), device=x.device, dtype=torch.int)
            weight_col_indices = torch.arange(min(4, num_blocks), device=x.device, dtype=torch.int)
            output_col_indices = torch.arange(min(4, num_blocks), device=x.device, dtype=torch.int)
            
            print(f"row_indices: {row_indices}")
            print(f"weight_col_indices: {weight_col_indices}")
            print(f"output_col_indices: {output_col_indices}")
            
            print(f"\n=== MoeMLP Debug - SDD/DSD Operations ===")
            print(f"x_padded shape: {x_padded.shape}, w1 shape: {self.w1.shape}")
            
            # Apply SDD 
            block_sparse = SDD.apply(
                x_padded, self.w1, 
                row_indices, weight_col_indices, output_col_indices,
                self.block_size, num_ffn_blocks
            )
            print(f"block_sparse shape after SDD: {block_sparse.shape}")
            print(f"block_sparse sample: {block_sparse[0, :4]}")
            
            block_sparse = F.gelu(block_sparse)
            print(f"block_sparse after GELU sample: {block_sparse[0, :4]}")
            
            expert_output = DSD.apply(
                block_sparse, self.w2,
                row_indices, weight_col_indices, output_col_indices,
                self.block_size
            )
            print(f"expert_output shape after DSD: {expert_output.shape}")
            print(f"expert_output sample: {expert_output[0, :4]}")
            
            # For now, return simplified output to check if kernels work
            output = torch.zeros((batch_size, seq_len, self.n_embd), 
                               dtype=x.dtype, device=x.device)
            print(f"Final output (simplified): shape {output.shape}")
        else:
            output = torch.zeros((batch_size, seq_len, self.n_embd), 
                               dtype=x.dtype, device=x.device)
            print(f"No blocks to process, returning zeros:\n{output}")
        
        return output, None, None  # Simplified

@dataclass 
class DebugConfig:
    n_embd: int = 32  # Minimum 32 for Triton kernels
    n_ctx: int = 4    # Small sequence length  
    num_experts: int = 2
    num_experts_per_tok: int = 1
    norm_topk_prob: bool = True
    block_size: int = 16  # Minimum block size for Triton
    block_k: int = 16
    bias: bool = False
    dropout: float = 0.0

def main():
    print("=== MoE Comparison Debug ===")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create small config for debugging
    config = DebugConfig()
    print(f"Config: {config}")
    
    # Create small input tensor for debugging
    batch_size, seq_len, n_embd = 1, 4, 32
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(batch_size, seq_len, n_embd) * 0.1  # Small values
    x = x.to(device)  # Move to GPU if available
    print(f"\nInput tensor shape: {x.shape}")
    print(f"Input tensor on {device}:\n{x}")
    
    # Create both models with same initialization
    torch.manual_seed(42)
    model_forloop = MoeMLPForLoopDebug(config).to(device)
    
    torch.manual_seed(42) 
    model_moe = MoeMLPDebug(config).to(device)
    
    print(f"\n{'='*50}")
    print("Testing MoeMLPForLoop:")
    with torch.no_grad():
        output_forloop, _, _ = model_forloop(x)
    
    print(f"\n{'='*50}")
    print("Testing MoeMLP:")
    with torch.no_grad():
        try:
            output_moe, _, _ = model_moe(x) 
        except Exception as e:
            print(f"MoeMLP failed with error: {e}")
            import traceback
            traceback.print_exc()
            output_moe = torch.zeros_like(x)
    
    print(f"\n{'='*50}")
    print("=== COMPARISON ===")
    print(f"MoeMLPForLoop output:\n{output_forloop}")
    print(f"MoeMLP output:\n{output_moe}")
    
    if output_forloop.shape == output_moe.shape:
        diff = torch.abs(output_forloop - output_moe)
        max_diff = diff.max().item()
        print(f"Max absolute difference: {max_diff}")
        print(f"Difference tensor:\n{diff}")
        
        if max_diff < 1e-6:
            print("✅ Outputs match within tolerance!")
        else:
            print("❌ Outputs differ significantly!")
    else:
        print(f"❌ Output shapes don't match: {output_forloop.shape} vs {output_moe.shape}")

if __name__ == "__main__":
    main()