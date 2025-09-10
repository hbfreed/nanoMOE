#!/usr/bin/env python3
"""
Simple MoE debug comparison - copy-pasted classes with minimal config.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from dataclasses import dataclass

# Import the SDD, DSD operations
from moe import SDD, DSD

# Copy-pasted MLP class from model.py
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

# Copy-pasted MoeMLPForLoop class from model.py
class MoeMLPForLoop(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok #top k
        self.norm_topk_prob = config.norm_topk_prob #bool, normalize the topk probabilities, or not?

        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)

        self.experts = nn.ModuleList([
            MLP(config) for _ in range(self.num_experts)            
        ])

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        router_logits = self.router(x_flat)
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float) #float32 here for stability
        router_weights, selected_experts = torch.topk(router_weights, self.num_experts_per_tok, dim=-1)

        if self.norm_topk_prob:
            router_weights /= router_weights.sum(dim=-1, keepdim=True) #normalize to 1 if we have normalization on
        router_weights = router_weights.to(x.dtype)

        output = torch.zeros_like(x_flat)

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts) #keep track which experts are active
        
        # n = batch * seq_len (number of tokens), k = num_experts_per_tok/ e = num_experts
        expert_mask = rearrange(expert_mask, 'n k e -> e k n')

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if len(top_x) > 0:
                current_state = x_flat[top_x]
                current_output = self.experts[expert_idx](current_state) * router_weights[top_x, idx, None]
                output.index_add_(0, top_x, current_output.to(x.dtype))
        
        # Skip aux losses for simplicity
        return output.view(batch_size, seq_len, hidden_dim), None, None

# Copy-pasted MoeMLP class from model.py  
class MoeMLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.n_embd = config.n_embd
        self.seq_len = config.n_ctx
        
        d_ffn = 4 * self.n_embd // self.num_experts_per_tok
        self.block_size = config.block_size  # Triton kernel block size, NOT sequence length!
        self.d_ffn = ((d_ffn + self.block_size - 1) // self.block_size) * self.block_size
        self.block_k = config.block_k
        
        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(self.n_embd, self.d_ffn * self.num_experts))
        self.w2 = nn.Parameter(torch.empty(self.d_ffn * self.num_experts, self.n_embd))
        
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)
    
    def _route_tokens(self, x_flat):
        """Route tokens to experts and compute weights."""
        router_logits = self.router(x_flat)
        router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        router_weights, selected_experts = torch.topk(router_weights, self.num_experts_per_tok, dim=-1)
        
        if self.norm_topk_prob:
            router_weights /= router_weights.sum(dim=-1, keepdim=True)
        
        router_weights = router_weights.to(x_flat.dtype)
        return router_weights, selected_experts, router_logits
    
    def _sort_by_expert(self, x_flat, router_weights, selected_experts):
        """Replicate tokens for each expert and sort by expert assignment."""
        x_rep = x_flat.repeat_interleave(self.num_experts_per_tok, dim=0)
        selected_experts_rep = selected_experts.reshape(-1)
        router_weights_rep = router_weights.reshape(-1, 1)
        
        expert_sort_indices = torch.argsort(selected_experts_rep, stable=True)
        x_sorted = x_rep[expert_sort_indices]
        selected_experts_sorted = selected_experts_rep[expert_sort_indices]
        router_weights_sorted = router_weights_rep[expert_sort_indices]
        
        inv_expert_sort_indices = torch.empty_like(expert_sort_indices)
        inv_expert_sort_indices[expert_sort_indices] = torch.arange(
            expert_sort_indices.numel(), device=x_flat.device
        )
        
        return x_sorted, selected_experts_sorted, router_weights_sorted, inv_expert_sort_indices
    
    def _pad_to_blocks(self, x_sorted, selected_experts_sorted):
        """Pad each expert's tokens to multiples of block_size and track unpadding indices."""
        device = x_sorted.device
        n = x_sorted.shape[0]
        d = x_sorted.shape[-1]
        e = self.num_experts
        b = self.block_size  # Triton token-block size (NOT seq len)

        # Upper-bound capacity in *blocks* based only on (n,e,b) → compile-safe
        m = min(n, e)
        max_blocks = m + (n - m) // b
        capacity_tokens = max_blocks * b

        # Per-expert counts via scatter_add (compile-safe; avoids bincount)
        counts = torch.zeros(e, dtype=torch.long, device=device)
        ones = torch.ones_like(selected_experts_sorted, dtype=torch.long)
        counts.scatter_add_(0, selected_experts_sorted, ones)

        # Round each expert up to a multiple of b
        tokens_per_expert_padded = ((counts + b - 1) // b) * b

        # Exclusive-prefix sums (orig vs padded) for placement
        off_orig = F.pad(counts.cumsum(0), (1, 0))              # [e+1]
        off_pad  = F.pad(tokens_per_expert_padded.cumsum(0), (1, 0))  # [e+1]

        # Allocate fixed capacity once; the tail is never indexed
        x_padded = x_sorted.new_zeros((capacity_tokens, d))

        # Map each sorted token to its padded position
        token_idx = torch.arange(n, device=device)
        idx_within_expert = token_idx - off_orig[selected_experts_sorted]
        unpad_indices = idx_within_expert + off_pad[selected_experts_sorted]

        # Scatter the actual tokens into their padded slots
        x_padded[unpad_indices] = x_sorted

        # Return exactly what you wanted
        return x_padded, tokens_per_expert_padded, unpad_indices

    def _create_sparse_indices(self, tokens_per_expert_padded):
        """Create indices using scatter operations to avoid dynamic shapes."""
        device = tokens_per_expert_padded.device
        print(f"\n--- _create_sparse_indices Debug ---")
        print(f"tokens_per_expert_padded: {tokens_per_expert_padded}")
        
        num_token_blocks_per_expert = tokens_per_expert_padded // self.block_size
        print(f"num_token_blocks_per_expert: {num_token_blocks_per_expert}")
        print(f"self.d_ffn: {self.d_ffn}, self.block_size: {self.block_size}")
        
        num_ffn_blocks = (self.d_ffn + self.block_size - 1) // self.block_size
        print(f"num_ffn_blocks: {num_ffn_blocks}")
        
        blocks_per_expert = num_token_blocks_per_expert * num_ffn_blocks
        print(f"blocks_per_expert: {blocks_per_expert}")
        
        # Calculate actual total blocks
        total_blocks = blocks_per_expert.sum()
        print(f"total_blocks: {total_blocks}")
        
        max_token_blocks_per_expert = (self.seq_len * self.num_experts_per_tok) // self.block_size
        # Ensure max_token_blocks_per_expert is at least as large as actual blocks
        max_token_blocks_per_expert = max(max_token_blocks_per_expert, num_token_blocks_per_expert.max().item())
        print(f"max_token_blocks_per_expert: {max_token_blocks_per_expert} (seq_len={self.seq_len}, num_experts_per_tok={self.num_experts_per_tok})")
        
        max_blocks = self.num_experts * max_token_blocks_per_expert * num_ffn_blocks
        print(f"max_blocks: {max_blocks} (num_experts={self.num_experts})")
        
        # Create indices for fixed size
        indices = torch.arange(max_blocks, device=device)
        print(f"indices shape: {indices.shape}, sample: {indices[:8] if len(indices) > 0 else 'empty'}")
        
        # Use searchsorted for expert assignment (compile-friendly!)
        cumsum = blocks_per_expert.cumsum(0)
        print(f"cumsum: {cumsum}")
        
        expert_ids = torch.searchsorted(cumsum, indices, right=True)
        print(f"expert_ids shape: {expert_ids.shape}, sample: {expert_ids[:8] if len(expert_ids) > 0 else 'empty'}")
        
        # Clamp expert_ids to valid range to avoid out of bounds
        expert_ids = torch.clamp(expert_ids, max=self.num_experts - 1)
        
        # Compute within-expert indices
        cumsum_padded = F.pad(cumsum[:-1], (1, 0))
        print(f"cumsum_padded: {cumsum_padded}")
        
        within_expert_idx = indices - cumsum_padded[expert_ids]
        print(f"within_expert_idx sample: {within_expert_idx[:8] if len(within_expert_idx) > 0 else 'empty'}")
        
        # Compute final indices
        token_block_offset = F.pad(num_token_blocks_per_expert.cumsum(0)[:-1], (1, 0))
        print(f"token_block_offset: {token_block_offset}")
        
        row_indices = token_block_offset[expert_ids] + (within_expert_idx // num_ffn_blocks)
        weight_col_indices = expert_ids * num_ffn_blocks + (within_expert_idx % num_ffn_blocks)
        output_col_indices = within_expert_idx % num_ffn_blocks
        
        # Set invalid indices to 0 (they'll be masked in the kernel)
        valid_mask = indices < total_blocks
        print(f"valid_mask sum: {valid_mask.sum()} out of {len(valid_mask)}")
        
        row_indices = torch.where(valid_mask, row_indices, torch.zeros_like(row_indices))
        weight_col_indices = torch.where(valid_mask, weight_col_indices, torch.zeros_like(weight_col_indices))
        output_col_indices = torch.where(valid_mask, output_col_indices, torch.zeros_like(output_col_indices))
        
        # Take only the valid indices
        row_indices = row_indices[valid_mask]
        weight_col_indices = weight_col_indices[valid_mask]  
        output_col_indices = output_col_indices[valid_mask]
        
        print(f"Final indices lengths: row={len(row_indices)}, weight={len(weight_col_indices)}, output={len(output_col_indices)}")
        
        return row_indices.int(), weight_col_indices.int(), output_col_indices.int()
    
    @torch.compiler.disable #sadly have to disable because of triton- TODO: fix this!
    def forward(self, x):
        print(f"\n--- MoeMLP Debug Forward ---")
        batch_size, seq_len, n_embd = x.shape
        print(f"Input shape: {x.shape}")
        x_flat = rearrange(x, 'batch seq hidden -> (batch seq) hidden')
        print(f"x_flat shape: {x_flat.shape}")
        
        router_weights, selected_experts, router_logits = self._route_tokens(x_flat)
        print(f"router_weights: {router_weights}")
        print(f"selected_experts: {selected_experts}")
        
        x_sorted, selected_experts_sorted, router_weights_sorted, inv_indices = self._sort_by_expert(
            x_flat, router_weights, selected_experts
        )
        print(f"x_sorted shape: {x_sorted.shape}")
        print(f"selected_experts_sorted: {selected_experts_sorted}")
        print(f"router_weights_sorted: {router_weights_sorted.flatten()}")
        
        x_padded, tokens_per_expert_padded, unpad_indices = self._pad_to_blocks(
            x_sorted, selected_experts_sorted
        )
        print(f"x_padded shape: {x_padded.shape}")
        print(f"tokens_per_expert_padded: {tokens_per_expert_padded}")
        print(f"unpad_indices: {unpad_indices}")
        
        row_indices, weight_col_indices, output_col_indices = self._create_sparse_indices(tokens_per_expert_padded)
        print(f"row_indices: {row_indices}")
        print(f"weight_col_indices: {weight_col_indices}")
        print(f"output_col_indices: {output_col_indices}")
        print(f"len(row_indices): {len(row_indices)}")
        
        if len(row_indices) > 0:
            # Compute num_ffn_blocks to pass to SDD
            num_ffn_blocks = (self.d_ffn + self.block_size - 1) // self.block_size
            print(f"num_ffn_blocks: {num_ffn_blocks}")
            
            # total_padded_tokens already computed above as a tensor
            block_sparse = SDD.apply(
                x_padded, self.w1, 
                row_indices, weight_col_indices, output_col_indices,
                self.block_size, num_ffn_blocks
            )
            print(f"block_sparse shape: {block_sparse.shape}")
            print(f"block_sparse sample: {block_sparse[0, :4]}")
            
            block_sparse = F.gelu(block_sparse)
            expert_output = DSD.apply(
                block_sparse, self.w2,
                row_indices, weight_col_indices, output_col_indices,
                self.block_size
            )
            print(f"expert_output shape: {expert_output.shape}")
            print(f"expert_output sample: {expert_output[0, :4]}")
            
            # Simply use the unpadding indices we computed during padding!
            output_unpadded = expert_output[unpad_indices]
            print(f"output_unpadded shape: {output_unpadded.shape}")
            print(f"output_unpadded sample: {output_unpadded[0, :4]}")
            
            # Apply router weights
            output_weighted = output_unpadded * router_weights_sorted
            print(f"output_weighted shape: {output_weighted.shape}")
            print(f"output_weighted sample: {output_weighted[0, :4]}")
            
            # Unpermute back to original token order
            output = output_weighted[inv_indices]
            print(f"output (after unpermute) shape: {output.shape}")
            print(f"output (after unpermute) sample: {output[0, :4]}")
            
            # Combine outputs from multiple experts per token using scatter_add
            # Since each token was sent to num_experts_per_tok experts,
            # we need to sum their weighted contributions
            num_tokens = batch_size * seq_len
            print(f"num_tokens: {num_tokens}")
            
            # Map each duplicated position (now in x_rep/original duplicate order)
            # to its original token index (0..num_tokens-1)
            original_token_indices = (
                torch.arange(num_tokens * self.num_experts_per_tok, device=output.device)
                // self.num_experts_per_tok
            )
            print(f"original_token_indices: {original_token_indices}")
            
            # Use scatter_add to efficiently combine all expert outputs at once
            combined_output = torch.zeros((num_tokens, self.n_embd), 
                                         dtype=output.dtype, device=output.device)
            print(f"combined_output (before scatter_add): {combined_output[0, :4]}")
            combined_output.scatter_add_(
                0,
                original_token_indices.unsqueeze(-1).expand(-1, self.n_embd),
                output
            )
            print(f"combined_output (after scatter_add): {combined_output[0, :4]}")
            output = combined_output
            
            # Reshape back to original batch dimensions
            output = rearrange(output, '(batch seq) hidden -> batch seq hidden', 
                             batch=batch_size, seq=seq_len)
            print(f"final output shape: {output.shape}")
            print(f"final output sample: {output[0, 0, :4]}")
        else:
            print("No tokens to process - returning zeros")
            # No tokens to process - return zeros
            output = torch.zeros((batch_size, seq_len, self.n_embd), 
                                dtype=x.dtype, device=x.device)
        
        # Skip aux losses for simplicity
        return output, None, None

@dataclass 
class SimpleConfig:
    n_embd: int = 32        # Keep minimum for Triton kernels
    n_ctx: int = 4          # Small sequence length  
    num_experts: int = 1    # Just 1 expert as you requested
    num_experts_per_tok: int = 1
    norm_topk_prob: bool = True
    block_size: int = 16    # Minimum block size for Triton
    block_k: int = 16
    bias: bool = False
    dropout: float = 0.0

def main():
    print("=== Simple MoE Comparison ===")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create simple config
    config = SimpleConfig()
    print(f"Config: {config}")
    
    # Create small input tensor
    batch_size, seq_len, n_embd = 1, 4, 32
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(batch_size, seq_len, n_embd) * 0.1
    x = x.to(device)
    print(f"\nInput tensor shape: {x.shape}")
    
    # Create both models with same initialization
    torch.manual_seed(42)
    model_forloop = MoeMLPForLoop(config).to(device)
    
    torch.manual_seed(42) 
    model_moe = MoeMLP(config).to(device)
    
    print(f"\n{'='*40}")
    print("Testing MoeMLPForLoop:")
    with torch.no_grad():
        output_forloop, _, _ = model_forloop(x)
        print(f"MoeMLPForLoop output shape: {output_forloop.shape}")
        print(f"Sample output: {output_forloop[0, 0, :4]}")
    
    print(f"\n{'='*40}")
    print("Testing MoeMLP:")
    with torch.no_grad():
        try:
            output_moe, _, _ = model_moe(x) 
            print(f"MoeMLP output shape: {output_moe.shape}")
            print(f"Sample output: {output_moe[0, 0, :4]}")
        except Exception as e:
            print(f"MoeMLP failed with error: {e}")
            import traceback
            traceback.print_exc()
            output_moe = torch.zeros_like(x)
    
    print(f"\n{'='*40}")
    print("=== COMPARISON ===")
    if output_forloop.shape == output_moe.shape:
        diff = torch.abs(output_forloop - output_moe)
        max_diff = diff.max().item()
        print(f"Max absolute difference: {max_diff}")
        
        if max_diff < 1e-6:
            print("✅ Outputs match within tolerance!")
        else:
            print("❌ Outputs differ significantly!")
            print(f"MoeMLPForLoop sample: {output_forloop[0, 0, :8]}")
            print(f"MoeMLP sample:        {output_moe[0, 0, :8]}")
            print(f"Difference sample:    {diff[0, 0, :8]}")
    else:
        print(f"❌ Output shapes don't match: {output_forloop.shape} vs {output_moe.shape}")

if __name__ == "__main__":
    main()