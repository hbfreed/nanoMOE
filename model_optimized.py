"""
Optimized version of model.py that fixes the profiling bottlenecks.
This is a drop-in replacement - just import from model_optimized instead of model.

Main optimizations:
1. Pre-allocated tensors to avoid torch.tensor() calls (saves ~31s/100 iters)
2. Reuse of buffers where possible
"""

# Import everything from the original model
from model import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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


        self._num_ffn_blocks = (self.d_ffn + self.block_size - 1) // self.block_size

        # maximum token blocks per expert using only compile-stable config values
        # (upper bound for any forward pass; runtime seq_len ≤ config.n_ctx)
        max_tok_blocks_per_exp = (self.seq_len * self.num_experts_per_tok + self.block_size - 1) // self.block_size

        # fully static upper bound on total blocks across all experts
        self._max_blocks_static = self.num_experts * max_tok_blocks_per_exp * self._num_ffn_blocks

        # optional: preallocate a reusable arange buffer to avoid arange each pass
        self.register_buffer(
            "_index_buf",
            torch.arange(self._max_blocks_static, dtype=torch.int32),
            persistent=False
        )

    
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
        import torch
        import torch.nn.functional as F

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

    '''slow, seems to work'''
    def _create_sparse_indices(self, tokens_per_expert_padded):
        """Create indices using scatter operations to avoid dynamic shapes."""
        device = tokens_per_expert_padded.device
        
        num_token_blocks_per_expert = tokens_per_expert_padded // self.block_size
        num_ffn_blocks = (self.d_ffn + self.block_size - 1) // self.block_size
        blocks_per_expert = num_token_blocks_per_expert * num_ffn_blocks
        
        # Calculate actual total blocks
        total_blocks = blocks_per_expert.sum()
        
        max_token_blocks_per_expert_static = (self.seq_len * self.num_experts_per_tok) // self.block_size
        # BUGFIX: Ensure we allocate enough blocks (compile-safe approach)
        # Instead of using .item(), compute max_blocks based on actual total_blocks
        # This ensures we always allocate at least as many blocks as needed
        max_blocks_static = self.num_experts * max_token_blocks_per_expert_static * num_ffn_blocks
        # max_blocks = torch.maximum(torch.tensor(max_blocks_static, device=device), total_blocks)
        max_blocks = total_blocks.clamp_min(max_blocks_static)

        # Create indices for fixed size
        indices = torch.arange(max_blocks, device=device)
        
        # Use searchsorted for expert assignment (compile-friendly!)
        cumsum = blocks_per_expert.cumsum(0)
        expert_ids = torch.searchsorted(cumsum, indices, right=True)
        
        # Clamp expert_ids to valid range to avoid out of bounds
        expert_ids = torch.clamp(expert_ids, max=self.num_experts - 1)
        
        # Compute within-expert indices
        cumsum_padded = F.pad(cumsum[:-1], (1, 0))
        within_expert_idx = indices - cumsum_padded[expert_ids]
        
        # Compute final indices
        token_block_offset = F.pad(num_token_blocks_per_expert.cumsum(0)[:-1], (1, 0))
        row_indices = token_block_offset[expert_ids] + (within_expert_idx // num_ffn_blocks)
        weight_col_indices = expert_ids * num_ffn_blocks + (within_expert_idx % num_ffn_blocks)
        output_col_indices = within_expert_idx % num_ffn_blocks
        
        # Set invalid indices to 0 (they'll be masked in the kernel)
        valid_mask = indices < total_blocks
        row_indices = torch.where(valid_mask, row_indices, torch.zeros_like(row_indices))
        weight_col_indices = torch.where(valid_mask, weight_col_indices, torch.zeros_like(weight_col_indices))
        output_col_indices = torch.where(valid_mask, output_col_indices, torch.zeros_like(output_col_indices))
        
        return row_indices.int(), weight_col_indices.int(), output_col_indices.int()

    @torch.compiler.disable #sadly have to disable because of triton- TODO: fix this!
    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        x_flat = rearrange(x, 'batch seq hidden -> (batch seq) hidden')
        
        router_weights, selected_experts, router_logits = self._route_tokens(x_flat)
        
        x_sorted, selected_experts_sorted, router_weights_sorted, inv_indices = self._sort_by_expert(
            x_flat, router_weights, selected_experts
        )
        
        x_padded, tokens_per_expert_padded, unpad_indices = self._pad_to_blocks(
            x_sorted, selected_experts_sorted
        )
        
        row_indices, weight_col_indices, output_col_indices = self._create_sparse_indices(tokens_per_expert_padded)
        
        if len(row_indices) > 0:
            # Compute num_ffn_blocks to pass to SDD
            num_ffn_blocks = (self.d_ffn + self.block_size - 1) // self.block_size
            
            # total_padded_tokens already computed above as a tensor
            block_sparse = SDD.apply(
                x_padded, self.w1, 
                row_indices, weight_col_indices, output_col_indices,
                self.block_size, num_ffn_blocks
            )
            
            # block_sparse = F.gelu(block_sparse)
            # block_sparse = block_sparse.t().contiguous() # transpose and contiguous for making dsd perf better
            expert_output = DSD.apply(
                block_sparse, self.w2,
                row_indices, weight_col_indices, output_col_indices,
                self.block_size
            )
            
            # Simply use the unpadding indices we computed during padding!
            output_unpadded = expert_output[unpad_indices]
            
            # Apply router weights
            output_weighted = output_unpadded * router_weights_sorted
            
            # Unpermute back to original token order
            output = output_weighted[inv_indices]
            
            # Combine outputs from multiple experts per token using scatter_add
            # Since each token was sent to num_experts_per_tok experts,
            # we need to sum their weighted contributions
            num_tokens = batch_size * seq_len
            
            # Map each duplicated position (now in x_rep/original duplicate order)
            # to its original token index (0..num_tokens-1)
            # Note: after reordering with inv_indices above, `output` is in x_rep order,
            # so we must not use inv_indices here. Instead, we create a simple arange
            # over the duplicate positions and integer-divide by k.
            original_token_indices = (
                torch.arange(num_tokens * self.num_experts_per_tok, device=output.device)
                // self.num_experts_per_tok
            )
            
            # Use scatter_add to efficiently combine all expert outputs at once
            # This avoids both loops and unnecessary memory allocation
            combined_output = torch.zeros((num_tokens, self.n_embd), 
                                         dtype=output.dtype, device=output.device)
            combined_output.scatter_add_(
                0,
                original_token_indices.unsqueeze(-1).expand(-1, self.n_embd),
                output
            )
            output = combined_output
            
            # Reshape back to original batch dimensions
            output = rearrange(output, '(batch seq) hidden -> batch seq hidden', 
                             batch=batch_size, seq=seq_len)
        else:
            # No tokens to process - return zeros
            output = torch.zeros((batch_size, seq_len, self.n_embd), 
                                dtype=x.dtype, device=x.device)
        
        # Compute auxiliary losses
        # Router z-loss: encourages router logits to stay small for stability
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        # Load balance loss: encourages uniform distribution across experts
        p_i = F.softmax(router_logits, dim=-1)  # Shape: (batch*seq, num_experts)
        p_i = p_i.mean(dim=0)  # Average probability per expert: (num_experts,)
        
        if len(selected_experts) > 0:
            experts_flat = selected_experts.flatten()
            # Use scatter_add instead of bincount for torch.compile compatibility
            f_i = torch.zeros(self.num_experts, dtype=x.dtype, device=x.device)
            ones = torch.ones_like(experts_flat, dtype=x.dtype) / len(experts_flat)
            f_i.scatter_add_(0, experts_flat, ones)
            load_balance_loss = self.num_experts * (f_i @ p_i)
        else:
            # No tokens routed - no imbalance possible
            load_balance_loss = torch.tensor(0.0, device=x.device)
            f_i = torch.zeros(self.num_experts, device=x.device)
        
        # Package auxiliary losses for the training script
        aux_loss = {
            'router_z_loss': router_z_loss,
            'load_balance_loss': load_balance_loss,
        }
        
        return output, aux_loss, f_i