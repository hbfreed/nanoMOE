"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo
from einops import rearrange
from moe import SDD, DSD 
import stk.ops
import stk.random
import stk.matrix 

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                                        .view(1, 1, config.n_ctx, config.n_ctx))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

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

#based right on https://github.com/huggingface/transformers/blob/6017f5e8ed33d48096cdf8630d1cc7cbf2550c90/src/transformers/models/olmoe/modeling_olmoe.py#L567
#This loops over experts. obviously we don't want that long-term, so we are writing the triton kernel.
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
        
        # Compute auxiliary losses
        # Router z-loss: encourages router logits to stay small for stability
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        # Load balance loss: encourages uniform distribution across experts
        p_i = F.softmax(router_logits, dim=-1)  # Shape: (batch*seq, num_experts)
        p_i = p_i.mean(dim=0)  # Average probability per expert: (num_experts,)
        
        if len(selected_experts) > 0:
            experts_flat = selected_experts.flatten()
            # Use scatter_add for torch.compile compatibility
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
        
        return output.view(batch_size, seq_len, hidden_dim), aux_loss, f_i

import numpy as np
import stk.ops
import torch
from stk import Matrix

import megablocks.ops as ops
from megablocks.layers import common, dmlp_registry, moe, mpu
from megablocks.layers.arguments import Arguments


def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x

class MoeMegaBlocks(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.n_embd = config.n_embd #hidden size
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



class MoeMLPSTK(nn.Module):

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

        expert_sort_indices = torch.argsort(selected_experts_rep, stable=True).long()
        x_sorted = x_rep[expert_sort_indices]
        selected_experts_sorted = selected_experts_rep[expert_sort_indices]
        router_weights_sorted = router_weights_rep[expert_sort_indices]

        # Compute the inverse permutation correctly
        inv_expert_sort_indices = torch.empty_like(expert_sort_indices)
        inv_expert_sort_indices[expert_sort_indices] = torch.arange(
            len(expert_sort_indices), device=expert_sort_indices.device
        )

        return x_sorted, selected_experts_sorted, router_weights_sorted, inv_expert_sort_indices

    def _pad_to_blocks(self, x_sorted, selected_experts_sorted):
        """Pad each expert's tokens to multiples of block_size and track unpadding indices."""
        device = x_sorted.device
        num_tokens = x_sorted.shape[0]
        token_dim = x_sorted.shape[-1]

        # Use self.num_experts and self.block_size directly
        min_tokens_or_experts = min(num_tokens, self.num_experts)
        max_blocks = min_tokens_or_experts + (num_tokens - min_tokens_or_experts) // self.block_size

        # Bucketize capacity to reduce torch.compile recompilations
        # Round to nearest 2048 tokens (16 blocks of 128 each)
        raw_capacity = max_blocks * self.block_size
        bucket_size = 2048
        capacity_tokens = ((raw_capacity + bucket_size - 1) // bucket_size) * bucket_size

        # Per-expert counts via scatter_add (compile-safe; avoids bincount)
        tokens_per_expert = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        ones = torch.ones_like(selected_experts_sorted, dtype=torch.long)
        tokens_per_expert.scatter_add_(0, selected_experts_sorted, ones)

        # Round each expert up to a multiple of block_size
        tokens_per_expert_padded = ((tokens_per_expert + self.block_size - 1) // self.block_size) * self.block_size

        # Exclusive-prefix sums (orig vs padded) for placement
        offset_original = F.pad(tokens_per_expert.cumsum(0), (1, 0))
        offset_padded  = F.pad(tokens_per_expert_padded.cumsum(0), (1, 0))

        # Allocate fixed capacity once; the tail is never indexed
        x_padded = x_sorted.new_zeros((capacity_tokens, token_dim))

        
        # Map each sorted token to its padded position
        token_idx = torch.arange(num_tokens, device=x_sorted.device)
        idx_within_expert = token_idx - offset_original[selected_experts_sorted]
        unpad_indices = idx_within_expert + offset_padded[selected_experts_sorted]

        # Scatter the actual tokens into their padded slots
        x_padded[unpad_indices] = x_sorted

        # Return exactly what you wanted
        return x_padded, tokens_per_expert_padded, unpad_indices

    def _make_topology(self, x_padded: torch.Tensor,
                    tokens_per_expert_padded: torch.Tensor,
                    ) -> stk.matrix.Matrix:
        """Construct a block-compressed sparse topology for stk.ops.sdd."""
        if x_padded.dim() != 2:
            raise ValueError("x_padded must be 2D [tokens, hidden].")

        device = x_padded.device
        dtype = x_padded.dtype

        if self.d_ffn % self.block_size != 0:
            raise ValueError(f"Model width d_ffn={self.d_ffn} must be divisible by block_size={self.block_size}.")

        # Skip validation to avoid .item() calls in hot path
        # if int(tokens_per_expert_padded.sum().item()) != x_padded.shape[0]:
        #     raise ValueError("x_padded row count must equal sum(tokens_per_expert_padded).")

        if x_padded.shape[0] % self.block_size != 0:
            raise ValueError("Padded rows must be a multiple of block_size for BCSR layout.")

        if torch.any(tokens_per_expert_padded % self.block_size):
            raise ValueError("Each expert's padded token count must be divisible by block_size.")

        tokens_per_expert_blocks = tokens_per_expert_padded // self.block_size
        num_row_blocks = tokens_per_expert_blocks.sum()
        col_blocks_per_expert = self.d_ffn // self.block_size
        total_col_blocks = col_blocks_per_expert * self.num_experts

        # Use int32 - should be sufficient for most use cases
        row_index_dtype = torch.int32
        column_index_dtype = torch.int32

        # Check if we have blocks using tensor comparison
        if num_row_blocks.eq(0):
            row_indices = torch.empty(0, dtype=row_index_dtype, device=device)
            column_indices = torch.empty(0, dtype=column_index_dtype, device=device)
            offsets = torch.zeros(1, dtype=torch.int32, device=device)
            data = torch.zeros((0, self.block_size, self.block_size), dtype=dtype, device=device)
            matrix_rows = 0
        else:
            row_indices = torch.arange(
                num_row_blocks, device=device, dtype=row_index_dtype
            ).repeat_interleave(col_blocks_per_expert)

            expert_ids_per_row = torch.repeat_interleave(
                torch.arange(self.num_experts, device=device),
                tokens_per_expert_blocks
            )

            base_offsets = expert_ids_per_row.to(column_index_dtype) * col_blocks_per_expert
            base_offsets = base_offsets.repeat_interleave(col_blocks_per_expert)

            col_range = torch.arange(
                col_blocks_per_expert, device=device, dtype=column_index_dtype
            ).repeat(num_row_blocks)

            column_indices = base_offsets + col_range

            offsets = torch.arange(
                num_row_blocks + 1, device=device, dtype=torch.int32
            ) * col_blocks_per_expert

            num_blocks = num_row_blocks * col_blocks_per_expert
            # Allocate minimal data tensor - STK needs this
            data = torch.zeros((num_blocks, self.block_size, self.block_size), device=device, dtype=dtype)
            matrix_rows = tokens_per_expert_padded.sum()

        topo = stk.matrix.Matrix(
            size=(matrix_rows, self.d_ffn * self.num_experts),
            data=data,
            row_indices=row_indices,
            column_indices=column_indices,
            offsets=offsets,
        )
        return topo

    @torch.compiler.disable
    def _run_stk_ops(self, x_padded, tokens_per_expert_padded):
        """Run STK operations without compilation to avoid recompilation overhead."""
        topo = self._make_topology(x_padded, tokens_per_expert_padded)

        block_sparse = stk.ops.sdd(x_padded, self.w1, topo)
        activated_blocks = F.gelu(block_sparse.data)
        block_sparse = stk.matrix.Matrix(
            size=block_sparse.size(),
            data=activated_blocks,
            row_indices=block_sparse.row_indices,
            column_indices=block_sparse.column_indices,
            offsets=block_sparse.offsets,
            column_indices_t=block_sparse.column_indices_t,
            offsets_t=block_sparse.offsets_t,
            block_offsets_t=block_sparse.block_offsets_t,
        )
        expert_output = stk.ops.dsd(block_sparse, self.w2)
        return expert_output

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        x_flat = rearrange(x, 'batch seq hidden -> (batch seq) hidden')
        
        router_weights, selected_experts, router_logits = self._route_tokens(x_flat)
        
        x_sorted, selected_experts_sorted, router_weights_sorted, sort_indices = self._sort_by_expert(
            x_flat, router_weights, selected_experts
        )
        
        x_padded, tokens_per_expert_padded, unpad_indices = self._pad_to_blocks(x_sorted, selected_experts_sorted)
        total_padded_tokens = tokens_per_expert_padded.sum()
        x_padded = x_padded[:total_padded_tokens]

        expert_output = self._run_stk_ops(x_padded, tokens_per_expert_padded)

        if router_weights_sorted.shape[0] != unpad_indices.shape[0]:
            raise RuntimeError(
                "Router weights and unpad indices disagree: "
                f"{router_weights_sorted.shape[0]} vs {unpad_indices.shape[0]}"
            )
        if unpad_indices.numel() and unpad_indices.max() >= expert_output.shape[0]:
            raise RuntimeError(
                "Unpad index out of range: "
                f"max={unpad_indices.max()} rows={expert_output.shape[0]}"
            )

        output_unpadded = torch.index_select(expert_output, 0, unpad_indices)
        output_weighted = output_unpadded * router_weights_sorted

        # Unpermute back to original token order using the original sort indices
        if sort_indices.numel() != output_weighted.shape[0]:
            raise RuntimeError(
                "Sort indices and output disagree: "
                f"{sort_indices.shape[0]} vs {output_weighted.shape[0]}"
            )
        # Properly apply inverse permutation
        output = output_weighted[sort_indices]

        num_tokens = batch_size * seq_len
        
        original_token_indices = (
            torch.arange(num_tokens * self.num_experts_per_tok, device=output.device)
            // self.num_experts_per_tok
        )
        
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

        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean() # router z-loss keeps router logits small
        
        # Load balance loss: encourages uniform distribution across experts
        p_i = F.softmax(router_logits, dim=-1)  # Shape: (batch*seq, num_experts)
        p_i = p_i.mean(dim=0)  # Average probability per expert: (num_experts,)
        
        if len(selected_experts) > 0:
            experts_flat = selected_experts.flatten()

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

        # Pre-allocate reusable index buffers to avoid repeated torch.arange calls
        # Maximum possible tokens in a batch (batch_size * seq_len * num_experts_per_tok)
        max_total_tokens = self.seq_len * 64 * self.num_experts_per_tok  # 64 is max batch size from config

        # Buffer for block indices (already exists, keep it)
        self.register_buffer(
            "_index_buf",
            torch.arange(self._max_blocks_static, dtype=torch.int32),
            persistent=False
        )

        # Buffer for token indices (used in _pad_to_blocks)
        self.register_buffer(
            "_token_indices",
            torch.arange(max_total_tokens, dtype=torch.long),
            persistent=False
        )

        # Buffer for inverse indices (used in _sort_by_expert)
        self.register_buffer(
            "_inv_indices_buffer",
            torch.arange(max_total_tokens, dtype=torch.long),
            persistent=False
        )

        # Pre-compute constants for _create_sparse_indices
        self._num_ffn_blocks_tensor = torch.tensor(self._num_ffn_blocks, dtype=torch.long)
        self.register_buffer(
            "_expert_ids_base",
            torch.arange(self.num_experts, dtype=torch.long).repeat_interleave(self._num_ffn_blocks),
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
        
        expert_sort_indices = torch.argsort(selected_experts_rep, stable=True).long()
        x_sorted = x_rep[expert_sort_indices]
        selected_experts_sorted = selected_experts_rep[expert_sort_indices]
        router_weights_sorted = router_weights_rep[expert_sort_indices]

        return x_sorted, selected_experts_sorted, router_weights_sorted, expert_sort_indices
    
    def _pad_to_blocks(self, x_sorted, selected_experts_sorted):
        """Pad each expert's tokens to multiples of block_size and track unpadding indices."""
        device = x_sorted.device
        num_tokens = x_sorted.shape[0]
        token_dim = x_sorted.shape[-1]

        # Use self.num_experts and self.block_size directly
        min_tokens_or_experts = min(num_tokens, self.num_experts)
        max_blocks = min_tokens_or_experts + (num_tokens - min_tokens_or_experts) // self.block_size

        # Bucketize capacity to reduce torch.compile recompilations
        # Round to nearest 2048 tokens (16 blocks of 128 each)
        raw_capacity = max_blocks * self.block_size
        bucket_size = 2048
        capacity_tokens = ((raw_capacity + bucket_size - 1) // bucket_size) * bucket_size

        # Per-expert counts via scatter_add (compile-safe; avoids bincount)
        tokens_per_expert = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        ones = torch.ones_like(selected_experts_sorted, dtype=torch.long)
        tokens_per_expert.scatter_add_(0, selected_experts_sorted, ones)

        # Round each expert up to a multiple of block_size
        tokens_per_expert_padded = ((tokens_per_expert + self.block_size - 1) // self.block_size) * self.block_size

        # Exclusive-prefix sums (orig vs padded) for placement
        offset_original = F.pad(tokens_per_expert.cumsum(0), (1, 0))              # [num_experts+1]
        offset_padded  = F.pad(tokens_per_expert_padded.cumsum(0), (1, 0))        # [num_experts+1]

        # Allocate fixed capacity once; the tail is never indexed
        x_padded = x_sorted.new_zeros((capacity_tokens, token_dim))

        # Map each sorted token to its padded position
        token_idx = self._token_indices[:num_tokens]
        idx_within_expert = token_idx - offset_original[selected_experts_sorted]
        unpad_indices = idx_within_expert + offset_padded[selected_experts_sorted]

        # Scatter the actual tokens into their padded slots
        x_padded[unpad_indices] = x_sorted

        # Return exactly what you wanted
        return x_padded, tokens_per_expert_padded, unpad_indices

    @torch.compiler.disable
    def _create_sparse_indices(self, tokens_per_expert_padded):
        device = tokens_per_expert_padded.device

        # Compute blocks per expert (vectorized)
        num_token_blocks_per_expert = tokens_per_expert_padded // self.block_size
        blocks_per_expert = num_token_blocks_per_expert * self._num_ffn_blocks

        # Convert to Python int ONCE to avoid dynamic shapes in torch.compile
        # This single .item() call prevents thousands of shape guard checks
        total_blocks = int(blocks_per_expert.sum())

        indices = torch.arange(total_blocks, device=device, dtype=torch.long)

        # Single cumsum for expert assignment
        cumsum = blocks_per_expert.cumsum(0)
        expert_ids = torch.searchsorted(cumsum, indices, right=True).clamp(max=self.num_experts - 1)

        # Fuse within-expert index computation
        cumsum_padded = F.pad(cumsum[:-1], (1, 0))
        within_expert_idx = indices - cumsum_padded[expert_ids]

        # Compute row indices (combine operations)
        token_block_cumsum = num_token_blocks_per_expert.cumsum(0)
        token_block_offset = F.pad(token_block_cumsum[:-1], (1, 0))

        # Fast modulo and division using bit operations when possible
        # Since _num_ffn_blocks is often a power of 2, we can optimize
        within_expert_block = within_expert_idx // self._num_ffn_blocks
        within_expert_ffn = within_expert_idx % self._num_ffn_blocks

        row_indices = token_block_offset[expert_ids] + within_expert_block
        weight_col_indices = expert_ids * self._num_ffn_blocks + within_expert_ffn
        output_col_indices = within_expert_ffn

        return row_indices.int(), weight_col_indices.int(), output_col_indices.int()

    # @torch.compiler.disable #sadly have to disable because of triton- TODO: fix this!
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
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if config.use_moe:
            self.mlp = MoeMLPSTK(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        mlp_out = self.mlp(self.ln_2(x))

        if isinstance(mlp_out, tuple): #handle moes and regular MLP
            if len(mlp_out) == 3:  # MoeMLP returns output, aux_loss, f_i
                mlp_x, aux_loss, f_i = mlp_out
            else:  # Might be old format with just output, aux_loss
                mlp_x, aux_loss = mlp_out
                f_i = None
        else:
            mlp_x, aux_loss, f_i = mlp_out, None, None
        x = x + mlp_x
        return x, aux_loss, f_i    
@dataclass
class GPTConfig:
    n_ctx: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True  # Normalize top-k router probabilities to sum to 1
    block_size: int = 128  # Triton kernel tile size for MoE
    block_k: int = 64  # Triton kernel K dimension for MoE

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.n_ctx is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.n_ctx, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        if config.use_moe:
            print("active parameters per token: %.2fM" % (self.get_active_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def get_active_num_params(self, non_embedding=True):
        """
        Return the number of active parameters per token in MoE models.
        For MoE, only num_experts_per_tok out of num_experts are active.
        """
        if not self.config.use_moe:
            return self.get_num_params(non_embedding)
        
        total_params = self.get_num_params(non_embedding)
        
        # Calculate MLP params (all experts)
        mlp_params = 0
        for name, param in self.named_parameters():
            if 'mlp' in name and ('w1' in name or 'w2' in name or 'experts' in name):
                mlp_params += param.numel()
        
        # Active MLP params = mlp_params * (num_experts_per_tok / num_experts)
        active_mlp_params = mlp_params * (self.config.num_experts_per_tok / self.config.num_experts)
        
        # Active total = total - all_mlp + active_mlp
        active_params = total_params - mlp_params + active_mlp_params
        
        return active_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.n_ctx, f"Cannot forward sequence of length {t}, context length is only {self.config.n_ctx}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        combined_aux_loss = None
        aux_loss_count = 0
        all_expert_usage = []  # Collect expert usage from all layers
        
        for block in self.transformer.h:
            block_out = block(x)
            
            # Handle different return formats
            if len(block_out) == 3:
                x, aux_loss, f_i = block_out
                if f_i is not None:
                    all_expert_usage.append(f_i)
            else:
                x, aux_loss = block_out

            if aux_loss is not None: #accumulate the aux
                if combined_aux_loss is None:
                    combined_aux_loss = {k: v.clone() for k, v in aux_loss.items()}
                else:
                    for key in aux_loss:
                        combined_aux_loss[key] += aux_loss[key]
                aux_loss_count += 1
        
        if combined_aux_loss is not None and aux_loss_count > 0:
            for key in combined_aux_loss:
                combined_aux_loss[key] /= aux_loss_count #average out the accumulated total
        
        # Average expert usage across layers
        if all_expert_usage:
            avg_expert_usage = torch.stack(all_expert_usage).mean(dim=0)
            combined_aux_loss['expert_usage'] = avg_expert_usage

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = ce_loss  # Start with cross-entropy loss
            if combined_aux_loss is not None: # add in the aux losses scaled by their weights
                loss = loss + 0.01 * combined_aux_loss['load_balance_loss'] + 0.001 * combined_aux_loss['router_z_loss']
                combined_aux_loss['ce_loss'] = ce_loss  # Store CE loss in aux dict for logging
            
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            ce_loss = None

        return logits, loss, combined_aux_loss  # Return aux losses for logging

    def crop_block_size(self, n_ctx):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert n_ctx <= self.config.n_ctx
        self.config.n_ctx = n_ctx
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:n_ctx])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:n_ctx,:n_ctx]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, n_ctx=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['n_ctx'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) based on GPU-specific peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_active_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.n_ctx
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Detect GPU and use appropriate peak FLOPS
        # Using Tensor Core FP16/BF16 performance (following Karpathy's approach with A100)
        gpu_flops_map = {
            'A100': 312e12,        # A100 BF16 tensor core peak: 312 TFLOPS  
            'RTX 3090': 71e12,     # RTX 3090 FP16 tensor core (dense): 71 TFLOPS
            'RTX 4090': 82.58e12,  # RTX 4090 FP16 tensor core: 82.58 TFLOPS
            'V100': 125e12,        # V100 FP16 tensor core: 125 TFLOPS
            'H100': 989e12,        # H100 FP16 tensor core: 989 TFLOPS
            'RTX 3080': 59e12,     # RTX 3080 FP16 tensor core: 59 TFLOPS
            'RTX 3070': 40e12,     # RTX 3070 FP16 tensor core: 40 TFLOPS
            'RTX 3060': 25e12,     # RTX 3060 FP16 tensor core: 25 TFLOPS
        }
        
        # Get GPU name
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # Match known GPU types
            flops_promised = None
            for gpu_key in gpu_flops_map:
                if gpu_key in gpu_name:
                    flops_promised = gpu_flops_map[gpu_key]
                    break
            
            if flops_promised is None:
                # Default to A100 if GPU not recognized, but warn
                print(f"Warning: Unknown GPU '{gpu_name}', defaulting to A100 FLOPS for MFU calculation")
                flops_promised = 312e12
        else:
            flops_promised = 312e12  # Default to A100 if no CUDA
        
        # express our flops throughput as ratio of GPU's peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at n_ctx
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx