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

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo
from einops import rearrange
import stk
import stk.ops
import stk.random
import stk.matrix
from megablocks import ops
from megablocks.layers.gelu import gelu
from topology_var import topology_var


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

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
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(
                    1, 1, config.n_ctx, config.n_ctx
                ),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


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


class MoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_embd = config.n_embd
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)

        # Use block_size from config instead of hardcoded blocking
        self.block_size = config.block_size
        base_d_ffn = 4 * self.n_embd // self.num_experts_per_tok
        default_d_ffn = (
            (base_d_ffn + self.block_size - 1) // self.block_size
        ) * self.block_size

        # Parse expert_sizes from config (list of (count, size) tuples)
        expert_sizes_spec = getattr(config, "expert_sizes", None)
        if expert_sizes_spec is not None:
            # Build expert_sizes list from tuple specification
            self.expert_sizes = []
            for count, size in expert_sizes_spec:
                # Round up to block_size multiple
                size_rounded = (
                    (size + self.block_size - 1) // self.block_size
                ) * self.block_size
                self.expert_sizes.extend([size_rounded] * count)

            # Validate that counts match num_experts
            if len(self.expert_sizes) != self.num_experts:
                raise ValueError(
                    f"expert_sizes specification produces {len(self.expert_sizes)} experts, "
                    f"but config.num_experts={self.num_experts}"
                )
        else:
            # Default: uniform sizing (backward compatible)
            self.expert_sizes = [default_d_ffn] * self.num_experts

        # Compute cumulative offsets for indexing into weight matrices
        self.expert_offsets = [0]
        for size in self.expert_sizes:
            self.expert_offsets.append(self.expert_offsets[-1] + size)
        self.total_expert_width = self.expert_offsets[-1]

        self.d_ffn = self.expert_sizes[0]

        # Register buffers for efficient CUDA kernel access
        self.register_buffer(
            "expert_size_blocks",
            torch.tensor(
                [s // self.block_size for s in self.expert_sizes], dtype=torch.int32
            ),
            persistent=False,
        )
        self.register_buffer(
            "expert_block_offsets",
            torch.tensor(
                [o // self.block_size for o in self.expert_offsets], dtype=torch.int32
            ),
            persistent=False,
        )

        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)

        self.w1 = nn.Parameter(torch.empty(self.n_embd, self.total_expert_width))
        self.w2 = nn.Parameter(torch.empty(self.total_expert_width, self.n_embd))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)
        
        # need these bits for the megablocks ops
        self.sort_end_bit = max(
            int(np.ceil(np.log2(self.num_experts))), 1
        )  

        # Use total_expert_width for max_column_index calculation
        max_column_index = self.total_expert_width // self.block_size

        self.transpose_sort_end_bit = max(int(np.ceil(np.log2(max_column_index))), 1)

    @torch.compiler.disable
    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape

        x_flat = rearrange(
            x, "batch_size seq_len n_embd -> (batch_size seq_len) n_embd"
        )

        router_logits = self.router(x_flat)

        router_probs = F.softmax(
            router_logits, dim=-1, dtype=torch.float32
        )  # Use float32 for stability

        expert_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )

        if self.norm_topk_prob:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        expert_weights = expert_weights.to(x.dtype)

        expert_weights_flat = rearrange(expert_weights, "... -> (...)")
        selected_experts_flat = rearrange(selected_experts, "... -> (...)")

        bin_ids, indices, tokens_per_expert = self._sort_tokens_by_expert(
            selected_experts_flat
        )

        padded_bins, topology = self._create_topology(x_flat, tokens_per_expert)

        x_permuted = self._gather_tokens(
            x_flat, indices, bin_ids, tokens_per_expert, padded_bins
        )

        x_permuted = stk.ops.sdd(x_permuted, self.w1, topology)

        x_permuted = gelu(x_permuted)

        x_permuted = stk.ops.dsd(x_permuted, self.w2)

        x_permuted = self._scatter_tokens(
            x_permuted,
            indices,
            bin_ids,
            expert_weights_flat,
            tokens_per_expert,
            padded_bins,
        )

        output = rearrange(
            x_permuted,
            "(batch_size seq_len) n_embd -> batch_size seq_len n_embd",
            batch_size=batch_size,
            seq_len=seq_len,
        )

        router_z_loss = (
            torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        )  # router z-loss keeps router logits small

        p_i = router_probs.mean(
            dim=0
        ).to(
            x.dtype
        )

        experts_flat = selected_experts.flatten()
        f_i = torch.zeros(self.num_experts, dtype=x.dtype, device=x.device)
        ones = torch.ones_like(experts_flat, dtype=x.dtype) / len(experts_flat)
        f_i.scatter_add_(0, experts_flat, ones)
        load_balance_loss = self.num_experts * (f_i @ p_i)

        aux_loss = {
            "router_z_loss": router_z_loss,
            "load_balance_loss": load_balance_loss,
        }

        return output, aux_loss, f_i

    def _sort_tokens_by_expert(self, selected_experts_flat):
        """Group token assignments by expert id."""

        bin_ids, indices = ops.sort(selected_experts_flat, self.sort_end_bit)
        tokens_per_expert = ops.histogram(selected_experts_flat, self.num_experts)

        return bin_ids, indices, tokens_per_expert

    def _create_topology(self, x, tokens_per_expert):
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.block_size)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = padded_bins.contiguous()

        padded_tokens = padded_bins[-1].clamp_min(self.block_size)

        block_rows = padded_tokens // self.block_size

        # Use variable-size topology with per-expert block counts
        column_indices = topology_var(
            padded_bins,
            self.expert_size_blocks,  # Per-expert block counts
            self.expert_block_offsets,  # Cumulative block offsets
            self.block_size,
            block_rows,
        )

        # Compute all expert token blocks at once
        prepend = padded_bins.new_zeros(1)
        bin_sizes = torch.diff(padded_bins, prepend=prepend)
        expert_token_blocks = bin_sizes // self.block_size

        # Repeat each expert's size by how many token blocks it handles
        repeated_sizes = torch.repeat_interleave(
            self.expert_size_blocks, expert_token_blocks
        )

        # Cumulative sum gives you offsets
        offsets = torch.cat([repeated_sizes.new_zeros(1), repeated_sizes.cumsum(0)])

        column_indices = column_indices.to(torch.int32)
        offsets = offsets.to(torch.int32)

        shape = (padded_tokens, self.total_expert_width)

        num_blocks = column_indices.numel()
        data_placeholder = torch.empty(
            num_blocks,
            self.block_size,
            self.block_size,
            dtype=x.dtype,
            device="meta",
        )

        row_indices = stk.ops.row_indices(
            shape, data_placeholder, offsets, column_indices
        )
        row_indices = row_indices.to(torch.int32)

        column_indices_t, offsets_t, block_offsets_t = self._sparse_transpose(
            shape, row_indices, column_indices
        )
        column_indices_t = column_indices_t.to(torch.int32)
        offsets_t = offsets_t.to(torch.int32)
        block_offsets_t = block_offsets_t.to(torch.int32)

        topology = stk.Matrix(
            shape,
            data_placeholder,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
        )

        return padded_bins, topology

    def _sparse_transpose(self, size, row_indices, column_indices):
        # Use total_expert_width instead of d_ffn * num_experts
        block_columns = self.total_expert_width // self.block_size

        _, gather_indices = ops.sort(
            column_indices.int(),
            self.transpose_sort_end_bit,
        )

        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        if nnz_per_column.dim() == 0:
            # This addresses an edge case when ffn_hidden_size is equal to self.block_size.
            nnz_per_column = nnz_per_column.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def _gather_tokens(self, x, indices, bin_ids, tokens_per_expert, padded_bins):
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.contiguous()

        return ops.padded_gather(
            x, indices, bin_ids, bins, padded_bins, self.num_experts_per_tok
        )

    def _scatter_tokens(
        self, x, indices, bin_ids, weights, tokens_per_expert, padded_bins
    ):
        """Un-permute tokens and apply expert weights."""
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.contiguous()

        return ops.padded_scatter(
            x, indices, bin_ids, weights, bins, padded_bins, self.num_experts_per_tok
        )

    def _get_topo_tensors(self, topology):
        """Extract the tensor components from STK Matrix for reconstruction."""
        return (
            topology.row_indices,
            topology.column_indices,
            topology.offsets,
            topology.column_indices_t,
            topology.offsets_t,
            topology.block_offsets_t,
        )


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if config.use_moe:
            self.mlp = MoeMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        mlp_out = self.mlp(self.ln_2(x))

        if isinstance(mlp_out, tuple):  # handle moes and regular MLP
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
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True  # Normalize top-k router probabilities to sum to 1
    block_size: int = 128  # Triton kernel tile size for MoE
    block_k: int = 64  # Triton kernel K dimension for MoE
    expert_sizes: Optional[list] = (
        None  # List of (count, size) tuples, e.g., [(1, 2048), (7, 1024)]
    )
    load_balance_loss_weight: float = 0.01  # Weight for load balance auxiliary loss
    router_z_loss_weight: float = 0.001  # Weight for router z-loss auxiliary loss


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.n_ctx is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.n_ctx, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
        if config.use_moe:
            # Check if we have variable-size experts
            has_variable_experts = (
                hasattr(config, "expert_sizes") and config.expert_sizes is not None
            )

            if has_variable_experts:
                # Show range for variable-size experts
                expected = self.get_active_num_params(mode="expected")
                min_active = self.get_active_num_params(mode="min")
                max_active = self.get_active_num_params(mode="max")
                print(
                    "active parameters per token (expected): %.2fM" % (expected / 1e6,)
                )
                print(
                    "  range: %.2fM (min) to %.2fM (max)"
                    % (min_active / 1e6, max_active / 1e6)
                )
            else:
                print(
                    "active parameters per token: %.2fM"
                    % (self.get_active_num_params() / 1e6,)
                )

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

    def get_active_num_params(self, non_embedding=True, mode="expected"):
        """
        Return the number of active parameters per token in MoE models.
        For MoE, only num_experts_per_tok out of num_experts are active.

        For variable-size experts, this depends on which experts are selected:

        Args:
            non_embedding: Whether to exclude embedding params
            mode: 'expected' (default), 'min', or 'max'
                - 'expected': Expected active params assuming uniform routing
                - 'min': Minimum (all k smallest experts)
                - 'max': Maximum (all k largest experts)
        """
        if not self.config.use_moe:
            return self.get_num_params(non_embedding)

        total_params = self.get_num_params(non_embedding)

        # Get the first MoE layer to check for variable sizing
        moe_layer = None
        for module in self.modules():
            if hasattr(module, "expert_sizes"):
                moe_layer = module
                break

        if moe_layer is None:
            # No MoE layer found, shouldn't happen but handle gracefully
            return total_params

        # Calculate total MLP params across all layers
        mlp_params = 0
        for name, param in self.named_parameters():
            if "mlp" in name and ("w1" in name or "w2" in name or "experts" in name):
                mlp_params += param.numel()

        # Calculate params per expert (for one layer)
        expert_params_per_layer = []
        for expert_size in moe_layer.expert_sizes:
            # w1: n_embd × expert_size, w2: expert_size × n_embd
            w1_params = self.config.n_embd * expert_size
            w2_params = expert_size * self.config.n_embd
            expert_params_per_layer.append(w1_params + w2_params)

        k = self.config.num_experts_per_tok

        if mode == "min":
            # Activate k smallest experts
            sorted_params = sorted(expert_params_per_layer)
            active_per_layer = sum(sorted_params[:k])
        elif mode == "max":
            # Activate k largest experts
            sorted_params = sorted(expert_params_per_layer, reverse=True)
            active_per_layer = sum(sorted_params[:k])
        else:  # mode == 'expected'
            # Expected value assuming uniform routing probability
            # Each expert has probability k/E of being selected
            # Expected = (k/E) * sum(all expert params)
            total_expert_params_per_layer = sum(expert_params_per_layer)
            active_per_layer = (
                k / self.config.num_experts
            ) * total_expert_params_per_layer

        # Scale by number of layers
        active_mlp_params = active_per_layer * self.config.n_layer

        # Active total = total - all_mlp + active_mlp
        active_params = total_params - mlp_params + active_mlp_params

        return active_params

    def get_active_params_stats(self, non_embedding=True):
        """
        Get comprehensive active parameter statistics for variable-size experts.
        Returns dict with min, max, expected, and per-expert breakdown.
        """
        if not self.config.use_moe:
            return {"total": self.get_num_params(non_embedding)}

        stats = {
            "total_params": self.get_num_params(non_embedding),
            "expected_active": self.get_active_num_params(
                non_embedding, mode="expected"
            ),
            "min_active": self.get_active_num_params(non_embedding, mode="min"),
            "max_active": self.get_active_num_params(non_embedding, mode="max"),
        }

        # Get expert sizes for reference
        moe_layer = None
        for module in self.modules():
            if hasattr(module, "expert_sizes"):
                moe_layer = module
                break

        if moe_layer is not None:
            stats["expert_sizes"] = moe_layer.expert_sizes
            stats["num_experts"] = self.config.num_experts
            stats["experts_per_tok"] = self.config.num_experts_per_tok

        return stats

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
        assert t <= self.config.n_ctx, (
            f"Cannot forward sequence of length {t}, context length is only {self.config.n_ctx}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        combined_aux_loss = None
        aux_loss_count = 0
        expert_usage_sum = None
        expert_usage_count = 0  # Track number of MoE layers contributing usage stats
        expert_usage_per_layer = []  # Store per-layer expert usage

        for block in self.transformer.h:
            block_out = block(x)

            # Handle different return formats
            if len(block_out) == 3:
                x, aux_loss, f_i = block_out
                if f_i is not None:
                    if expert_usage_sum is None:
                        expert_usage_sum = f_i.clone()
                    else:
                        expert_usage_sum += f_i
                    expert_usage_count += 1
                    # Store per-layer expert usage
                    expert_usage_per_layer.append(f_i.clone())
            else:
                x, aux_loss = block_out

            if aux_loss is not None:  # accumulate the aux
                if combined_aux_loss is None:
                    combined_aux_loss = {k: v.clone() for k, v in aux_loss.items()}
                else:
                    for key in aux_loss:
                        combined_aux_loss[key] += aux_loss[key]
                aux_loss_count += 1

        if combined_aux_loss is not None and aux_loss_count > 0:
            for key in combined_aux_loss:
                combined_aux_loss[key] /= (
                    aux_loss_count  # average out the accumulated total
                )

        # Average expert usage across layers
        if expert_usage_sum is not None and expert_usage_count > 0:
            avg_expert_usage = expert_usage_sum / expert_usage_count
            if combined_aux_loss is None:
                combined_aux_loss = {}
            combined_aux_loss["expert_usage"] = avg_expert_usage
            combined_aux_loss["expert_usage_per_layer"] = expert_usage_per_layer

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            loss = ce_loss  # Start with cross-entropy loss
            if (
                combined_aux_loss is not None
            ):  # add in the aux losses scaled by their weights
                loss = (
                    loss
                    + self.config.load_balance_loss_weight
                    * combined_aux_loss["load_balance_loss"]
                    + self.config.router_z_loss_weight
                    * combined_aux_loss["router_z_loss"]
                )
                combined_aux_loss["ce_loss"] = (
                    ce_loss  # Store CE loss in aux dict for logging
                )

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
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
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :n_ctx, :n_ctx]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, n_ctx=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["n_ctx"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
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
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) based on GPU-specific peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_active_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_ctx
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Detect GPU and use appropriate peak FLOPS
        # Using Tensor Core FP16/BF16 performance (following Karpathy's approach with A100)
        gpu_flops_map = {
            "A100": 312e12,  # A100 BF16 tensor core peak: 312 TFLOPS
            "RTX 3090": 71e12,  # RTX 3090 FP16 tensor core (dense): 71 TFLOPS
            "RTX 4090": 82.58e12,  # RTX 4090 FP16 tensor core: 82.58 TFLOPS
            "V100": 125e12,  # V100 FP16 tensor core: 125 TFLOPS
            "H100": 989e12,  # H100 FP16 tensor core: 989 TFLOPS
            "RTX 3080": 59e12,  # RTX 3080 FP16 tensor core: 59 TFLOPS
            "RTX 3070": 40e12,  # RTX 3070 FP16 tensor core: 40 TFLOPS
            "RTX 3060": 25e12,  # RTX 3060 FP16 tensor core: 25 TFLOPS
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
                print(
                    f"Warning: Unknown GPU '{gpu_name}', defaulting to A100 FLOPS for MFU calculation"
                )
                flops_promised = 312e12
        else:
            flops_promised = 312e12  # Default to A100 if no CUDA

        # express our flops throughput as ratio of GPU's peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
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
            idx_cond = (
                idx
                if idx.size(1) <= self.config.n_ctx
                else idx[:, -self.config.n_ctx :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

"""
MoeMLPWithTracking and GPTWithTracking are the same as MoeMLP and GPT, but with extra tracking of expert assignments and router logits.
"""

class MoeMLPWithTracking(MoeMLP):
    """Add expert assignment tracking to the mlp layer's forward pass"""

    @torch.compiler.disable
    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape

        x_flat = rearrange(x, 'batch_size seq_len n_embd -> (batch_size seq_len) n_embd')

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        expert_weights, selected_experts = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)

        if self.norm_topk_prob:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.to(x.dtype)
        expert_weights_flat = rearrange(expert_weights, '... -> (...)')
        selected_experts_flat = rearrange(selected_experts, '... -> (...)')

        bin_ids, indices, tokens_per_expert = self._sort_tokens_by_expert(selected_experts_flat)
        padded_bins, topology = self._create_topology(x_flat, tokens_per_expert)
        x_permuted = self._gather_tokens(x_flat, indices, bin_ids, tokens_per_expert, padded_bins)
        x_permuted = stk.ops.sdd(x_permuted, self.w1, topology)
        x_permuted = gelu(x_permuted)
        x_permuted = stk.ops.dsd(x_permuted, self.w2)

        x_permuted = self._scatter_tokens(x_permuted, indices, bin_ids, expert_weights_flat, tokens_per_expert, padded_bins)
        output = rearrange(x_permuted, '(batch_size seq_len) n_embd -> batch_size seq_len n_embd', batch_size=batch_size, seq_len=seq_len)

        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        p_i = router_probs.mean(dim=0).to(torch.bfloat16)

        experts_flat = selected_experts.flatten()
        f_i = torch.zeros(self.num_experts, dtype=x.dtype, device=x.device)
        ones = torch.ones_like(experts_flat, dtype=x.dtype) / len(experts_flat)
        f_i.scatter_add(0, experts_flat, ones)
        load_balance_loss = self.num_experts * (f_i @ p_i)
        
        expert_assignments = rearrange(selected_experts, '(batch seq) k -> batch seq k', batch=batch_size, seq=seq_len)
        router_logits_reshaped = rearrange(router_logits, '(batch seq) num_experts -> batch seq num_experts', batch=batch_size, seq=seq_len)
        router_probs_reshaped = rearrange(router_probs, '(batch seq) num_experts -> batch seq num_experts', batch=batch_size, seq=seq_len)

        aux_loss = {
            'router_z_loss': router_z_loss,
            'load_balance_loss': load_balance_loss,
            'expert_assignments': expert_assignments,
            'router_logits': router_logits_reshaped,
            'router_probs': router_probs_reshaped,
        }
        
        return output, aux_loss, f_i

class GPTWithTracking(GPT):
    """Track expert assignments across layers"""

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.n_ctx, f"Cannot forward sequence of length {t}, context length is only {self.config.n_ctx}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        combined_aux_loss = {}
        all_expert_usage = []
        all_expert_assignments = {}
        all_router_logits = {}
        all_router_probs = {}
        all_layer_entropies = {}
        aux_loss_count = 0

        for layer_idx, block in enumerate(self.transformer.h):
            block_out = block(x)
            
            x, aux_loss, f_i = block_out
            
            # Compute intermediate entropy after this layer
            # Apply layer norm and project to vocab to get intermediate predictions
            x_normed = self.transformer.ln_f(x)
            intermediate_logits = self.lm_head(x_normed)
            intermediate_probs = F.softmax(intermediate_logits, dim=-1, dtype=torch.float32)
            epsilon = 1e-10
            layer_entropy = -(intermediate_probs * torch.log(intermediate_probs + epsilon)).sum(dim=-1)
            
            if f_i is not None:
                all_expert_usage.append(f_i)
            
            all_expert_assignments[f'layer_{layer_idx}'] = aux_loss['expert_assignments']
            all_router_logits[f'layer_{layer_idx}'] = aux_loss['router_logits']
            all_router_probs[f'layer_{layer_idx}'] = aux_loss['router_probs']
            all_layer_entropies[f'layer_{layer_idx}'] = layer_entropy
            
            if layer_idx == 0:
                combined_aux_loss = {k: v.clone() for k, v in aux_loss.items() 
                                if k not in ['expert_assignments', 'router_logits', 'router_probs']}
            else:
                for key in aux_loss:
                    if key not in ['expert_assignments', 'router_logits', 'router_probs']:
                        combined_aux_loss[key] += aux_loss[key]
            
            aux_loss_count += 1

        for key in combined_aux_loss:
            combined_aux_loss[key] /= aux_loss_count

        if all_expert_usage:
            avg_expert_usage = torch.stack(all_expert_usage).mean(dim=0)
            combined_aux_loss['expert_usage'] = avg_expert_usage

        combined_aux_loss['expert_assignments'] = all_expert_assignments
        combined_aux_loss['router_logits'] = all_router_logits
        combined_aux_loss['router_probs'] = all_router_probs
        combined_aux_loss['layer_entropies'] = all_layer_entropies

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = ce_loss
            if combined_aux_loss is not None:
                loss = loss + self.config.load_balance_loss_weight * combined_aux_loss['load_balance_loss'] + self.config.router_z_loss_weight * combined_aux_loss['router_z_loss']
                combined_aux_loss['ce_loss'] = ce_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            ce_loss = None

        return logits, loss, combined_aux_loss