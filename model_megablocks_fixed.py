"""
Fixed MoEMLPMegaBlocks implementation with correct softmax ordering
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import stk
from megablocks import ops
from einops import rearrange

@torch.compiler.disable
class MoeMLPMegaBlocksFixed(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_embd = config.n_embd
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)

        # FIX: Ensure d_ffn is divisible by blocking (128) to avoid Triton kernel issues
        self.blocking = 128
        base_d_ffn = 4 * self.n_embd // self.num_experts_per_tok

        # Round up d_ffn to nearest multiple of blocking
        self.d_ffn = ((base_d_ffn + self.blocking - 1) // self.blocking) * self.blocking

        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)

        self.w1 = nn.Parameter(torch.empty(self.n_embd, self.num_experts * self.d_ffn))
        self.w2 = nn.Parameter(torch.empty(self.num_experts * self.d_ffn, self.n_embd))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=0.02, a=-0.06, b=0.06)

        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))),1)

        max_column_index = (self.d_ffn * self.num_experts) // self.blocking

        self.transpose_sort_end_bit = max(int(np.ceil(np.log2(max_column_index))),1)

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape

        x_flat = rearrange(x, 'batch_size seq_len n_embd -> (batch_size seq_len) n_embd')  # [batch*seq, n_embd]

        router_logits = self.router(x_flat)

        # FIX: Apply softmax BEFORE topk to match STK behavior
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)  # Use float32 for stability

        # Select top-k experts based on probabilities
        expert_weights, selected_experts = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)

        # Normalize the top-k weights to sum to 1
        if self.norm_topk_prob:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Convert back to model dtype
        expert_weights = expert_weights.to(x.dtype)

        expert_weights_flat = rearrange(expert_weights, '... -> (...)')
        selected_experts_flat = rearrange(selected_experts, '... -> (...)')

        if selected_experts_flat.numel() == 0:
            output = torch.zeros((batch_size, seq_len, self.n_embd), dtype=x.dtype, device=x.device)
            aux_loss = {
                'router_z_loss': torch.tensor(0.0, device=x.device),
                'load_balance_loss': torch.tensor(0.0, device=x.device),
            }
            f_i = torch.zeros(self.num_experts, device=x.device)
            return output, aux_loss, f_i

        bin_ids, indices, tokens_per_expert = self._sort_tokens_by_expert(selected_experts_flat)

        if tokens_per_expert.sum() == 0:
            output = torch.zeros((batch_size, seq_len, self.n_embd), dtype=x.dtype, device=x.device)
            aux_loss = {
                'router_z_loss': torch.tensor(0.0, device=x.device),
                'load_balance_loss': torch.tensor(0.0, device=x.device),
            }
            f_i = torch.zeros(self.num_experts, device=x.device)
            return output, aux_loss, f_i

        padded_bins, topology = self._create_topology(x_flat, tokens_per_expert)

        x_permuted = self._gather_tokens(x_flat, indices, bin_ids, tokens_per_expert, padded_bins)

        if x_permuted.shape[0] != topology.shape[0]:
            if x_permuted.shape[0] < topology.shape[0]:
                padding = torch.zeros((topology.shape[0] - x_permuted.shape[0], x_permuted.shape[1]),
                                     dtype=x_permuted.dtype, device=x_permuted.device)
                x_permuted = torch.cat([x_permuted, padding], dim=0)
            else:
                x_permuted = x_permuted[:topology.shape[0]]

        x_permuted = stk.ops.sdd(x_permuted, self.w1, topology)

        x_permuted = F.gelu(x_permuted.data)

        x_permuted = stk.ops.dsd(
            stk.Matrix(topology.shape, x_permuted, *self._get_topo_tensors(topology)),
            self.w2
        )

        x_permuted = self._scatter_tokens(
            x_permuted, indices, bin_ids, expert_weights_flat, tokens_per_expert, padded_bins
        )

        output = rearrange(
            x_permuted,
            '(batch_size seq_len) n_embd -> batch_size seq_len n_embd',
            batch_size=batch_size,
            seq_len=seq_len
        )

        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean() # router z-loss keeps router logits small

        p_i = F.softmax(router_logits, dim=-1)  # Shape: (batch*seq, num_experts)
        p_i = p_i.mean(dim=0)  # Average probability per expert: (num_experts,)

        if len(selected_experts) > 0:
            experts_flat = selected_experts.flatten()
            f_i = torch.zeros(self.num_experts, dtype=x.dtype, device=x.device)
            ones = torch.ones_like(experts_flat, dtype=x.dtype) / len(experts_flat)
            f_i.scatter_add_(0, experts_flat, ones)
            load_balance_loss = self.num_experts * (f_i @ p_i)
        else:
            load_balance_loss = torch.tensor(0.0, device=x.device)
            f_i = torch.zeros(self.num_experts, device=x.device)

        aux_loss = {
            'router_z_loss': router_z_loss,
            'load_balance_loss': load_balance_loss,
        }

        return output, aux_loss, f_i

    def _sort_tokens_by_expert(self, selected_experts_flat):
        """Group token assignments by expert id."""

        bin_ids, indices = ops.sort(selected_experts_flat, self.sort_end_bit)
        tokens_per_expert = ops.histogram(selected_experts_flat, self.num_experts)

        return bin_ids, indices, tokens_per_expert

    def _create_topology(self, x, tokens_per_expert):
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = padded_bins.contiguous()

        if padded_bins.numel() == 0:
            padded_bins = torch.zeros(1, dtype=torch.int32, device=x.device)
        elif padded_bins.dim() == 0:
            padded_bins = padded_bins.unsqueeze(0)

        # FIX: Ensure padded_tokens is at least blocking size to avoid empty topology
        padded_tokens = padded_bins[-1].item() if padded_bins.numel() > 0 else self.blocking
        padded_tokens = max(padded_tokens, self.blocking)  # Ensure minimum size

        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.d_ffn // self.blocking  # Now guaranteed to be integer

        column_indices = ops.topology(
            padded_bins, self.blocking, block_rows, blocks_per_row
        )

        offsets = torch.arange(
            0, block_rows * blocks_per_row + 1, blocks_per_row,
            dtype=torch.int32, device=x.device
        )

        column_indices = column_indices.to(torch.int32)
        offsets = offsets.to(torch.int32)

        shape = (padded_tokens, self.d_ffn * self.num_experts)

        # FIX: Handle empty topology case properly
        num_blocks = column_indices.numel()
        if num_blocks > 0:
            data_placeholder = torch.empty(
                num_blocks, self.blocking, self.blocking,
                dtype=x.dtype, device='meta'
            )
        else:
            # Create minimal valid topology for empty case
            data_placeholder = torch.empty(
                1, self.blocking, self.blocking,
                dtype=x.dtype, device='meta'
            )
            column_indices = torch.zeros(1, dtype=torch.int32, device=x.device)

        row_indices = stk.ops.row_indices(shape, data_placeholder, offsets, column_indices)
        row_indices = row_indices.to(torch.int32)

        column_indices_t, offsets_t, block_offsets_t = self._sparse_transpose(
            shape, row_indices, column_indices, offsets, padded_tokens_per_expert
        )
        column_indices_t = column_indices_t.to(torch.int32)
        offsets_t = offsets_t.to(torch.int32)
        block_offsets_t = column_indices_t.to(torch.int32)

        topology = stk.Matrix(
            shape, data_placeholder, row_indices, column_indices,
            offsets, column_indices_t, offsets_t, block_offsets_t
        )

        return padded_bins, topology

    def _sparse_transpose(self, shape, row_indices, column_indices, offsets, padded_tokens_per_expert):
        block_columns = shape[1] // self.blocking

        # FIX: Handle empty topology case
        if column_indices.numel() == 0:
            zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
            return column_indices, torch.cat([zero, zero]), column_indices

        _, gather_indices = ops.sort(column_indices.int(), self.transpose_sort_end_bit)

        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        nnz_per_column = nnz_per_column.contiguous()
        if nnz_per_column.dim() == 0:
            nnz_per_column = nnz_per_column.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_column])

        return column_indices_t, offsets_t, block_offsets_t

    def _gather_tokens(self, x, indices, bin_ids, tokens_per_expert, padded_bins):
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.contiguous()
        if bins.numel() == 0:
            bins = torch.zeros(1, dtype=torch.int32, device=x.device)
        elif bins.dim() == 0:
            bins = bins.unsqueeze(0)

        return ops.padded_gather(
            x,
            indices,
            bin_ids,
            bins,
            padded_bins,
            self.num_experts_per_tok
        )

    def _scatter_tokens(self, x, indices, bin_ids, weights, tokens_per_expert, padded_bins):
        """Un-permute tokens and apply expert weights."""
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.contiguous()
        if bins.numel() == 0:
            bins = torch.zeros(1, dtype=torch.int32, device=x.device)
        elif bins.dim() == 0:
            bins = bins.unsqueeze(0)

        return ops.padded_scatter(
            x,
            indices,
            bin_ids,
            weights,
            bins,
            padded_bins,
            self.num_experts_per_tok
        )

    def _get_topo_tensors(self, topology):
        """Extract the tensor components from STK Matrix for reconstruction."""
        return (
            topology.row_indices,
            topology.column_indices,
            topology.offsets,
            topology.column_indices_t,
            topology.offsets_t,
            topology.block_offsets_t
        )