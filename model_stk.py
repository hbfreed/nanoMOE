import torch
import torch.nn as nn
import torch.nn.functional as F
import stk.ops
import stk.matrix
from einops import rearrange

# Setup
block_size = 128
m, n = 1024, 2048
hidden_size = 512
num_experts = 64 
num_experts_per_tok = 8
expert_capacity = hidden_size  # Each expert has hidden_size dimension
norm_topk_prob = True
batch_size = 4

router = nn.Linear(hidden_size, num_experts, bias=False, device='cuda')

def route_tokens(x_flat):
    """Route tokens to experts and compute weights."""
    router_logits = router(x_flat)
    router_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    router_weights, selected_experts = torch.topk(router_weights, num_experts_per_tok, dim=-1)
    
    if norm_topk_prob:
        router_weights /= router_weights.sum(dim=-1, keepdim=True)
    
    router_weights = router_weights.to(x_flat.dtype)
    return router_weights, selected_experts, router_logits

def sort_by_expert(x_flat, router_weights, selected_experts):
    """Replicate tokens for each expert and sort by expert assignment."""
    x_rep = x_flat.repeat_interleave(num_experts_per_tok, dim=0)
    selected_experts_rep = selected_experts.reshape(-1)
    router_weights_rep = router_weights.reshape(-1, 1)
    
    expert_sort_indices = torch.argsort(selected_experts_rep, stable=True)
    x_sorted = x_rep[expert_sort_indices]
    selected_experts_sorted = selected_experts_rep[expert_sort_indices]
    router_weights_sorted = router_weights_rep[expert_sort_indices]
    
    # Compute inverse indices for unsort
    inv_expert_sort_indices = torch.empty_like(expert_sort_indices)
    inv_expert_sort_indices[expert_sort_indices] = torch.arange(
        len(expert_sort_indices), device=expert_sort_indices.device
    )
    
    return x_sorted, selected_experts_sorted, router_weights_sorted, inv_expert_sort_indices


def pad_to_blocks( x_sorted, selected_experts_sorted):
    """Pad each expert's tokens to multiples of block_size and track unpadding indices."""
    device = x_sorted.device
    num_tokens = x_sorted.shape[0]
    token_dim = x_sorted.shape[-1]

    # Use self.num_experts and self.block_size directly
    min_tokens_or_experts = min(num_tokens, num_experts)
    max_blocks = min_tokens_or_experts + (num_tokens - min_tokens_or_experts) // block_size
    capacity_tokens = max_blocks * block_size

    # Per-expert counts via scatter_add (compile-safe; avoids bincount)
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.long, device=device)
    ones = torch.ones_like(selected_experts_sorted, dtype=torch.long)
    tokens_per_expert.scatter_add_(0, selected_experts_sorted, ones)

    # Round each expert up to a multiple of block_size
    tokens_per_expert_padded = ((tokens_per_expert + block_size - 1) // block_size) * block_size

    # Exclusive-prefix sums (orig vs padded) for placement
    offset_original = F.pad(tokens_per_expert.cumsum(0), (1, 0))
    offset_padded  = F.pad(tokens_per_expert_padded.cumsum(0), (1, 0))

    # Allocate fixed capacity once; the tail is never indexed
    x_padded = x_sorted.new_zeros((capacity_tokens, token_dim))

    
    # Map each sorted token to its padded position
    token_idx = torch.arange(num_tokens, device=x_sorted.device)
    print(token_idx)
    idx_within_expert = token_idx - offset_original[selected_experts_sorted]
    print(tokens_per_expert_padded)
    unpad_indices = idx_within_expert + offset_padded[selected_experts_sorted]

    # Scatter the actual tokens into their padded slots
    x_padded[unpad_indices] = x_sorted

    # Return exactly what you wanted
    return x_padded, tokens_per_expert_padded, unpad_indices



# Your existing code
x = torch.randn(batch_size, m, hidden_size, device='cuda', requires_grad=True)
x_flat = rearrange(x, 'batch seq hidden -> (batch seq) hidden')

router_weights, selected_experts, router_logits = route_tokens(x_flat)
x_sorted, selected_experts_sorted, router_weights_sorted, inv_indices = sort_by_expert(
    x_flat, router_weights, selected_experts
)

x_padded, tokens_per_expert_padded, unpad_indices = pad_to_blocks(x_sorted, selected_experts_sorted)
total_padded_tokens = int(tokens_per_expert_padded.sum().item())
x_padded = x_padded[:total_padded_tokens]


def make_topology(x_padded: torch.Tensor,
                  tokens_per_expert_padded: torch.Tensor,
                  block_size: int,
                  n: int) -> stk.matrix.Matrix:
    """Construct a block-compressed sparse topology for stk.ops.sdd."""
    if x_padded.dim() != 2:
        raise ValueError("x_padded must be 2D [tokens, hidden].")

    device = x_padded.device
    dtype = x_padded.dtype

    if n % block_size != 0:
        raise ValueError(f"Model width n={n} must be divisible by block_size={block_size}.")

    if int(tokens_per_expert_padded.sum().item()) != x_padded.shape[0]:
        raise ValueError("x_padded row count must equal sum(tokens_per_expert_padded).")

    if x_padded.shape[0] % block_size != 0:
        raise ValueError("Padded rows must be a multiple of block_size for BCSR layout.")

    if torch.any(tokens_per_expert_padded % block_size):
        raise ValueError("Each expert's padded token count must be divisible by block_size.")

    num_row_blocks = x_padded.shape[0] // block_size
    col_blocks = n // block_size

    def _choose_index_dtype(max_value: int) -> torch.dtype:
        if max_value <= torch.iinfo(torch.int16).max:
            return torch.int16
        if max_value <= torch.iinfo(torch.int32).max:
            return torch.int32
        return torch.int64

    max_row_index = max(0, num_row_blocks - 1)
    max_col_index = max(0, col_blocks - 1)
    row_index_dtype = _choose_index_dtype(max_row_index)
    column_index_dtype = _choose_index_dtype(max_col_index)

    row_indices = torch.arange(
        num_row_blocks, device=device, dtype=row_index_dtype
    ).repeat_interleave(col_blocks)
    column_indices = torch.arange(
        col_blocks, device=device, dtype=column_index_dtype
    ).repeat(num_row_blocks)

    offsets = torch.arange(
        num_row_blocks + 1, device=device, dtype=torch.int32
    ) * col_blocks

    num_blocks = num_row_blocks * col_blocks
    data = torch.zeros((num_blocks, block_size, block_size), device=device, dtype=dtype)

    topo = stk.matrix.Matrix(
        size=(x_padded.shape[0], n),
        data=data,
        row_indices=row_indices,
        column_indices=column_indices,
        offsets=offsets,
    )
    return topo


# Create the topology
topology = make_topology(x_padded, tokens_per_expert_padded, block_size, n)


# Now you can use the topology for sparse matmul
# Create expert weights (all experts stacked)
all_expert_weights = torch.randn(
    num_experts * expert_capacity, 
    n, 
    device='cuda',
    requires_grad=True
)

# Sparse matrix multiply with expert weights
w1 = torch.randn(hidden_size, n, device='cuda', requires_grad=True)
w2 = torch.randn(n, hidden_size, device='cuda', requires_grad=True)

result_sparse = stk.ops.sdd(x_padded, w1, topology)
result_sparse = F.gelu(result_sparse)
expert_output= stk.ops.to_dense(result_sparse)


output = expert_output * router_weights_sorted

# Unpermute back to original token order
output = output[inv_indices]

num_tokens = batch_size * seq_len

original_token_indices = (
    torch.arange(num_tokens * num_experts_per_tok, device=output.device)
    // num_experts_per_tok
)

combined_output = torch.zeros((num_tokens, n_embd), 
                                dtype=output.dtype, device=output.device)
combined_output.scatter_add_(
    0,
    original_token_indices.unsqueeze(-1).expand(-1, n_embd),
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

    f_i = torch.zeros(num_experts, dtype=x.dtype, device=x.device)
    ones = torch.ones_like(experts_flat, dtype=x.dtype) / len(experts_flat)
    f_i.scatter_add_(0, experts_flat, ones)
    load_balance_loss = num_experts * (f_i @ p_i)
else:
    # No tokens routed - no imbalance possible
    load_balance_loss = torch.tensor(0.0, device=x.device)
    f_i = torch.zeros(num_experts, device=x.device)

# Package auxiliary losses for the training script
aux_loss = {
    'router_z_loss': router_z_loss,
    'load_balance_loss': load_balance_loss,
}

print(f"Result shape: {result.shape}")
