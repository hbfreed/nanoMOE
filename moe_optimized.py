import torch
import triton
import triton.language as tl
import triton.testing
import os
import math

# Enable verbose Triton autotuning output unless explicitly disabled by the environment
if os.environ.get("TRITON_PRINT_AUTOTUNING") is None:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
# @triton.autotune(
#     configs=[
#         # Small blocks for small batch sizes or debugging
#         triton.Config({'BLOCK_SIZE': 16, 'BLOCK_K': 16}),
#         triton.Config({'BLOCK_SIZE': 16, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 32, 'BLOCK_K': 16}),
#         triton.Config({'BLOCK_SIZE': 32, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 32, 'BLOCK_K': 64}),
#         # Medium blocks
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 16}),
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 64}),
#         # Removed larger configs that exceed RTX 3090 shared memory (101KB)
#     ],
#     key=['hidden_size', 'd_ffn'],
# )

@triton.jit
def sdd_kernel(
    x_ptr, w1_ptr, output_ptr,
    m_block_to_expert, n_block_to_expert,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    # dynamic sizes
    batch_size, hidden_size, output_width,
    num_m_blocks, num_n_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Early guard: avoid OOB scalar loads from mapping arrays
    if (pid_m >= num_m_blocks) or (pid_n >= num_n_blocks):
        return
    
    expert_m = tl.load(m_block_to_expert + pid_m)
    expert_n = tl.load(n_block_to_expert + pid_n)
    
    if expert_m != expert_n:
        return
    
    m_start = pid_m * BLOCK_SIZE
    n_start = pid_n * BLOCK_SIZE
    if (m_start >= batch_size) or (n_start >= hidden_size):
        return
    # If this tile starts beyond matrix bounds, skip entirely
    if (m_start >= batch_size) or (n_start >= output_width):
        return
    
    row_offsets = tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    m_offsets = m_start + row_offsets
    n_offsets = n_start + col_offsets
    m_offsets_64 = m_offsets.to(tl.int64)
    n_offsets_64 = n_offsets.to(tl.int64)

    stride_xm_64 = tl.full([], stride_xm, dtype=tl.int64)
    stride_xk_64 = tl.full([], stride_xk, dtype=tl.int64)
    stride_wk_64 = tl.full([], stride_wk, dtype=tl.int64)
    stride_wn_64 = tl.full([], stride_wn, dtype=tl.int64)
    stride_om_64 = tl.full([], stride_om, dtype=tl.int64)
    stride_on_64 = tl.full([], stride_on, dtype=tl.int64)

    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, hidden_size, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_offsets_64 = k_offsets.to(tl.int64)
        
        # Mask for batch_size and hidden_size
        x_ptrs = x_ptr + m_offsets_64[:, None] * stride_xm_64 + k_offsets_64[None, :] * stride_xk_64
        x = tl.load(x_ptrs, mask=(m_offsets[:, None] < batch_size) & (k_offsets[None, :] < hidden_size))
        
        # Mask for hidden_size and output_width
        w_ptrs = w1_ptr + k_offsets_64[:, None] * stride_wk_64 + n_offsets_64[None, :] * stride_wn_64
        w = tl.load(w_ptrs, mask=(k_offsets[:, None] < hidden_size) & (n_offsets[None, :] < output_width))
        
        accumulator += tl.dot(x, w)
    
    # Mask for batch_size and output_width when storing
    output_ptrs = output_ptr + m_offsets_64[:, None] * stride_om_64 + n_offsets_64[None, :] * stride_on_64
    mask = (m_offsets[:, None] < batch_size) & (n_offsets[None, :] < output_width)
    tl.store(output_ptrs, accumulator, mask=mask)

# @triton.autotune(
#     configs=[
#         # Small blocks for small batch sizes or debugging
#         triton.Config({'BLOCK_SIZE': 16, 'BLOCK_K': 16}),
#         triton.Config({'BLOCK_SIZE': 16, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 32, 'BLOCK_K': 16}),
#         triton.Config({'BLOCK_SIZE': 32, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 32, 'BLOCK_K': 64}),
#         # Medium blocks
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 16}),
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 64}),
#         # Removed larger configs that exceed RTX 3090 shared memory (101KB)
#     ],
#     key=['d_ffn', 'hidden_size'],
# )

@triton.jit
def dsd_kernel(
    x_ptr, w2_ptr, output_ptr,
    m_block_to_expert, n_block_to_expert,
    weight_offsets,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    batch_size, d_ffn, hidden_size, output_width,
    num_m_blocks, num_n_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Early guard before any scalar mapping loads
    if (pid_m >= num_m_blocks) or (pid_n >= num_n_blocks):
        return
    
    # Compute tile origin and check page bounds first
    m_start = pid_m * BLOCK_SIZE
    n_start = pid_n * BLOCK_SIZE
    if (m_start >= batch_size) or (n_start >= hidden_size):
        return
    
    # Safe to read mapping now
    expert_m = tl.load(m_block_to_expert + pid_m)
    
    row_offsets = tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    m_offsets = m_start + row_offsets
    n_offsets = n_start + col_offsets
    m_offsets_64 = m_offsets.to(tl.int64)
    n_offsets_64 = n_offsets.to(tl.int64)

    stride_xm_64 = tl.full([], stride_xm, dtype=tl.int64)
    stride_xk_64 = tl.full([], stride_xk, dtype=tl.int64)
    stride_wk_64 = tl.full([], stride_wk, dtype=tl.int64)
    stride_wn_64 = tl.full([], stride_wn, dtype=tl.int64)
    stride_om_64 = tl.full([], stride_om, dtype=tl.int64)
    stride_on_64 = tl.full([], stride_on, dtype=tl.int64)

    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Determine expert slice [ffn_start, ffn_end) and clamp to [0, d_ffn]
    ffn_start = tl.load(weight_offsets + expert_m)
    ffn_end = tl.load(weight_offsets + expert_m + 1)
    if ffn_start < 0:
        ffn_start = 0
    if ffn_end > d_ffn:
        ffn_end = d_ffn
    if ffn_end <= ffn_start:
        return
    
    # Iterate only over this expert's K-range
    for k in range(ffn_start, ffn_end, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_offsets_64 = k_offsets.to(tl.int64)

        # Load X within [ffn_start, ffn_end) and within d_ffn bounds
        x_ptrs = x_ptr + m_offsets_64[:, None] * stride_xm_64 + k_offsets_64[None, :] * stride_xk_64
        x_mask = (m_offsets[:, None] < batch_size) & (k_offsets[None, :] < ffn_end) & (k_offsets[None, :] < d_ffn)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load corresponding rows of W2
        w_ptrs = w2_ptr + k_offsets_64[:, None] * stride_wk_64 + n_offsets_64[None, :] * stride_wn_64
        w_mask = (k_offsets[:, None] < ffn_end) & (k_offsets[:, None] < d_ffn) & (n_offsets[None, :] < hidden_size)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        accumulator += tl.dot(x, w)

    output_ptrs = output_ptr + m_offsets_64[:, None] * stride_om_64 + n_offsets_64[None, :] * stride_on_64
    mask = (m_offsets[:, None] < batch_size) & (n_offsets[None, :] < hidden_size)
    tl.store(output_ptrs, accumulator, mask=mask)

@triton.jit
def sdd_backward_act_kernel(
    grad_sparse_ptr,
    w1_t_ptr,
    grad_input_ptr,
    m_block_to_expert_ptr,
    weight_offsets_ptr,
    stride_gm, stride_gk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    batch_size,
    hidden_size,
    d_ffn,
    num_token_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_m >= num_token_blocks:
        return

    expert_id = tl.load(m_block_to_expert_ptr + pid_m)
    ffn_start = tl.load(weight_offsets_ptr + expert_id)
    ffn_end = tl.load(weight_offsets_ptr + expert_id + 1)
    if ffn_start >= ffn_end:
        return

    token_base = pid_m * BLOCK_SIZE
    if token_base >= batch_size:
        return

    hidden_base = pid_h * BLOCK_SIZE
    if hidden_base >= hidden_size:
        return

    token_offsets = token_base + tl.arange(0, BLOCK_SIZE)
    hidden_offsets = hidden_base + tl.arange(0, BLOCK_SIZE)
    token_offsets_64 = token_offsets.to(tl.int64)
    hidden_offsets_64 = hidden_offsets.to(tl.int64)

    stride_gm_64 = tl.full([], stride_gm, dtype=tl.int64)
    stride_gk_64 = tl.full([], stride_gk, dtype=tl.int64)
    stride_wk_64 = tl.full([], stride_wk, dtype=tl.int64)
    stride_wn_64 = tl.full([], stride_wn, dtype=tl.int64)
    stride_om_64 = tl.full([], stride_om, dtype=tl.int64)
    stride_on_64 = tl.full([], stride_on, dtype=tl.int64)

    token_mask = token_offsets < batch_size
    hidden_mask = hidden_offsets < hidden_size

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(ffn_start, ffn_end, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = (k_offsets < ffn_end) & (k_offsets < d_ffn)
        k_offsets_64 = k_offsets.to(tl.int64)

        grad_ptrs = grad_sparse_ptr + token_offsets_64[:, None] * stride_gm_64 + k_offsets_64[None, :] * stride_gk_64
        grad_tile = tl.load(grad_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)

        w1_ptrs = w1_t_ptr + k_offsets_64[:, None] * stride_wk_64 + hidden_offsets_64[None, :] * stride_wn_64
        w1_tile = tl.load(w1_ptrs, mask=k_mask[:, None] & hidden_mask[None, :], other=0.0)

        acc += tl.dot(grad_tile, w1_tile)

    out_ptrs = grad_input_ptr + token_offsets_64[:, None] * stride_om_64 + hidden_offsets_64[None, :] * stride_on_64
    tl.store(out_ptrs, acc, mask=token_mask[:, None] & hidden_mask[None, :])

@triton.jit
def sdd_backward_weight_kernel(
    input_t_ptr,
    grad_sparse_ptr,
    grad_weight_ptr,
    m_block_to_expert_ptr,
    weight_offsets_ptr,
    stride_xm, stride_xk,
    stride_gm, stride_gk,
    stride_wm, stride_wn,
    batch_size,
    hidden_size,
    d_ffn,
    num_token_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid_blocks = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_blocks >= num_token_blocks:
        return

    expert_id = tl.load(m_block_to_expert_ptr + pid_blocks)
    ffn_start = tl.load(weight_offsets_ptr + expert_id)
    ffn_end = tl.load(weight_offsets_ptr + expert_id + 1)
    if ffn_start >= ffn_end:
        return

    token_base = pid_blocks * BLOCK_SIZE
    if token_base >= batch_size:
        return

    hidden_base = pid_h * BLOCK_SIZE
    if hidden_base >= hidden_size:
        return

    token_offsets = token_base + tl.arange(0, BLOCK_SIZE)
    hidden_offsets = hidden_base + tl.arange(0, BLOCK_SIZE)
    token_offsets_64 = token_offsets.to(tl.int64)
    hidden_offsets_64 = hidden_offsets.to(tl.int64)

    token_mask = token_offsets < batch_size
    hidden_mask = hidden_offsets < hidden_size

    stride_xm_64 = tl.full([], stride_xm, dtype=tl.int64)
    stride_xk_64 = tl.full([], stride_xk, dtype=tl.int64)
    stride_gm_64 = tl.full([], stride_gm, dtype=tl.int64)
    stride_gk_64 = tl.full([], stride_gk, dtype=tl.int64)
    stride_wm_64 = tl.full([], stride_wm, dtype=tl.int64)
    stride_wn_64 = tl.full([], stride_wn, dtype=tl.int64)

    input_row_offsets = hidden_offsets_64[:, None]
    input_col_offsets = token_offsets_64[None, :]
    input_ptrs = input_t_ptr + input_row_offsets * stride_xm_64 + input_col_offsets * stride_xk_64
    input_tile = tl.load(input_ptrs, mask=hidden_mask[:, None] & token_mask[None, :], other=0.0)

    for k in range(ffn_start, ffn_end, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        k_mask = (k_offsets < ffn_end) & (k_offsets < d_ffn)
        k_offsets_64 = k_offsets.to(tl.int64)

        grad_row_offsets = token_offsets_64[:, None]
        grad_col_offsets = k_offsets_64[None, :]
        grad_ptrs = grad_sparse_ptr + grad_row_offsets * stride_gm_64 + grad_col_offsets * stride_gk_64
        grad_tile = tl.load(grad_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)

        result = tl.dot(input_tile, grad_tile)

        weight_row_offsets = hidden_offsets_64[:, None]
        weight_col_offsets = k_offsets_64[None, :]
        weight_ptrs = grad_weight_ptr + weight_row_offsets * stride_wm_64 + weight_col_offsets * stride_wn_64
        weight_mask = hidden_mask[:, None] & k_mask[None, :]
        tl.atomic_add(weight_ptrs, result, mask=weight_mask)

@triton.jit
def dsd_backward_act_kernel(
    grad_output_ptr,
    w2_t_ptr,
    grad_act_ptr,
    m_block_to_expert_ptr,
    weight_offsets_ptr,
    stride_gm, stride_gk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    batch_size,
    hidden_size,
    d_ffn,
    num_token_blocks,
    max_ffn_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_ffn = tl.program_id(1)

    if pid_block >= num_token_blocks:
        return

    expert_id = tl.load(m_block_to_expert_ptr + pid_block)
    ffn_start = tl.load(weight_offsets_ptr + expert_id)
    ffn_end = tl.load(weight_offsets_ptr + expert_id + 1)
    if ffn_start >= ffn_end:
        return

    token_base = pid_block * BLOCK_SIZE
    if token_base >= batch_size:
        return

    ffn_blocks = (ffn_end - ffn_start + BLOCK_SIZE - 1) // BLOCK_SIZE
    if pid_ffn >= ffn_blocks:
        return

    col_base = ffn_start + pid_ffn * BLOCK_SIZE

    token_offsets = token_base + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_base + tl.arange(0, BLOCK_SIZE)
    token_offsets_64 = token_offsets.to(tl.int64)
    col_offsets_64 = col_offsets.to(tl.int64)

    token_mask = token_offsets < batch_size
    col_mask = (col_offsets < ffn_end) & (col_offsets < d_ffn)

    stride_gm_64 = tl.full([], stride_gm, dtype=tl.int64)
    stride_gk_64 = tl.full([], stride_gk, dtype=tl.int64)
    stride_wk_64 = tl.full([], stride_wk, dtype=tl.int64)
    stride_wn_64 = tl.full([], stride_wn, dtype=tl.int64)
    stride_om_64 = tl.full([], stride_om, dtype=tl.int64)
    stride_on_64 = tl.full([], stride_on, dtype=tl.int64)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, hidden_size, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < hidden_size
        k_offsets_64 = k_offsets.to(tl.int64)

        grad_row_offsets = token_offsets_64[:, None]
        grad_col_offsets = k_offsets_64[None, :]
        grad_ptrs = grad_output_ptr + grad_row_offsets * stride_gm_64 + grad_col_offsets * stride_gk_64
        grad_tile = tl.load(grad_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)

        w2_row_offsets = k_offsets_64[:, None]
        w2_col_offsets = col_offsets_64[None, :]
        w2_ptrs = w2_t_ptr + w2_row_offsets * stride_wk_64 + w2_col_offsets * stride_wn_64
        w2_tile = tl.load(w2_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)

        acc += tl.dot(grad_tile, w2_tile)

    out_row_offsets = token_offsets_64[:, None]
    out_col_offsets = col_offsets_64[None, :]
    out_ptrs = grad_act_ptr + out_row_offsets * stride_om_64 + out_col_offsets * stride_on_64
    tl.store(out_ptrs, acc, mask=token_mask[:, None] & col_mask[None, :])


@triton.jit
def dsd_backward_weight_kernel(
    sparse_act_t_ptr,
    grad_output_ptr,
    grad_weight_ptr,
    m_block_to_expert_ptr,
    weight_offsets_ptr,
    stride_am, stride_ak,
    stride_gm, stride_gk,
    stride_wm, stride_wn,
    batch_size,
    hidden_size,
    d_ffn,
    num_token_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid_blocks = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_blocks >= num_token_blocks:
        return

    expert_id = tl.load(m_block_to_expert_ptr + pid_blocks)
    ffn_start = tl.load(weight_offsets_ptr + expert_id)
    ffn_end = tl.load(weight_offsets_ptr + expert_id + 1)
    if ffn_start >= ffn_end:
        return

    token_base = pid_blocks * BLOCK_SIZE
    if token_base >= batch_size:
        return

    hidden_base = pid_h * BLOCK_SIZE
    if hidden_base >= hidden_size:
        return

    token_offsets = token_base + tl.arange(0, BLOCK_SIZE)
    hidden_offsets = hidden_base + tl.arange(0, BLOCK_SIZE)
    token_offsets_64 = token_offsets.to(tl.int64)
    hidden_offsets_64 = hidden_offsets.to(tl.int64)

    token_mask = token_offsets < batch_size
    hidden_mask = hidden_offsets < hidden_size

    stride_am_64 = tl.full([], stride_am, dtype=tl.int64)
    stride_ak_64 = tl.full([], stride_ak, dtype=tl.int64)
    stride_gm_64 = tl.full([], stride_gm, dtype=tl.int64)
    stride_gk_64 = tl.full([], stride_gk, dtype=tl.int64)
    stride_wm_64 = tl.full([], stride_wm, dtype=tl.int64)
    stride_wn_64 = tl.full([], stride_wn, dtype=tl.int64)

    grad_row_offsets = token_offsets_64[:, None]
    grad_col_offsets = hidden_offsets_64[None, :]
    grad_ptrs = grad_output_ptr + grad_row_offsets * stride_gm_64 + grad_col_offsets * stride_gk_64
    grad_tile = tl.load(grad_ptrs, mask=token_mask[:, None] & hidden_mask[None, :], other=0.0)

    for k in range(ffn_start, ffn_end, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        k_mask = (k_offsets < ffn_end) & (k_offsets < d_ffn)
        k_offsets_64 = k_offsets.to(tl.int64)

        act_row_offsets = k_offsets_64[:, None]
        act_col_offsets = token_offsets_64[None, :]
        act_ptrs = sparse_act_t_ptr + act_row_offsets * stride_am_64 + act_col_offsets * stride_ak_64
        act_tile = tl.load(act_ptrs, mask=k_mask[:, None] & token_mask[None, :], other=0.0)

        result = tl.dot(act_tile, grad_tile)

        weight_row_offsets = k_offsets_64[:, None]
        weight_col_offsets = hidden_offsets_64[None, :]
        weight_ptrs = grad_weight_ptr + weight_row_offsets * stride_wm_64 + weight_col_offsets * stride_wn_64
        weight_mask = k_mask[:, None] & hidden_mask[None, :]
        tl.atomic_add(weight_ptrs, result, mask=weight_mask)


@triton.jit
def gelu(x):
    '''gelu based on https://arxiv.org/pdf/1606.08415#page=2'''
    pass

@triton.jit
def approx_gelu(x):
    '''approximated gelu based on https://arxiv.org/pdf/1606.08415#page=2'''
    pass

@triton.jit
def gelu_sigmoid(x):
    """GELU with casting for bfloat16 compatibility"""
    x_fp32 = x.to(tl.float32)
    result = x_fp32 * tl.sigmoid(1.702 * x_fp32)
    return result.to(x.dtype)  # Cast back to original dtype

@triton.jit
def tanh(x):
    pass


class SDD(torch.autograd.Function):
    """Sparse-Dense-Dense operation for MoE first layer (X @ W1)
    
    Computes Y = X @ W1 where:
    - X is dense input (batch_tokens, hidden_size)
    - W1 is dense weights (hidden_size, num_experts * d_ffn)
    - Y is sparse output stored compactly based on expert assignment
    """
    @staticmethod
    def forward(ctx, x, w1, m_block_to_expert, n_block_to_expert, weight_offsets, block_size=64):
        batch_size = x.shape[0]
        hidden_size = x.shape[1]  # Input dimension
        output_width = w1.shape[1]  # Total concatenated FFN size
        
        # Allocate output tensor (compact storage)
        output = torch.zeros((batch_size, output_width), dtype=x.dtype, device=x.device)
        
        total_m_blocks = (batch_size + block_size - 1) // block_size  # Ceiling division!
        total_n_blocks = (output_width + block_size - 1) // block_size  # Ceiling division!
        
        # Launch kernel with dimensions
        m_blocks = int(m_block_to_expert.numel())
        n_blocks = int(n_block_to_expert.numel())
        # Be conservative: don't exceed physical tile counts
        grid = (min(m_blocks, total_m_blocks), min(n_blocks, total_n_blocks))
        # Debug (kept lightweight): helps catch grid/mapping mismatches
        # print(f"SDD: x {tuple(x.shape)} | w1 {tuple(w1.shape)} | out {tuple(output.shape)} | m_blocks {m_blocks} n_blocks {n_blocks} | grid {grid} | bs {block_size}")
        active_m_blocks, active_n_blocks = grid
        sdd_kernel[grid](
            x, w1, output,
            m_block_to_expert, n_block_to_expert,
            x.stride(0), x.stride(1),
            w1.stride(0), w1.stride(1),
            output.stride(0), output.stride(1),
            batch_size, hidden_size, output_width,
            active_m_blocks, active_n_blocks,
            BLOCK_SIZE=block_size,
            BLOCK_K=block_size,
        )
        
        # Save for backward
        ctx.save_for_backward(x, w1, m_block_to_expert, weight_offsets)
        ctx.block_size = block_size
        ctx.batch_size = batch_size
        ctx.hidden_size = hidden_size
        ctx.output_width = output_width
        ctx.d_ffn = output_width
        ctx.num_token_blocks = int(m_block_to_expert.numel())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes gradients for x and w1 given grad_output
        
        Args:
            grad_output: Gradient w.r.t. sparse output (num_padded_tokens, d_ffn_compact)
        
        Returns:
            Gradients for: x, w1, m_block_to_expert, n_block_to_expert, weight_offsets, block_size
        """
        # Retrieve saved tensors and metadata
        x, w1, m_block_to_expert, weight_offsets = ctx.saved_tensors
        block_size = ctx.block_size
        hidden_size = ctx.hidden_size
        d_ffn = ctx.d_ffn
        num_token_blocks = ctx.num_token_blocks

        x_t, w1_t = x.t(), w1.t()

        grad_x = torch.zeros_like(x)
        if num_token_blocks == 0:
            return grad_x, torch.zeros_like(w1), None, None, None, None

        hidden_tiles = math.ceil(hidden_size / block_size)
        x_grid = (num_token_blocks, hidden_tiles)
        w1_grid = (num_token_blocks, hidden_tiles)

        grad_w1 = torch.zeros(w1.shape, dtype=torch.float32, device=w1.device)

        sdd_backward_act_kernel[x_grid](
            grad_output,
            w1_t,
            grad_x,
            m_block_to_expert,
            weight_offsets,
            grad_output.stride(0), grad_output.stride(1),
            w1_t.stride(0), w1_t.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            ctx.batch_size,
            hidden_size,
            d_ffn,
            num_token_blocks,
            BLOCK_SIZE=block_size,
            BLOCK_K=min(block_size, d_ffn),
        )

        sdd_backward_weight_kernel[w1_grid](
            x_t,
            grad_output,
            grad_w1,
            m_block_to_expert,
            weight_offsets,
            x_t.stride(0), x_t.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_w1.stride(0), grad_w1.stride(1),
            ctx.batch_size,
            hidden_size,
            d_ffn,
            num_token_blocks,
            BLOCK_SIZE=block_size,
        )

        grad_w1 = grad_w1.to(w1.dtype)

        return grad_x, grad_w1, None, None, None, None


class DSD(torch.autograd.Function):
    """Dense-Sparse-Dense operation for MoE second layer (X @ W2)
    
    Computes Y = X @ W2 where:
    - X is sparse input in compact form (batch_tokens, d_ffn_compact)
    - W2 is dense weights (num_experts * d_ffn, hidden_size)
    - Y is dense output (batch_tokens, hidden_size)
    """
    
    @staticmethod
    def forward(ctx, x, w2, m_block_to_expert, n_block_to_expert, weight_offsets, block_size=64, max_ffn_blocks_per_expert=None):        
        """
        Args:
            x: Sparse input tensor in compact form (num_padded_tokens, d_ffn_compact)
            w2: Dense weight tensor (num_experts * d_ffn, hidden_size)
            m_block_to_expert: Which expert owns each token block
            n_block_to_expert: Not really used in DSD but kept for consistency
            weight_offsets: Where each expert's weights start
            block_size: Block size for Triton kernel
        
        Returns:
            Dense output (num_padded_tokens, hidden_size)
        """
        # Enforce standard contiguous layout for predictable strides
        if not x.is_contiguous():
            x = x.contiguous()
        if not w2.is_contiguous():
            w2 = w2.contiguous()
        batch_size = x.shape[0]
        d_ffn = x.shape[1]
        hidden_size = w2.shape[1]
        output_width = w2.shape[0]  # Total concatenated FFN size

        
        # Allocate dense output
        output = torch.zeros((batch_size, hidden_size), dtype=x.dtype, device=x.device)
        
        # Size grid by mapping arrays to avoid OOB accesses
        m_blocks = int(m_block_to_expert.numel())
        n_blocks = (hidden_size + block_size - 1) // block_size
        grid = (m_blocks, n_blocks)

        dsd_kernel[grid](
            x, w2, output,
            m_block_to_expert, n_block_to_expert,
            weight_offsets,
            x.stride(0), x.stride(1),
            w2.stride(0), w2.stride(1),
            output.stride(0), output.stride(1),
            batch_size, d_ffn, hidden_size, output_width,
            m_blocks, n_blocks,
            BLOCK_SIZE=block_size,
            BLOCK_K=min(block_size, d_ffn),
        )
        
        ctx.save_for_backward(x, w2, m_block_to_expert, weight_offsets)
        ctx.block_size = block_size
        ctx.hidden_size = hidden_size
        ctx.d_ffn = d_ffn
        ctx.batch_size = batch_size
        ctx.num_token_blocks = int(m_block_to_expert.numel())
        if max_ffn_blocks_per_expert is None:
            if weight_offsets.numel() > 1:
                per_expert_widths = weight_offsets[1:] - weight_offsets[:-1]
                max_ffn_blocks_per_expert = int(
                    ((per_expert_widths + block_size - 1) // block_size).max().item()
                )
            else:
                max_ffn_blocks_per_expert = 0
        ctx.max_ffn_blocks = max(max_ffn_blocks_per_expert, 1)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes gradients for x and w2 given grad_output
        
        Args:
            grad_output: Gradient w.r.t. dense output (num_padded_tokens, hidden_size)
        
        Returns:
            Gradients for: x, w2, m_block_to_expert, n_block_to_expert, weight_offsets, block_size
        """
        # Retrieve saved tensors
        x, w2, m_block_to_expert, weight_offsets = ctx.saved_tensors
        block_size = ctx.block_size
        hidden_size = ctx.hidden_size
        d_ffn = ctx.d_ffn
        batch_size = ctx.batch_size
        num_token_blocks = ctx.num_token_blocks
        max_ffn_blocks = max(ctx.max_ffn_blocks, 1)

        x_t = x.t()
        w2_t = w2.t()

        grad_x = torch.zeros_like(x)
        if num_token_blocks == 0:
            return grad_x, torch.zeros_like(w2), None, None, None, None

        hidden_tiles = math.ceil(hidden_size / block_size)

        x_grid = (num_token_blocks, max_ffn_blocks)
        w2_grid = (num_token_blocks, hidden_tiles)

        grad_w2 = torch.zeros(w2.shape, dtype=torch.float32,device=w2.device)

        dsd_backward_act_kernel[x_grid](
            grad_output,
            w2_t,
            grad_x,
            m_block_to_expert,
            weight_offsets,
            grad_output.stride(0), grad_output.stride(1),
            w2_t.stride(0), w2_t.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            batch_size,
            hidden_size,
            d_ffn,
            num_token_blocks,
            max_ffn_blocks,
            BLOCK_SIZE=block_size,
            BLOCK_K=min(block_size, hidden_size),
        )
        
        dsd_backward_weight_kernel[w2_grid](
            x_t,
            grad_output,
            grad_w2,
            m_block_to_expert,
            weight_offsets,
            x_t.stride(0), x_t.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_w2.stride(0), grad_w2.stride(1),
            batch_size,
            hidden_size,
            d_ffn,
            num_token_blocks,
            BLOCK_SIZE=block_size,
        )
        
        grad_w2 = grad_w2.to(w2.dtype)
        
        return grad_x, grad_w2, None, None, None, None, None
