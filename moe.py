import torch
import triton
import triton.language as tl
import triton.testing
'''Note: Karpathy calls hidden_size the n_embd. ffn_hidden_size is 4*n_embd (or hidden size!)'''

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 32, 'BLOCK_K': 16}),
#         triton.Config({'BLOCK_SIZE': 32, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 16}),
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 64}),
#         triton.Config({'BLOCK_SIZE': 128, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 128, 'BLOCK_K': 64}),
#         triton.Config({'BLOCK_SIZE': 128, 'BLOCK_K': 128}),
#     ],
#     key=['hidden_size'],
# )

@triton.jit
def sdd_kernel(
    x_ptr, w1_ptr, output_ptr,
    row_indices_ptr, 
    weight_col_indices_ptr,  # For loading weights (includes expert offset)
    output_col_indices_ptr,  # For writing output (no expert offset!)
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """SDD kernel for compact block-sparse matrix multiplication."""
    pid = tl.program_id(0)
    
    # Load indices for this block
    row_idx = tl.load(row_indices_ptr + pid)
    weight_col_idx = tl.load(weight_col_indices_ptr + pid)  # Which expert's weights
    output_col_idx = tl.load(output_col_indices_ptr + pid)  # Where in compact output
    
    # Initialize accumulator for this block
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over K dimension in tiles
    for k in range(0, hidden_size, BLOCK_K):
        # Load BLOCK_SIZE x BLOCK_K tile from input
        x_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
        x_col_offsets = k + tl.arange(0, BLOCK_K)[None, :]
        x_ptrs = x_ptr + x_row_offsets * stride_xm + x_col_offsets * stride_xk
        
        x_mask = (x_col_offsets < hidden_size)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load BLOCK_K x BLOCK_SIZE tile from weights
        # Use weight_col_idx to select the right expert's weights
        w1_row_offsets = k + tl.arange(0, BLOCK_K)[:, None]
        w1_col_offsets = weight_col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        w1_ptrs = w1_ptr + w1_row_offsets * stride_wk + w1_col_offsets * stride_wn
        
        w1_mask = (w1_row_offsets < hidden_size)
        w1_tile = tl.load(w1_ptrs, mask=w1_mask, other=0.0)
        
        # Accumulate the matrix multiplication
        acc += tl.dot(x_tile, w1_tile)
    
    # Store BLOCK_SIZE x BLOCK_SIZE output block
    # Use output_col_idx for compact output (no expert offset!)
    output_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    output_col_offsets = output_col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    output_ptrs = output_ptr + output_row_offsets * stride_om + output_col_offsets * stride_on
    
    tl.store(output_ptrs, acc)

@triton.jit
def dsd(
    block_sparse_ptr,
    w2_ptr,
    output_ptr,
):
    pass

@triton.jit
def gelu():
    pass

@triton.jit
def padded_scatter():
    pass