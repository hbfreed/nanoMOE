import torch
import triton
import triton.language as tl
import triton.testing
import os

# Enable verbose Triton autotuning output unless explicitly disabled by the environment
if os.environ.get("TRITON_PRINT_AUTOTUNING") is None:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
'''Note: Karpathy calls hidden_size the n_embd. ffn_hidden_size is 4*n_embd (or hidden size!)'''

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
#         triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 128}),
#         # Large blocks for large batch/hidden sizes
#         triton.Config({'BLOCK_SIZE': 128, 'BLOCK_K': 32}),
#         triton.Config({'BLOCK_SIZE': 128, 'BLOCK_K': 64}),
#         triton.Config({'BLOCK_SIZE': 128, 'BLOCK_K': 128}),
#         triton.Config({'BLOCK_SIZE': 128, 'BLOCK_K': 256}),
#         # Even larger for big models
#         triton.Config({'BLOCK_SIZE': 256, 'BLOCK_K': 64}),
#         triton.Config({'BLOCK_SIZE': 256, 'BLOCK_K': 128}),
#         triton.Config({'BLOCK_SIZE': 256, 'BLOCK_K': 256}),
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
    """lots of comments because i'm sill learning triton!"""
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
def dsd_kernel(
    x_ptr, # the block sparse matrix, but compact: total_padded_tokens x d_ffn
    w2_ptr, # the dense weights: d_ffn * num_experts x hidden_size
    output_ptr, # the out matrix: total_padded_tokens x hidden_size
    row_indices_ptr, 
    weight_row_indices_ptr,
    stride_xm, stride_xk, # strides for x
    stride_wk, stride_wn, # strides for *w2*          
    stride_om, stride_on, # strides for the output
    d_ffn, hidden_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr, # reduction dimension

):
    pid = tl.program_id(0)

    # Load indices for the block
    row_idx = tl.load(row_indices_ptr + pid) # pick the tokens we're working on
    weight_row_idx = tl.load(weight_row_indices_ptr + pid) # since w2 is (d_ffn*n_exp,hidden size), pick out the right rows to operate with 

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32) # (block_size, block_k)@(block_k, block_size)
    
    for k in range(0, d_ffn, BLOCK_K):
        # Load BLOCK_SIZE x BLOCK_K tile from input
        x_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
        x_col_offsets = k + tl.arange(0, BLOCK_K)[None, :]
        x_ptrs = x_ptr + x_row_offsets * stride_xm + x_col_offsets * stride_xk
        
        x_mask = (x_col_offsets < d_ffn)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w2_row_offsets = weight_row_idx * d_ffn + k + tl.arange(0, BLOCK_K)[:, None]
        w2_col_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        w2_ptrs = w2_ptr + w2_row_offsets * stride_wk + w2_col_offsets * stride_wn
        
        w2_mask = (w2_col_offsets < hidden_size)
        w2_tile = tl.load(w2_ptrs, mask=w2_mask, other=0.0)

        acc += tl.dot(x_tile, w2_tile)
    
    output_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    output_col_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    output_ptrs = output_ptr + output_row_offsets * stride_om + output_col_offsets * stride_on

    tl.store(output_ptrs, acc)

@triton.jit
def sdd_backward_act_kernel():
    pass

@triton.jit
def sdd_backward_weight_kernel():
    pass

@triton.jit
def dsd_backward_act_kernel():
    """Computes: grad_sparse_act = grad_output @ weight_down.T
       Pretty similar to the sdd calculation! We calculate ∂L/∂X = ∂L/∂Y @ W_2^T, and make sure each expert's tokens goes to the right place."""


@triton.jit
def dsd_backward_weight_kernel():
    """Computes: grad_weight_down = sparse_activations.T @ grad_output
       We calculate ∂L/∂W = X^T @ ∂L/∂Y"""
    # Each block handles an expert's weight gradient
    # Needs to accumulate across all tokens for that expert
    # Might need atomics if multiple blocks update same weight

@triton.jit
def gelu(x):
    '''gelu based on https://arxiv.org/pdf/1606.08415#page=2'''
    pass

def approx_gelu(x):
    '''approximated gelu based on https://arxiv.org/pdf/1606.08415#page=2'''
    pass

@triton.jit
def tanh(x):
    pass