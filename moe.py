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
    weight_row_idx = tl.load(weight_row_indices_ptr + pid) # since w2 is (d_ffn*num_experts, hidden_size), pick out the right rows to operate with 

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
def sdd_backward_act_kernel(
    grad_sparse_ptr,         # Sparse gradient ∂L/∂Y (num_padded_tokens, d_ffn) - compact storage
    w1_t_ptr,                # W1 transposed: (d_ffn * num_experts, hidden_size)
    grad_input_ptr,          # Output: dense gradient for input (num_padded_tokens, hidden_size)
    row_indices_ptr,         # Which token blocks to process
    weight_col_indices_ptr,  # Which expert each block belongs to (for selecting W1 slice)
    output_col_indices_ptr,  # Which column in compact storage
    stride_gm, stride_gk,    # Strides for grad_sparse
    stride_wk, stride_wn,    # Strides for w1_t
    stride_om, stride_on,    # Strides for grad_input
    d_ffn, hidden_size,      # Dimensions
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,   # Reduction dimension block size
):
    """Computes: grad_input = grad_sparse @ W1^T (∂L/∂X = ∂L/∂Y @ W1^T)
       This is a DSD operation: sparse @ dense → dense.
       
       Key difference from forward: output is dense, not sparse!
       Each thread block processes one sparse gradient block and scatters to dense output."""

    pid = tl.program_id(0)  # Which sparse block to process
    
    # Load indices for this block
    row_idx = tl.load(row_indices_ptr + pid)           # Which token block
    expert_idx = tl.load(weight_col_indices_ptr + pid) # Which expert's W1 to use
    output_col_idx = tl.load(output_col_indices_ptr + pid) # Position in compact storage
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over d_ffn dimension (reduction dimension)
    for k in range(0, d_ffn, BLOCK_K):
        # Load from sparse gradient [token_block, d_ffn chunk]
        # grad_sparse is compact, so use output_col_idx for indexing
        grad_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
        grad_col_offsets = output_col_idx * BLOCK_SIZE + k + tl.arange(0, BLOCK_K)[None, :]
        grad_ptrs = grad_sparse_ptr + grad_row_offsets * stride_gm + grad_col_offsets * stride_gk
        
        grad_mask = (k + tl.arange(0, BLOCK_K)[None, :] < d_ffn)
        grad_tile = tl.load(grad_ptrs, mask=grad_mask, other=0.0)
        # Shape: (BLOCK_SIZE, BLOCK_K)
        
        # Load from W1^T [expert's d_ffn chunk, hidden_size]
        # W1^T is (d_ffn * num_experts, hidden_size), select expert's slice
        w1t_row_offsets = expert_idx * d_ffn + k + tl.arange(0, BLOCK_K)[:, None]
        w1t_col_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        w1t_ptrs = w1_t_ptr + w1t_row_offsets * stride_wk + w1t_col_offsets * stride_wn
        
        w1t_mask = (w1t_row_offsets < (expert_idx + 1) * d_ffn) & (w1t_col_offsets < hidden_size)
        w1t_tile = tl.load(w1t_ptrs, mask=w1t_mask, other=0.0)
        # Shape: (BLOCK_K, BLOCK_SIZE)
        
        # Accumulate: (BLOCK_SIZE, BLOCK_K) @ (BLOCK_K, BLOCK_SIZE) = (BLOCK_SIZE, BLOCK_SIZE)
        acc += tl.dot(grad_tile, w1t_tile)
    
    # Store to dense output (no atomics needed - each token written once!)
    output_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    output_col_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    output_ptrs = grad_input_ptr + output_row_offsets * stride_om + output_col_offsets * stride_on
    
    output_mask = (output_col_offsets < hidden_size)
    tl.store(output_ptrs, acc, mask=output_mask)

@triton.jit
def sdd_backward_weight_kernel(
    input_t_ptr,             # Dense input ALREADY TRANSPOSED: (hidden_size, num_padded_tokens)
    grad_sparse_ptr,         # Sparse gradient from loss (num_padded_tokens, d_ffn) - compact storage
    grad_weight_ptr,         # Output: gradient for w1 (hidden_size, num_experts * d_ffn)
    row_indices_ptr,         # Which token blocks to process
    expert_indices_ptr,      # Which expert each block belongs to  
    output_col_indices_ptr,  # Which column in compact storage (for grad_sparse)
    stride_xm, stride_xk,    # Strides for input_t (transposed!)
    stride_gm, stride_gk,    # Strides for grad_sparse
    stride_wm, stride_wn,    # Strides for grad_weight
    hidden_size, d_ffn,      # Dimensions
    BLOCK_SIZE: tl.constexpr,
    # no block k since we don't loop
):
    """Computes: grad_weight = input^T @ grad_sparse (∂L/∂W1 = X^T @ ∂L/∂Y)
       Expects input already transposed!
       This is a DSD operation: dense @ sparse → dense (in sparse pattern).
       
       Grid dimensions:
       - dim 0: num_token_blocks
       - dim 1: hidden_size tiles  
       - dim 2: d_ffn tiles"""

    pid_blocks = tl.program_id(0)  # Which token block
    pid_h = tl.program_id(1)       # Which hidden_size tile
    pid_d = tl.program_id(2)       # Which d_ffn tile

    # Load which tokens and expert this block handles
    row_idx = tl.load(row_indices_ptr + pid_blocks)       # Which token block
    expert_idx = tl.load(expert_indices_ptr + pid_blocks) # Which expert
    output_col_idx = tl.load(output_col_indices_ptr + pid_blocks) # Column in compact storage
    
    # No loop needed - token block size matches tile size!
    
    # Load from transposed input 
    # input_t is already (hidden_size, num_padded_tokens)
    # We want tile [hidden_tile, token_block]
    input_row_offsets = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    input_col_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    input_ptrs = input_t_ptr + input_row_offsets * stride_xm + input_col_offsets * stride_xk
    
    input_mask = (input_row_offsets < hidden_size)  # Only mask hidden dimension
    input_tile = tl.load(input_ptrs, mask=input_mask, other=0.0)
    # Shape: (BLOCK_SIZE, BLOCK_SIZE) representing (hidden_chunk, token_chunk)
    
    # Load from sparse gradient [token_block, d_ffn_tile]
    # Note: grad_sparse is in compact form, use output_col_idx for proper indexing
    grad_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    grad_col_offsets = output_col_idx * BLOCK_SIZE + pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    grad_ptrs = grad_sparse_ptr + grad_row_offsets * stride_gm + grad_col_offsets * stride_gk
    
    grad_mask = (grad_col_offsets < (output_col_idx + 1) * BLOCK_SIZE)  # Mask within compact block
    grad_tile = tl.load(grad_ptrs, mask=grad_mask, other=0.0)
    # Shape: (BLOCK_SIZE, BLOCK_SIZE) representing (token_chunk, d_ffn_chunk)
    
    # Compute: (hidden_chunk, token_chunk) @ (token_chunk, d_ffn_chunk) = (hidden_chunk, d_ffn_chunk)
    result = tl.dot(input_tile, grad_tile)
    
    # Atomically add to the expert's weight gradient
    # grad_weight[hidden_tile, expert_idx * d_ffn + d_ffn_tile]  
    weight_row_offsets = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    weight_col_offsets = expert_idx * d_ffn + pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    weight_ptrs = grad_weight_ptr + weight_row_offsets * stride_wm + weight_col_offsets * stride_wn
    
    weight_mask = (weight_row_offsets < hidden_size) & (weight_col_offsets < (expert_idx + 1) * d_ffn)
    tl.atomic_add(weight_ptrs, result, mask=weight_mask)

@triton.jit
def dsd_backward_act_kernel(
    grad_output_ptr, # ∂L/∂Y
    w2_t_ptr, # w2 transposed
    grad_act_ptr, # ∂L/∂X
    row_indices_ptr,
    weight_col_indices_ptr,
    output_col_indices_ptr, 
    stride_gm, stride_gk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Computes grad_sparse_act = grad_output @ w2.T"""
    pid = tl.program_id(0)
    row_idx = tl.load(row_indices_ptr + pid)
    weight_col_idx = tl.load(weight_col_indices_ptr + pid)
    output_col_idx = tl.load(output_col_indices_ptr + pid)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, hidden_size, BLOCK_K):
        # Load from grad_output
        grad_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
        grad_col_offsets = k + tl.arange(0, BLOCK_K)[None, :]
        grad_ptrs = grad_output_ptr + grad_row_offsets * stride_gm + grad_col_offsets * stride_gk
        grad_mask = (grad_col_offsets < hidden_size)
        grad_tile = tl.load(grad_ptrs, mask=grad_mask, other=0.0)

        # Load from w2.T
        w2t_row_offsets = k + tl.arange(0, BLOCK_K)[:, None]
        w2t_col_offsets = weight_col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        w2t_ptrs = w2_t_ptr + w2t_row_offsets * stride_wk + w2t_col_offsets * stride_wn
        w2t_mask = (w2t_row_offsets < hidden_size)  # Only mask the reduction dimension
        w2t_tile = tl.load(w2t_ptrs, mask=w2t_mask, other=0.0)

        # Accumulate
        acc += tl.dot(grad_tile, w2t_tile)

    # Store result
    output_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    output_col_offsets = output_col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    output_ptrs = grad_act_ptr + output_row_offsets * stride_om + output_col_offsets * stride_on
    tl.store(output_ptrs, acc)

@triton.jit
def dsd_backward_weight_kernel(
    sparse_act_t_ptr,        # Sparse activations ALREADY TRANSPOSED: (d_ffn, num_padded_tokens)
    grad_output_ptr,         # Gradient from loss (num_padded_tokens, hidden_size)
    grad_weight_ptr,         # Output: gradient for w2 (num_experts * d_ffn, hidden_size)
    row_indices_ptr,         # Which token blocks to process
    expert_indices_ptr,      # Which expert each block belongs to
    stride_am, stride_ak,    # Strides for sparse_act_t (note: transposed!)
    stride_gm, stride_gk,    # Strides for grad_output
    stride_wm, stride_wn,    # Strides for grad_weight
    d_ffn, hidden_size,      # Dimensions
    BLOCK_SIZE: tl.constexpr,
    # no block k since we don't loop
):
    """Computes: grad_weight = sparse_activations^T @ grad_output (∂L/∂W = X^T @ ∂L/∂Y)
       Expects sparse_act already transposed!
       This is essentially an SDD operation.
       
       Grid dimensions:
       - dim 0: num_token_blocks  
       - dim 1: d_ffn tiles
       - dim 2: hidden_size tiles"""

    pid_blocks = tl.program_id(0)  # Which token block
    pid_d = tl.program_id(1)       # Which d_ffn tile  
    pid_h = tl.program_id(2)       # Which hidden_size tile

    # Load which tokens and expert this block handles
    row_idx = tl.load(row_indices_ptr + pid_blocks)       # Which token block
    expert_idx = tl.load(expert_indices_ptr + pid_blocks) # Which expert
    
    # Since each token block has exactly BLOCK_SIZE tokens, we don't need a loop!
    # Just one tile load and one dot product.
    
    # Load from transposed sparse activations
    # sparse_act_t is already (d_ffn, num_padded_tokens)
    # We want tile [d_ffn_tile, token_block]
    act_row_offsets = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    act_col_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    act_ptrs = sparse_act_t_ptr + act_row_offsets * stride_am + act_col_offsets * stride_ak
    
    act_mask = (act_row_offsets < d_ffn)  # Only mask d_ffn dimension
    act_tile = tl.load(act_ptrs, mask=act_mask, other=0.0)
    # Shape: (BLOCK_SIZE, BLOCK_SIZE) representing (d_ffn_chunk, token_chunk)
    
    # Load from grad_output[token_block, hidden_tile]
    grad_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    grad_col_offsets = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    grad_ptrs = grad_output_ptr + grad_row_offsets * stride_gm + grad_col_offsets * stride_gk
    
    grad_mask = (grad_col_offsets < hidden_size)  # Only mask hidden dimension
    grad_tile = tl.load(grad_ptrs, mask=grad_mask, other=0.0)
    # Shape: (BLOCK_SIZE, BLOCK_SIZE) representing (token_chunk, hidden_chunk)
    
    # Compute: (d_ffn_chunk, token_chunk) @ (token_chunk, hidden_chunk) = (d_ffn_chunk, hidden_chunk)
    result = tl.dot(act_tile, grad_tile)
    
    # Atomically add to the expert's weight gradient
    # grad_weight[expert_idx * d_ffn + d_ffn_tile, hidden_tile]
    weight_row_offsets = expert_idx * d_ffn + pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    weight_col_offsets = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    weight_ptrs = grad_weight_ptr + weight_row_offsets * stride_wm + weight_col_offsets * stride_wn
    
    weight_mask = (weight_row_offsets < (expert_idx + 1) * d_ffn) & (weight_col_offsets < hidden_size)
    tl.atomic_add(weight_ptrs, result, mask=weight_mask)

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