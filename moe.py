import torch
import triton
import triton.language as tl
import triton.testing
import os
import math

'''Note: Karpathy calls hidden_size the n_embd. ffn_hidden_size is 4*n_embd (or hidden size!)
   Also!! Block size is the context length in the original karpathy config!! So we had some silliness with block_size this was a huge oversight by me
'''

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
    
    # Mask to prevent out-of-bounds access (compact storage may not align perfectly)
    # No mask needed here as the compact storage is sized exactly for the blocks
    tl.store(output_ptrs, acc)

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
    x_ptr, # the block sparse matrix, but compact: total_padded_tokens x d_ffn
    w2_ptr, # the dense weights: (num_experts * d_ffn) x hidden_size
    output_ptr, # the out matrix: total_padded_tokens x hidden_size
    row_indices_ptr, 
    weight_row_indices_ptr,
    stride_xm, stride_xk, # strides for x
    stride_wk, stride_wn, # strides for *w2*          
    stride_om, stride_on, # strides for the output
    d_ffn, hidden_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr, 
    GROUP_M: tl.constexpr=8
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1) 
    
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_M)

    # Load indices for the block
    row_idx = tl.load(row_indices_ptr + pid_m) # pick the tokens we're working on
    weight_col_idx = tl.load(weight_row_indices_ptr + pid_m) # encodes expert_id and ffn_block
    
    # Extract expert_id from weight_col_idx
    # weight_col_idx = expert_id * num_ffn_blocks + ffn_block_idx
    # w2 is laid out as (num_experts * d_ffn, hidden_size)
    num_ffn_blocks = d_ffn // BLOCK_SIZE  # 1024 / 64 = 16
    expert_id = weight_col_idx // num_ffn_blocks  # Get which expert (0-7)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32) # (block_size, block_k)@(block_k, block_size)
    
    for k in range(0, d_ffn, BLOCK_K):
        # Load BLOCK_SIZE x BLOCK_K tile from input
        x_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
        x_col_offsets = k + tl.arange(0, BLOCK_K)[None, :]
        
        x_ptrs = x_ptr + x_row_offsets * stride_xm + x_col_offsets * stride_xk #flipped these because we transpose the input
        # Need to mask both dimensions - x might have padded tokens
        x_mask = (x_col_offsets < d_ffn)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        x_tile = gelu_sigmoid(x_tile)

        # w2 weights for this expert start at expert_id * d_ffn
        w2_row_offsets = expert_id * d_ffn + k + tl.arange(0, BLOCK_K)[:, None]
        w2_col_offsets = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        w2_ptrs = w2_ptr + w2_row_offsets * stride_wk + w2_col_offsets * stride_wn
        
        w2_row_mask = (w2_row_offsets < (expert_id + 1) * d_ffn)
        w2_mask = w2_row_mask & (w2_col_offsets < hidden_size)
        w2_tile = tl.load(w2_ptrs, mask=w2_mask, other=0.0)

        acc += tl.dot(x_tile, w2_tile, allow_tf32=True)
    
    output_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    output_col_offsets = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    output_ptrs = output_ptr + output_row_offsets * stride_om + output_col_offsets * stride_on

    # Add mask to prevent out-of-bounds writes - need to check both dimensions
    # The last block might go beyond the actual number of tokens
    output_mask = (output_col_offsets < hidden_size)
    tl.store(output_ptrs, acc, mask=output_mask)

@triton.jit
def sdd_backward_act_kernel(
    grad_sparse_ptr,         # Sparse gradient ∂L/∂Y (num_padded_tokens, d_ffn) - compact storage
    w1_t_ptr,                # W1 transposed: (d_ffn * num_experts, hidden_size)
    grad_input_ptr,          # Output: dense gradient for input (num_padded_tokens, hidden_size)
    row_indices_ptr,         # Which token blocks to process
    weight_col_indices_ptr,  # Packed: expert_id * num_ffn_blocks + ffn_block_idx
    output_col_indices_ptr,  # Which column in compact storage
    stride_gm, stride_gk,    # Strides for grad_sparse
    stride_wk, stride_wn,    # Strides for w1_t
    stride_om, stride_on,    # Strides for grad_input
    d_ffn, hidden_size,      # Dimensions
    num_ffn_blocks,          # Number of FFN blocks per expert (for decoding)
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,   # Reduction dimension block size
):
    """Computes: grad_input = grad_sparse @ W1^T (∂L/∂X = ∂L/∂Y @ W1^T)
       This is a DSD operation: sparse @ dense → dense.
       
       Key difference from forward: output is dense, not sparse!
       Each thread block processes one sparse gradient block and scatters to dense output."""

    pid = tl.program_id(0)  # Which sparse block to process
    pid_n = tl.program_id(1)
    
    # Load indices for this block
    row_idx = tl.load(row_indices_ptr + pid)           # Which token block
    weight_col_idx = tl.load(weight_col_indices_ptr + pid) # Packed index
    output_col_idx = tl.load(output_col_indices_ptr + pid) # Position in compact storage
    
    # Decode the packed weight_col_idx to get expert ID and FFN block
    expert_idx = weight_col_idx // num_ffn_blocks  # Which expert
    ffn_block_idx = weight_col_idx % num_ffn_blocks  # Which FFN block within expert
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # This kernel processes ONE block of the sparse gradient
    # That block has BLOCK_SIZE rows and BLOCK_SIZE columns
    # We only need to loop over those BLOCK_SIZE columns for reduction
    for k in range(0, BLOCK_SIZE, BLOCK_K):
        # Load from sparse gradient [token_block, within-block chunk]
        grad_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
        grad_col_offsets = output_col_idx * BLOCK_SIZE + k + tl.arange(0, BLOCK_K)[None, :]
        grad_ptrs = grad_sparse_ptr + grad_row_offsets * stride_gm + grad_col_offsets * stride_gk
        
        # Mask to stay within the block's BLOCK_SIZE columns
        grad_mask = ((output_col_idx * BLOCK_SIZE + k + tl.arange(0, BLOCK_K)[None, :]) < d_ffn) & \
                    (k + tl.arange(0, BLOCK_K)[None, :] < BLOCK_SIZE)
        grad_tile = tl.load(grad_ptrs, mask=grad_mask, other=0.0)
        # Shape: (BLOCK_SIZE, BLOCK_K)
        
        # Load from W1^T [expert's d_ffn chunk, hidden_size]
        # W1^T is (d_ffn * num_experts, hidden_size), select expert's slice
        # Use ffn_block_idx (not output_col_idx) for the position in the expert's weights
        w1t_row_offsets = expert_idx * d_ffn + ffn_block_idx * BLOCK_SIZE + k + tl.arange(0, BLOCK_K)[:, None]
        w1t_col_offsets = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        w1t_ptrs = w1_t_ptr + w1t_row_offsets * stride_wk + w1t_col_offsets * stride_wn
        
        w1t_mask = (w1t_row_offsets < (expert_idx + 1) * d_ffn) & (w1t_col_offsets < hidden_size)
        w1t_tile = tl.load(w1t_ptrs, mask=w1t_mask, other=0.0)
        # Shape: (BLOCK_K, BLOCK_SIZE)
        
        # Accumulate: (BLOCK_SIZE, BLOCK_K) @ (BLOCK_K, BLOCK_SIZE) = (BLOCK_SIZE, BLOCK_SIZE)
        acc += tl.dot(grad_tile, w1t_tile)
    
    # Store to dense output (no atomics needed - each token written once!)
    output_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    output_col_offsets = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    output_ptrs = grad_input_ptr + output_row_offsets * stride_om + output_col_offsets * stride_on
    
    output_mask = (output_col_offsets < hidden_size)
    tl.store(output_ptrs, acc, mask=output_mask)

@triton.jit
def sdd_backward_weight_kernel(
    input_t_ptr,             # Dense input ALREADY TRANSPOSED: (hidden_size, num_padded_tokens)
    grad_sparse_ptr,         # Sparse gradient from loss (num_padded_tokens, d_ffn) - compact storage
    grad_weight_ptr,         # Output: gradient for w1 (hidden_size, num_experts * d_ffn)
    row_indices_ptr,         # Which token blocks to process
    expert_indices_ptr,      # Packed: expert_id * num_ffn_blocks + ffn_block_idx  
    output_col_indices_ptr,  # Which column in compact storage (for grad_sparse)
    stride_xm, stride_xk,    # Strides for input_t (transposed!)
    stride_gm, stride_gk,    # Strides for grad_sparse
    stride_wm, stride_wn,    # Strides for grad_weight
    hidden_size, d_ffn,      # Dimensions
    num_ffn_blocks,          # Number of FFN blocks per expert (for decoding)
    BLOCK_SIZE: tl.constexpr,
    # no block k since we don't loop
):
    """Computes: grad_weight = input^T @ grad_sparse (∂L/∂W1 = X^T @ ∂L/∂Y)
       Expects input already transposed!
       This is a DSD operation: dense @ sparse → dense (in sparse pattern).
       
       Grid dimensions:
       - dim 0: num_token_blocks (each with its specific FFN position)
       - dim 1: hidden_size tiles"""

    pid_blocks = tl.program_id(0)  # Which token block
    pid_h = tl.program_id(1)       # Which hidden_size tile
    # No pid_d needed - output_col_idx tells us the FFN position!

    # Load which tokens and expert this block handles
    row_idx = tl.load(row_indices_ptr + pid_blocks)       # Which token block
    weight_col_idx = tl.load(expert_indices_ptr + pid_blocks) # Packed index
    output_col_idx = tl.load(output_col_indices_ptr + pid_blocks) # Column in compact storage
    
    # Decode the packed weight_col_idx to get expert ID and FFN block
    expert_idx = weight_col_idx // num_ffn_blocks  # Which expert
    ffn_block_idx = weight_col_idx % num_ffn_blocks  # Which FFN block within expert
    
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
    grad_col_offsets = output_col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    grad_ptrs = grad_sparse_ptr + grad_row_offsets * stride_gm + grad_col_offsets * stride_gk
    
    grad_mask = (grad_col_offsets < (output_col_idx + 1) * BLOCK_SIZE)  # Mask within compact block
    grad_tile = tl.load(grad_ptrs, mask=grad_mask, other=0.0)
    # Shape: (BLOCK_SIZE, BLOCK_SIZE) representing (token_chunk, d_ffn_chunk)
    
    # Compute: (hidden_chunk, token_chunk) @ (token_chunk, d_ffn_chunk) = (hidden_chunk, d_ffn_chunk)
    result = tl.dot(input_tile, grad_tile)
    
    # Atomically add to the expert's weight gradient
    # grad_weight[hidden_tile, expert_idx * d_ffn + ffn_block_position]
    weight_row_offsets = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    # Use ffn_block_idx (not output_col_idx) for position in expert's weights
    weight_col_offsets = expert_idx * d_ffn + ffn_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
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
    d_ffn, hidden_size,  # Need both dimensions!
    num_ffn_blocks,      # Number of FFN blocks per expert (for decoding)
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
        # Decode the packed weight_col_idx to get expert ID and FFN block
        expert_id = weight_col_idx // num_ffn_blocks
        ffn_block_idx = weight_col_idx % num_ffn_blocks
        
        w2t_row_offsets = k + tl.arange(0, BLOCK_K)[:, None]
        # Use ffn_block_idx (not output_col_idx) for position in expert's weights
        w2t_col_offsets = expert_id * d_ffn + ffn_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        w2t_ptrs = w2_t_ptr + w2t_row_offsets * stride_wk + w2t_col_offsets * stride_wn
        w2t_mask = (w2t_row_offsets < hidden_size) & (w2t_col_offsets < (expert_id + 1) * d_ffn)
        w2t_tile = tl.load(w2t_ptrs, mask=w2t_mask, other=0.0)

        # Accumulate
        acc += tl.dot(grad_tile, w2t_tile)

    # Store result
    output_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    output_col_offsets = output_col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    output_ptrs = grad_act_ptr + output_row_offsets * stride_om + output_col_offsets * stride_on
    # No mask needed - compact storage is sized exactly for the blocks
    tl.store(output_ptrs, acc)

@triton.jit
def dsd_backward_weight_kernel(
    sparse_act_t_ptr,        # Sparse activations ALREADY TRANSPOSED: (d_ffn, num_padded_tokens)
    grad_output_ptr,         # Gradient from loss (num_padded_tokens, hidden_size)
    grad_weight_ptr,         # Output: gradient for w2 (num_experts * d_ffn, hidden_size)
    row_indices_ptr,         # Which token blocks to process
    expert_indices_ptr,      # Which expert each block belongs to
    output_col_indices_ptr,  # Which column in compact storage
    stride_am, stride_ak,    # Strides for sparse_act_t (note: transposed!)
    stride_gm, stride_gk,    # Strides for grad_output
    stride_wm, stride_wn,    # Strides for grad_weight
    d_ffn,           # The compact dimension (actual width of sparse_act)
    hidden_size,             # Hidden dimension (e.g., 384)
    num_ffn_blocks,          # Number of FFN blocks per expert (for decoding)
    BLOCK_SIZE: tl.constexpr,
    # no block k since we don't loop
):
    """Computes: grad_weight = sparse_activations^T @ grad_output (∂L/∂W = X^T @ ∂L/∂Y)
       Expects sparse_act already transposed!
       This is essentially an SDD operation.
       
       Grid dimensions:
       - dim 0: num_token_blocks (each with its specific FFN position)
       - dim 1: hidden_size tiles"""

    pid_blocks = tl.program_id(0)  # Which token block
    pid_h = tl.program_id(1)       # Which hidden_size tile
    # No pid_d needed - output_col_idx tells us the FFN position!

    # Load which tokens and expert this block handles
    row_idx = tl.load(row_indices_ptr + pid_blocks)       # Which token block
    weight_col_idx = tl.load(expert_indices_ptr + pid_blocks) # Packed index
    output_col_idx = tl.load(output_col_indices_ptr + pid_blocks)
    
    # Decode the packed weight_col_idx to get expert ID and FFN block
    expert_idx = weight_col_idx // num_ffn_blocks  # Which expert
    ffn_block_idx = weight_col_idx % num_ffn_blocks  # Which FFN block within expert
    
    # Since each token block has exactly BLOCK_SIZE tokens, we don't need a loop!
    # Just one tile load and one dot product.
    
    # Load from transposed sparse activations
    # It's (d_ffn, num_padded_tokens)
    # Since each block only has BLOCK_SIZE columns in compact storage:
    act_row_offsets = output_col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    act_col_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    act_ptrs = sparse_act_t_ptr + act_row_offsets * stride_am + act_col_offsets * stride_ak
    
    act_mask = (act_row_offsets < d_ffn)  # Mask within compact dimension
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
    # Map from compact storage position to expert weight position
    # Use ffn_block_idx (not output_col_idx) for position in expert's weights
    weight_row_offsets = expert_idx * d_ffn + ffn_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    weight_col_offsets = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    weight_ptrs = grad_weight_ptr + weight_row_offsets * stride_wm + weight_col_offsets * stride_wn
    
    weight_mask = (weight_row_offsets < (expert_idx + 1) * d_ffn) & (weight_col_offsets < hidden_size)
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
    @torch.compiler.disable
    def forward(ctx, x, w1, row_indices, weight_col_indices, output_col_indices, block_size=64, num_ffn_blocks=None):
        """
        Args:
            x: Dense input tensor (num_padded_tokens, hidden_size)
            w1: Dense weight tensor (hidden_size, num_experts * d_ffn)
            row_indices: Which token blocks to process
            weight_col_indices: Which expert each block belongs to (for selecting W1 slice)
            output_col_indices: Which column in compact storage
            block_size: Block size for Triton kernel
        
        Returns:
            Sparse output in compact form (num_padded_tokens, d_ffn_compact)
        """
        batch_size, hidden_size = x.shape
        num_experts_times_dffn = w1.shape[1]
        num_blocks = len(row_indices)
        
        # Use the provided num_ffn_blocks to determine output width
        # This avoids data-dependent operations for torch.compile
        if num_ffn_blocks is None:
            # Fallback: infer from the maximum output column index
            # This path should rarely be used when called from compiled code
            if len(output_col_indices) > 0:
                # We need to convert to int for tensor allocation, but do it efficiently
                max_output_col_int = int(output_col_indices.max())
                output_width = (max_output_col_int + 1) * block_size
            else:
                output_width = block_size
        else:
            output_width = num_ffn_blocks * block_size
        
        # Allocate output tensor (compact storage)
        output = torch.zeros((batch_size, output_width), dtype=x.dtype, device=x.device)
        
        # Configure grid
        grid = (num_blocks,)
        
        # Launch kernel
        sdd_kernel[grid](
            x, w1, output,
            row_indices, weight_col_indices, output_col_indices,
            x.stride(0), x.stride(1),
            w1.stride(0), w1.stride(1),
            output.stride(0), output.stride(1),
            hidden_size,
            BLOCK_SIZE=block_size,  # This is the Triton tile size (64), NOT context length!
            BLOCK_K=min(block_size, hidden_size),
        )
        '''save the all the things we need for our backward passes: tensors, integer inputs
           the current plan is to transpose in the backward pass since torch transposes are cheap to save the vram,
           but we'll have to see how it works out '''
        ctx.save_for_backward(x, w1, row_indices, weight_col_indices, output_col_indices)
        ctx.block_size = block_size
        ctx.hidden_size = hidden_size
        ctx.d_ffn = output_width
        
        return output
    
    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output):
        """
        Computes gradients for x and w1 given grad_output
        
        Args:
            grad_output: Gradient w.r.t. sparse output (num_padded_tokens, d_ffn_compact)
        
        Returns:
            Gradients for: x, w1, row_indices, weight_col_indices, output_col_indices, block_size
        """
        # Retrieve saved tensors and metadata
        x, w1, row_indices, weight_col_indices, output_col_indices = ctx.saved_tensors
        block_size = ctx.block_size
        hidden_size = ctx.hidden_size
        d_ffn = ctx.d_ffn

        x_t, w1_t = x.t(), w1.t()

        num_blocks = (row_indices != 0).sum() #len(row_indices)
        hidden_tiles = math.ceil(hidden_size / block_size)
        d_ffn_tiles = math.ceil(d_ffn / block_size)
        num_ffn_blocks = d_ffn_tiles  # Number of FFN blocks per expert

        x_grid = (num_blocks, hidden_tiles) # 2 pids
        w1_grid = (num_blocks, hidden_tiles) # 2D grid - each block already knows its FFN position
        
        # Allocate gradient tensors
        grad_x = torch.zeros_like(x) 
        # Use float32 for weight gradients to support atomic operations
        grad_w1 = torch.zeros_like(w1, dtype=torch.float32)

        sdd_backward_act_kernel[x_grid](
            grad_output,         # grad_sparse_ptr
            w1_t,               # w1_t_ptr
            grad_x,             # grad_input_ptr
            row_indices,        # row_indices_ptr
            weight_col_indices, # weight_col_indices_ptr
            output_col_indices, # output_col_indices_ptr
            grad_output.stride(0), grad_output.stride(1),  # stride_gm, stride_gk
            w1_t.stride(0), w1_t.stride(1),                # stride_wk, stride_wn
            grad_x.stride(0), grad_x.stride(1),            # stride_om, stride_on
            d_ffn, hidden_size,                             # dimensions
            num_ffn_blocks,                                 # For decoding packed indices
            BLOCK_SIZE=block_size,
            BLOCK_K=min(block_size, d_ffn),
        )
        
        expert_indices = weight_col_indices  # Assuming these are the same
        
        sdd_backward_weight_kernel[w1_grid](
            x_t,                # input_t_ptr
            grad_output,        # grad_sparse_ptr
            grad_w1,            # grad_weight_ptr
            row_indices,        # row_indices_ptr
            expert_indices,     # expert_indices_ptr (packed indices)
            output_col_indices, # output_col_indices_ptr
            x_t.stride(0), x_t.stride(1),           # stride_xm, stride_xk
            grad_output.stride(0), grad_output.stride(1),  # stride_gm, stride_gk
            grad_w1.stride(0), grad_w1.stride(1),   # stride_wm, stride_wn
            hidden_size, d_ffn,                     # dimensions
            num_ffn_blocks,                         # For decoding packed indices
            BLOCK_SIZE=block_size,  # Triton tile size
        )

        
        # Convert weight gradient back to original dtype
        grad_w1 = grad_w1.to(w1.dtype)
        
        return grad_x, grad_w1, None, None, None, None, None


class DSD(torch.autograd.Function):
    """Dense-Sparse-Dense operation for MoE second layer (X @ W2)
    
    Computes Y = X @ W2 where:
    - X is sparse input in compact form (batch_tokens, d_ffn_compact)
    - W2 is dense weights (num_experts * d_ffn, hidden_size)
    - Y is dense output (batch_tokens, hidden_size)
    """
    
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, x, w2, row_indices, weight_row_indices, output_col_indices, block_size=64):
        """
        Args:
            x: Sparse input tensor in compact form (num_padded_tokens, d_ffn_compact)
            w2: Dense weight tensor (num_experts * d_ffn, hidden_size)
            row_indices: Which token blocks to process
            weight_row_indices: Which expert weights to use
            output_col_indices: Position in compact storage (from SDD)
            block_size: Block size for Triton kernel
        
        Returns:
            Dense output (num_padded_tokens, hidden_size)
        """
        batch_size = x.shape[0]
        d_ffn = x.shape[1]
        hidden_size = w2.shape[1]
        num_blocks = len(row_indices)#(row_indices != 0).sum()
        
        # Allocate dense output
        output = torch.zeros((batch_size, hidden_size), dtype=x.dtype, device=x.device)
        
        # Configure grid - need to tile over hidden_size dimension
        hidden_tiles = math.ceil(hidden_size / block_size)
        grid = (num_blocks, hidden_tiles)
        
        # Launch kernel
        dsd_kernel[grid](
            x, w2, output,
            row_indices, weight_row_indices,
            x.stride(0), x.stride(1),
            w2.stride(0), w2.stride(1),
            output.stride(0), output.stride(1),
            d_ffn, hidden_size,
            BLOCK_SIZE=block_size,  # Triton tile size
            BLOCK_K=min(block_size, d_ffn),
            GROUP_M=8
        )
        
        ctx.save_for_backward(x, w2, row_indices, weight_row_indices, output_col_indices)
        ctx.block_size = block_size
        ctx.hidden_size = hidden_size
        ctx.d_ffn = d_ffn
        
        return output
    
    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output):
        """
        Computes gradients for x and w2 given grad_output
        
        Args:
            grad_output: Gradient w.r.t. dense output (num_padded_tokens, hidden_size)
        
        Returns:
            Gradients for: x, w2, row_indices, weight_row_indices, output_col_indices, block_size
        """
        # Retrieve saved tensors
        x, w2, row_indices, weight_row_indices, output_col_indices = ctx.saved_tensors
        block_size = ctx.block_size
        hidden_size = ctx.hidden_size
        d_ffn = ctx.d_ffn  # The d_ffn dimension from forward

        x_t = x.t()
        w2_t = w2.t()

        num_blocks = len(row_indices)
        hidden_tiles = math.ceil(hidden_size / block_size)
        d_ffn_tiles = math.ceil(d_ffn / block_size)
        num_ffn_blocks = d_ffn_tiles  # Number of FFN blocks per expert

        x_grid = (num_blocks,)
        w2_grid = (num_blocks, hidden_tiles) # 2D grid - each block already knows its FFN position

        grad_x = torch.zeros_like(x) 
        # Use float32 for weight gradients to support atomic operations
        grad_w2 = torch.zeros_like(w2, dtype=torch.float32)

        dsd_backward_act_kernel[x_grid](
            grad_output,         # grad_output_ptr
            w2_t,               # w2_t_ptr
            grad_x,             # grad_act_ptr (output)
            row_indices,        # row_indices_ptr
            weight_row_indices, # weight_col_indices_ptr (which expert)
            output_col_indices, # output_col_indices_ptr (position in compact)
            grad_output.stride(0), grad_output.stride(1),  # stride_gm, stride_gk
            w2_t.stride(0), w2_t.stride(1),                # stride_wk, stride_wn
            grad_x.stride(0), grad_x.stride(1),            # stride_om, stride_on
            d_ffn, hidden_size,                            # dimensions
            num_ffn_blocks,                                # For decoding packed indices
            BLOCK_SIZE=block_size,  # Triton tile size
            BLOCK_K=min(block_size, hidden_size),
        )
        
        expert_indices = weight_row_indices

        dsd_backward_weight_kernel[w2_grid](
            x_t,                # sparse_act_t_ptr (already transposed)
            grad_output,        # grad_output_ptr
            grad_w2,            # grad_weight_ptr (output)
            row_indices,        # row_indices_ptr
            expert_indices,     # expert_indices_ptr
            output_col_indices, # output_col_indices_ptr
            x_t.stride(0), x_t.stride(1),           # stride_am, stride_ak
            grad_output.stride(0), grad_output.stride(1),  # stride_gm, stride_gk
            grad_w2.stride(0), grad_w2.stride(1),   # stride_wm, stride_wn
            d_ffn, hidden_size,                     # dimensions
            num_ffn_blocks,                         # For decoding packed indices
            BLOCK_SIZE=block_size,  # Triton tile size
        )
        
        # Convert weight gradient back to original dtype
        grad_w2 = grad_w2.to(w2.dtype)
        
        return grad_x, grad_w2, None, None, None, None