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

def reconstruct_from_blocks(output, num_row_blocks, num_col_blocks, block_size=64):
    """Reconstruct a matrix from packed blocks."""
    result = torch.zeros(
        num_row_blocks * block_size, 
        num_col_blocks * block_size,
        device=output.device,
        dtype=output.dtype
    )
    
    block_idx = 0
    for row_block in range(num_row_blocks):
        for col_block in range(num_col_blocks):
            row_start = row_block * block_size
            row_end = row_start + block_size
            col_start = col_block * block_size
            col_end = col_start + block_size
            
            # Extract this block from packed output
            block_data = output[block_idx * block_size:(block_idx + 1) * block_size, :]
            result[row_start:row_end, col_start:col_end] = block_data
            block_idx += 1
    
    return result

@triton.jit
def sdd_kernel(
    x_ptr, w1_ptr, output_ptr,
    row_indices_ptr, col_indices_ptr,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,  # Used for both M and N
    BLOCK_K: tl.constexpr,     # Reduction tiling can be different
):
    """lots of comments because i'm still learning triton!"""
    pid = tl.program_id(0)
    
    # figure out which row/col we want to operate on: each pid gets it's own block
    row_idx = tl.load(row_indices_ptr + pid)
    col_idx = tl.load(col_indices_ptr + pid)
    
    # initialize the *one block* we're computing per program
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # one output block of block_size x block_size will take multiple block_size x block_k tiles to compute, so loop
    for k in range(0, hidden_size, BLOCK_K):  # loop to hidden_size taking block_k steps
        # Load BLOCK_SIZE x BLOCK_K tile from input
        # add the beginning pointer we're working with to the arange of block sizes giving us a vector of the pointers we're operating on
        x_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
        # add which column we're currently operating on to the arange of block sizes
        x_col_offsets = k + tl.arange(0, BLOCK_K)[None, :]
        # (pointwise multiplication) using the numbers from above, pick out the pointers from x we'll use
        x_ptrs = x_ptr + x_row_offsets * stride_xm + x_col_offsets * stride_xk
        
        # if a column offset gets larger than the hidden size, we don't want to operate on it, so mask it
        x_mask = (x_col_offsets < hidden_size)
        # load up the pointers we've decided on, mask it out if necessary
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load BLOCK_K x BLOCK_SIZE tile from weights
        # which column we're operating on, similar to x_col_offsets to pick up where we left off with last block
        w1_row_offsets = k + tl.arange(0, BLOCK_K)[:, None]
        # pick the correct columns, but crucially we're choosing which expert we're using!
        w1_col_offsets = col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
        # same as above's pointer choosing step, pick out w1's pointers
        w1_ptrs = w1_ptr + w1_row_offsets * stride_wk + w1_col_offsets * stride_wn
        
        # same masking
        w1_mask = (w1_row_offsets < hidden_size)
        # same loading
        w1_tile = tl.load(w1_ptrs, mask=w1_mask, other=0.0)
        
        # do the matmul between the tiles we've selected
        acc += tl.dot(x_tile, w1_tile)
    
    # Store BLOCK_SIZE x BLOCK_SIZE output block
    # where in the output matrix we'll go (which token block)
    output_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    # which expert/ffn columns we're writing to (col_idx tells us the expert!)
    output_col_offsets = col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    # again, pointwise multiplication, decide which pointers we're dealing with
    output_ptrs = output_ptr + output_row_offsets * stride_om + output_col_offsets * stride_on
    
    # store to the output matrix
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