import torch
import triton
import triton.language as tl
import triton.testing
'''Note: Karpathy calls hidden_size the n_embd. ffn_hidden_size is 4*n_embd (or hidden size!)'''

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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}),
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
    ],
    key=['hidden_size'],
)
@triton.jit
def sdd_kernel(
    x_ptr, w1_ptr, output_ptr,
    row_indices_ptr, col_indices_ptr,
    stride_x_row, stride_w1_row, stride_output_row, stride_output_col,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """lots of comments because i'm still learning triton!"""
    pid = tl.program_id(0)
    
    row_idx = tl.load(row_indices_ptr + pid) # figure out which row/col we want to operate on: each pid gets it's own block
    col_idx = tl.load(col_indices_ptr + pid)
    
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32) # initialize the *one block* we're computing per program
    
    # one output block of block_size x block_size will take multiple block_size x block_size tiles to compute, so loop
    for k in range(0, hidden_size, BLOCK_SIZE): # loop to hidden_size taking block size steps
        x_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] # add the beginning pointer we're working with to the arange of block sizes giving us a vector of the pointers we're operating on.
        x_col_offsets = k + tl.arange(0, BLOCK_SIZE)[None, :] # add which column we're currently operating on to the arange of block sizes
        x_ptrs = x_ptr + x_row_offsets * stride_x_row + x_col_offsets #(pointwise multiplication) using the numbers from above, pick out the pointers from x we'll use
        
        x_mask = (x_col_offsets < hidden_size) # if a column offset gets larger than the hidden size, we don't want to operate on it, so mask it.
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0) #load up the pointers we've decided on, mask it out if necessary.
        
        w1_row_offsets = k + tl.arange(0, BLOCK_SIZE)[:, None] # which column we're operating on, similar to x_col_offsets to pick up where we left off with last block
        w1_col_offsets = col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :] # pick the correct columns, but crucially we're choosing which expert we're using!
        w1_ptrs = w1_ptr + w1_row_offsets * stride_w1_row + w1_col_offsets #same as above's pointer choosing step, pick out w1's pointers.
        
        w1_mask = (w1_row_offsets < hidden_size) #same masking
        w1_tile = tl.load(w1_ptrs, mask=w1_mask, other=0.0) # same loading
        
        acc += tl.dot(x_tile, w1_tile) # do the matmul between the tiles we've selected
    
    output_row_offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] #where in the output matrix we'll go
    output_col_offsets = col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :] #just need the block size for column since we only compute block size blocks
    output_ptrs = output_ptr + output_row_offsets * stride_output_row + output_col_offsets * stride_output_col
    tl.store(output_ptrs, acc) #store to the output matrix


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