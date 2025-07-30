import torch
import triton
import triton.language as tl
'''Note: Karpathy calls hidden_size the n_embd. ffn_hidden_size is 4*n_embd (or hidden size!)'''

@triton.jit
def sdd_kernel(
    x_ptr, # permuted tokens, dense [num_tokens, hidden_size]
    w1_ptr, # weight matrix, dense [hidden_size, ffn_hidden * num_experts]
    output_ptr, # output matrix, sparse! [num_tokens, ffn_hidden * num_experts]
    row_indices_ptr, # which row block each active block is in
    col_indices_ptr, # which col block (expert) each active block is in
    M, # num_tokens (padded to multiple of BLOCK_M)
    N, # ffn_hidden per expert (d_ffn)
    K, # hidden_size
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, # 128 (from paper, probably autotune this?)
    BLOCK_N: tl.constexpr, # 128 (from paper, probably autotune this too?)
    BLOCK_K: tl.constexpr, # for inner loop tiling
):
    block_id = tl.program_id(0)
    
    row_idx = tl.load(row_indices_ptr + block_id)
    col_idx = tl.load(col_indices_ptr + block_id)
    
    row_start = row_idx * BLOCK_M  # token offset
    col_start = col_idx * BLOCK_N  # expert offset
    
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = col_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k + offs_k[None, :]) * stride_xk
        x_block = tl.load(x_ptrs, mask=(k + offs_k[None, :] < K), other=0.0)
        
        w_ptrs = w1_ptr + (k + offs_k[:, None]) * stride_wk + offs_n[None, :] * stride_wn
        w_block = tl.load(w_ptrs, mask=(k + offs_k[:, None] < K), other=0.0)
        
        acc += tl.dot(x_block, w_block)
    
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc)


@triton.jit
def dsd():
    pass

@triton.jit
def gelu():
    pass

@triton.jit
def padded_scatter():
    pass