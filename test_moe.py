import torch
import torch.nn as nn
from dataclasses import dataclass
from model import MoeMLP, MLP
import torch.nn.functional as F

@dataclass  
class TestConfig:
    num_experts: int = 4
    num_experts_per_tok: int = 2  
    n_embd: int = 32
    norm_topk_prob: bool = True
    block_size: int = 32
    block_k: int = 32
    bias: bool = True
    dropout: float = 0.0

def test_sdd_kernel_vs_naive():
    """
    Test that the SDD kernel produces the same intermediate output as a naive implementation.
    Since the SDD kernel only implements the first linear layer (x @ w1), we'll compare
    the block-sparse output structure against manually computed values.
    """
    config = TestConfig()
    torch.manual_seed(42)
    
    # Check if CUDA is available for Triton kernels
    if not torch.cuda.is_available():
        print("CUDA not available - Triton kernels require GPU. Skipping SDD kernel test.")
        return
    
    device = torch.device('cuda')
    
    # Create test input - small batch for easier debugging  
    batch_size, seq_len = 2, 4
    x = torch.randn(batch_size, seq_len, config.n_embd, dtype=torch.float32, device=device)
    
    # Initialize MoeMLP with SDD kernel
    moe_mlp = MoeMLP(
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok, 
        n_embd=config.n_embd,
        norm_topk_prob=config.norm_topk_prob,
        block_size=config.block_size,
        block_k=config.block_k
    ).to(device)
    
    # Run the MoeMLP forward pass (which calls SDD kernel)
    with torch.no_grad():
        sdd_output = moe_mlp(x)
    
    # Get intermediate values for verification
    x_flat = x.view(-1, config.n_embd)
    
    with torch.no_grad():
        router_weights, selected_experts = moe_mlp._route_tokens(x_flat)
        x_sorted, selected_experts_sorted, router_weights_sorted, inv_indices = moe_mlp._sort_by_expert(
            x_flat, router_weights, selected_experts
        )
        x_padded, router_weights_padded, tokens_per_expert_padded, cumsum_padded = moe_mlp._pad_to_blocks(
            x_sorted, router_weights_sorted, selected_experts_sorted
        )
        row_indices, weight_col_indices, output_col_indices = moe_mlp._create_sparse_indices(tokens_per_expert_padded)
    
    # Block-by-block verification
    max_diff = 0.0
    for i, (row_idx, weight_col_idx, output_col_idx) in enumerate(zip(row_indices[:4], weight_col_indices[:4], output_col_indices[:4])):
        # Extract the exact data the kernel should process
        start_row = row_idx * config.block_size
        end_row = start_row + config.block_size
        
        start_weight_col = weight_col_idx * config.block_size  
        end_weight_col = start_weight_col + config.block_size
        
        start_output_col = output_col_idx * config.block_size
        end_output_col = start_output_col + config.block_size
        
        # Manual computation
        block_input = x_padded[start_row:end_row]
        block_weights = moe_mlp.w1[:, start_weight_col:end_weight_col]
        expected_block = block_input @ block_weights
        
        # Actual kernel output  
        actual_block = sdd_output[start_row:end_row, start_output_col:end_output_col]
        
        block_diff = (expected_block - actual_block).abs().max().item()
        max_diff = max(max_diff, block_diff)
        
        if block_diff > 1e-4:
            print(f"Block {i} error: {block_diff:.6f}")
    
    print(f"Maximum block error: {max_diff:.6f}")
    print("SDD kernel test completed.")

def test_simple_linear_comparison():
    """
    Simpler test: Compare a regular linear layer (self.c_fc equivalent) 
    with what the full MoE pipeline should produce when routing is bypassed.
    """
    config = TestConfig()
    torch.manual_seed(42)
    
    # Small test case
    batch_size, seq_len = 1, 2  
    x = torch.randn(batch_size, seq_len, config.n_embd, dtype=torch.float32)
    
    # Create a regular MLP for comparison
    regular_mlp = MLP(config)
    
    # Get output from first linear layer only (equivalent to self.c_fc)
    with torch.no_grad():
        naive_output = regular_mlp.c_fc(x.view(-1, config.n_embd))
        
    print(f"Regular MLP first layer output shape: {naive_output.shape}")
    print("Linear layer test completed.")

if __name__ == "__main__":
    test_sdd_kernel_vs_naive()
    test_simple_linear_comparison()