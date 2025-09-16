import torch
from moe_optimized import DSD

def main():
    if not torch.cuda.is_available():
        print("CUDA required!")
        return

    device = torch.device("cuda")
    
    # Simple test parameters
    batch_size = 32
    d_ffn = 256  # FFN dimension
    hidden_size = 128  # Output hidden size
    num_experts = 4
    block_size = 64
    
    print(f"Testing DSD with:")
    print(f"  Batch size: {batch_size}")
    print(f"  FFN dim: {d_ffn}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num experts: {num_experts}")
    print(f"  Block size: {block_size}")
    
    # Create test data
    x = torch.randn(batch_size, d_ffn, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(num_experts * d_ffn, hidden_size, device=device, dtype=torch.bfloat16)
    
    # Create expert mapping (each token block belongs to expert 0 for simplicity)
    m_block_to_expert = torch.zeros(1, dtype=torch.int32, device=device)  # 1 block, all expert 0
    n_block_to_expert = torch.zeros(1, dtype=torch.int32, device=device)  # Not used in DSD
    
    # Weight offsets for experts
    weight_offsets = torch.arange(0, (num_experts + 1) * d_ffn, d_ffn, dtype=torch.int32, device=device)
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  w2: {w2.shape}")
    print(f"  m_block_to_expert: {m_block_to_expert.shape}")
    print(f"  weight_offsets: {weight_offsets}")
    
    print("\nRunning DSD...")
    
    try:
        # Run DSD
        output = DSD.apply(x, w2, m_block_to_expert, n_block_to_expert, weight_offsets, block_size)
        
        print(f"Success! Output shape: {output.shape}")
        print(f"Output sample: {output[0, :5]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 