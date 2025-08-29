import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from moe import sdd_kernel, dsd_kernel
from benchmark_moe import MoeMLP_ForLoop, MoeMLP_Batched

class TritonMoE(nn.Module):
    """MOE using our Triton kernels"""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.n_embd
        self.ffn_hidden_size = 4 * config.n_embd
        
        # Router (same as PyTorch versions)
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
        # Expert weights stored as single tensors (like MegaBlocks)
        # w1: (hidden_size, num_experts * ffn_hidden_size)
        # w2: (num_experts * ffn_hidden_size, hidden_size)
        self.w1 = nn.Parameter(torch.empty(self.hidden_size, self.num_experts * self.ffn_hidden_size))
        self.w2 = nn.Parameter(torch.empty(self.num_experts * self.ffn_hidden_size, self.hidden_size))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        
        # Block size for Triton kernels - needs to be tuned
        self.BLOCK_SIZE = 16
        
    def create_block_sparse_indices(self, selected_experts, routing_weights):
        """Create block-sparse indices for Triton kernels
        This mimics MegaBlocks' approach"""
        batch_size, seq_len = selected_experts.shape[0] // self.hidden_size, 1
        num_tokens = selected_experts.shape[0]
        
        # Flatten and sort by expert ID to group tokens going to same expert
        selected_experts_flat = selected_experts.view(-1)  # [num_tokens * k]
        expert_ids = selected_experts_flat
        token_ids = torch.arange(num_tokens, device=expert_ids.device).repeat_interleave(self.num_experts_per_tok)
        
        # Sort by expert to group tokens
        sorted_expert_ids, sort_idx = expert_ids.sort(stable=True)
        sorted_token_ids = token_ids[sort_idx]
        sorted_weights = routing_weights.view(-1)[sort_idx]
        
        # Count tokens per expert and create blocks
        # For simplicity, we'll process in blocks of BLOCK_SIZE tokens
        # Real MegaBlocks does more sophisticated blocking
        
        blocks = []
        current_expert = -1
        current_tokens = []
        
        for i in range(len(sorted_expert_ids)):
            expert = sorted_expert_ids[i].item()
            token = sorted_token_ids[i].item()
            
            if expert != current_expert:
                # Finish previous expert's blocks
                if current_tokens:
                    blocks.extend(self.create_blocks_for_tokens(current_tokens, current_expert))
                current_expert = expert
                current_tokens = [token]
            else:
                current_tokens.append(token)
        
        # Don't forget last expert
        if current_tokens:
            blocks.extend(self.create_blocks_for_tokens(current_tokens, current_expert))
        
        return blocks, sorted_weights
    
    def create_blocks_for_tokens(self, token_ids, expert_id):
        """Create blocks of BLOCK_SIZE for given tokens and expert"""
        blocks = []
        for i in range(0, len(token_ids), self.BLOCK_SIZE):
            block_tokens = token_ids[i:i+self.BLOCK_SIZE]
            # Pad if necessary
            while len(block_tokens) < self.BLOCK_SIZE:
                block_tokens.append(block_tokens[-1])  # Repeat last token as padding
            blocks.append({
                'tokens': block_tokens,
                'expert': expert_id,
                'block_idx': len(blocks)
            })
        return blocks
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        # Routing (same as PyTorch versions)
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        
        # Create block-sparse indices
        blocks, sorted_weights = self.create_block_sparse_indices(selected_experts, routing_weights)
        
        if not blocks:
            return torch.zeros_like(x), router_logits
        
        # Prepare tensors for kernels
        num_blocks = len(blocks)
        row_indices = torch.tensor([b['block_idx'] for b in blocks], dtype=torch.int32, device=x.device)
        weight_col_indices = torch.tensor([b['expert'] for b in blocks], dtype=torch.int32, device=x.device)
        output_col_indices = torch.arange(num_blocks, dtype=torch.int32, device=x.device)
        
        # Create padded input tensor
        total_padded_tokens = num_blocks * self.BLOCK_SIZE
        padded_x = torch.zeros(total_padded_tokens, hidden_dim, dtype=x.dtype, device=x.device)
        
        for i, block in enumerate(blocks):
            start_idx = i * self.BLOCK_SIZE
            for j, token_id in enumerate(block['tokens']):
                padded_x[start_idx + j] = x_flat[token_id]
        
        # Allocate output tensors
        hidden_output = torch.zeros(total_padded_tokens, self.ffn_hidden_size, 
                                   dtype=x.dtype, device=x.device)
        
        # Debug prints
        print(f"Debug: num_blocks={num_blocks}, BLOCK_SIZE={self.BLOCK_SIZE}")
        print(f"Debug: padded_x shape={padded_x.shape}, w1 shape={self.w1.shape}")
        print(f"Debug: hidden_output shape={hidden_output.shape}")
        print(f"Debug: First block tokens: {blocks[0] if blocks else 'No blocks'}")
        
        # Call SDD kernel (x @ w1)
        grid = (num_blocks,)
        sdd_kernel[grid](
            padded_x, self.w1, hidden_output,
            row_indices, weight_col_indices, output_col_indices,
            padded_x.stride(0), padded_x.stride(1),
            self.w1.stride(0), self.w1.stride(1),
            hidden_output.stride(0), hidden_output.stride(1),
            hidden_dim,
            BLOCK_SIZE=self.BLOCK_SIZE,
            BLOCK_K=min(32, hidden_dim)
        )
        
        # Apply GELU activation
        hidden_output = F.gelu(hidden_output)
        
        # Prepare for DSD kernel - now we need different indices
        # For DSD, we read from compact activation but write to full output
        final_output = torch.zeros(total_padded_tokens, hidden_dim, 
                                  dtype=x.dtype, device=x.device)
        
        # Weight row indices for w2 (which expert's w2 to use)
        weight_row_indices = torch.tensor([b['expert'] for b in blocks], dtype=torch.int32, device=x.device)
        
        # Call DSD kernel (hidden @ w2)
        grid = (num_blocks, (hidden_dim + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE)
        dsd_kernel[grid](
            hidden_output, self.w2, final_output,
            row_indices, weight_row_indices,
            hidden_output.stride(0), hidden_output.stride(1),
            self.w2.stride(0), self.w2.stride(1),
            final_output.stride(0), final_output.stride(1),
            self.ffn_hidden_size, hidden_dim,
            BLOCK_SIZE=self.BLOCK_SIZE,
            BLOCK_K=min(32, self.ffn_hidden_size)
        )
        
        # Unpack results back to original token positions with routing weights
        output = torch.zeros_like(x_flat)
        for i, block in enumerate(blocks):
            start_idx = i * self.BLOCK_SIZE
            for j, token_id in enumerate(block['tokens']):
                if j < len(block['tokens']):  # Skip padding
                    # Apply routing weight
                    # Note: this is simplified - need to properly track which weight goes with which token
                    output[token_id] += final_output[start_idx + j]
        
        return output.view(batch_size, seq_len, hidden_dim), router_logits


def test_triton_vs_pytorch():
    """Compare Triton implementation against PyTorch implementations"""
    print("="*60)
    print("Testing Triton MOE vs PyTorch Implementations")
    print("="*60)
    
    # Configuration
    batch_size = 2
    seq_len = 32
    hidden_dim = 256
    num_experts_val = 8
    num_experts_per_tok_val = 2
    device = 'cuda'
    dtype = torch.float32  # Use float32 for better numerical comparison
    
    class Config:
        n_embd = hidden_dim
        num_experts = num_experts_val
        num_experts_per_tok = num_experts_per_tok_val
        norm_topk_prob = True
        bias = False
        dropout = 0.0
    
    config = Config()
    
    # Create models
    triton_moe = TritonMoE(config).to(device).to(dtype)
    loop_moe = MoeMLP_ForLoop(config).to(device).to(dtype)
    batched_moe = MoeMLP_Batched(config).to(device).to(dtype)
    
    # Sync weights between implementations
    with torch.no_grad():
        # Copy router weights
        loop_moe.router.weight.data = triton_moe.router.weight.data.clone()
        batched_moe.router.weight.data = triton_moe.router.weight.data.clone()
        
        # Copy expert weights to loop implementation
        for i in range(config.num_experts):
            mlp = loop_moe.experts[i]
            # Extract expert i's weights from triton's combined tensors
            w1_start = i * (4 * hidden_dim)
            w1_end = (i + 1) * (4 * hidden_dim)
            mlp.c_fc.weight.data = triton_moe.w1[:, w1_start:w1_end].T.clone()
            mlp.c_proj.weight.data = triton_moe.w2[w1_start:w1_end, :].T.clone()
            
            # Copy to batched implementation
            batched_moe.expert_w1[i] = mlp.c_fc.weight.data.T.clone()
            batched_moe.expert_w2[i] = mlp.c_proj.weight.data.T.clone()
    
    # Test input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    
    # Run all implementations
    with torch.no_grad():
        print("\nRunning implementations...")
        
        # For-loop version
        output_loop, router_loop = loop_moe(x)
        print(f"For-loop output shape: {output_loop.shape}")
        print(f"For-loop output sample: {output_loop[0,0,:4]}")
        
        # Batched einops version  
        output_batched, router_batched = batched_moe(x)
        print(f"Batched output shape: {output_batched.shape}")
        print(f"Batched output sample: {output_batched[0,0,:4]}")
        
        # Triton version
        output_triton, router_triton = triton_moe(x)
        print(f"Triton output shape: {output_triton.shape}")
        print(f"Triton output sample: {output_triton[0,0,:4]}")
    
    # Compare outputs
    print("\n" + "="*40)
    print("Comparing outputs...")
    
    # Loop vs Batched
    loop_batch_diff = torch.abs(output_loop - output_batched).max().item()
    print(f"Loop vs Batched max diff: {loop_batch_diff:.2e}")
    
    # Loop vs Triton
    loop_triton_diff = torch.abs(output_loop - output_triton).max().item()
    print(f"Loop vs Triton max diff: {loop_triton_diff:.2e}")
    
    # Batched vs Triton
    batch_triton_diff = torch.abs(output_batched - output_triton).max().item()
    print(f"Batched vs Triton max diff: {batch_triton_diff:.2e}")
    
    # Check if close enough
    tolerance = 1e-3
    loop_batch_match = torch.allclose(output_loop, output_batched, atol=tolerance)
    loop_triton_match = torch.allclose(output_loop, output_triton, atol=tolerance)
    batch_triton_match = torch.allclose(output_batched, output_triton, atol=tolerance)
    
    print(f"\nMatching (tolerance={tolerance}):")
    print(f"Loop == Batched: {'✅' if loop_batch_match else '❌'}")
    print(f"Loop == Triton: {'✅' if loop_triton_match else '❌'}")
    print(f"Batched == Triton: {'✅' if batch_triton_match else '❌'}")
    
    return loop_triton_match and batch_triton_match


def benchmark_all_implementations():
    """Benchmark all three implementations"""
    print("\n" + "="*60)
    print("Benchmarking All MOE Implementations")
    print("="*60)
    
    # Configuration
    batch_size = 4
    seq_len = 128
    hidden_dim = 512
    num_experts_val = 16
    num_experts_per_tok_val = 4
    device = 'cuda'
    dtype = torch.bfloat16
    
    class Config:
        n_embd = hidden_dim
        num_experts = num_experts_val
        num_experts_per_tok = num_experts_per_tok_val
        norm_topk_prob = True
        bias = False
        dropout = 0.0
    
    config = Config()
    
    # Create models
    triton_moe = TritonMoE(config).to(device).to(dtype)
    loop_moe = MoeMLP_ForLoop(config).to(device).to(dtype)
    batched_moe = MoeMLP_Batched(config).to(device).to(dtype)
    
    # Compile PyTorch versions
    loop_moe = torch.compile(loop_moe)
    batched_moe = torch.compile(batched_moe)
    # Note: Triton kernels are already compiled
    
    # Test input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = loop_moe(x)
            _ = batched_moe(x)
            _ = triton_moe(x)
    
    # Benchmark
    num_runs = 100
    
    def time_model(model, x, num_runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        torch.cuda.synchronize()
        return (time.time() - start) / num_runs * 1000
    
    print(f"\nConfiguration:")
    print(f"  Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_dim}")
    print(f"  Experts: {num_experts}, K: {num_experts_per_tok}")
    print(f"  Runs: {num_runs}")
    
    print(f"\nResults:")
    time_loop = time_model(loop_moe, x, num_runs)
    print(f"  For-loop:     {time_loop:.2f} ms")
    
    time_batched = time_model(batched_moe, x, num_runs)
    print(f"  Batched:      {time_batched:.2f} ms")
    
    time_triton = time_model(triton_moe, x, num_runs)
    print(f"  Triton:       {time_triton:.2f} ms")
    
    print(f"\nSpeedups:")
    print(f"  Batched vs Loop:   {time_loop/time_batched:.2f}x")
    print(f"  Triton vs Loop:    {time_loop/time_triton:.2f}x")
    print(f"  Triton vs Batched: {time_batched/time_triton:.2f}x")


if __name__ == "__main__":
    # First test correctness
    correct = test_triton_vs_pytorch()
    
    if correct:
        print("\n✅ Triton implementation matches PyTorch!")
        # Then benchmark
        benchmark_all_implementations()
    else:
        print("\n❌ Triton implementation doesn't match PyTorch - fix correctness first!")