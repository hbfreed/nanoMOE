import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math
from moe import dsd_backward_act_kernel

def _create_sparse_indices(token_expert_indices, num_experts, d_ffn, block_size):
    """
    Create indices for sparse computation based on token-expert assignments.
    Returns indices needed for the backward pass kernel.
    """
    batch_size, seq_len = token_expert_indices.shape
    total_tokens = batch_size * seq_len
    
    # Create mapping from linear token index to expert
    token_to_expert = token_expert_indices.flatten()
    
    # Group tokens by expert and sort
    expert_assignments = []
    for expert_id in range(num_experts):
        expert_tokens = torch.where(token_to_expert == expert_id)[0]
        expert_assignments.append(expert_tokens)
    
    # Pad tokens per expert to multiples of block_size
    padded_tokens_per_expert = []
    for expert_tokens in expert_assignments:
        num_tokens = len(expert_tokens)
        if num_tokens == 0:
            padded_tokens_per_expert.append(0)
            continue
        padded_count = ((num_tokens + block_size - 1) // block_size) * block_size
        padded_tokens_per_expert.append(padded_count)
    
    total_padded_tokens = sum(padded_tokens_per_expert)
    
    # Create indices for kernel
    row_indices = []
    weight_col_indices = []
    output_col_indices = []
    
    current_row_offset = 0
    current_output_offset = 0
    
    for expert_id, expert_tokens in enumerate(expert_assignments):
        num_padded = padded_tokens_per_expert[expert_id]
        if num_padded == 0:
            continue
            
        num_blocks = num_padded // block_size
        d_ffn_blocks = (d_ffn + block_size - 1) // block_size
        
        for row_block in range(num_blocks):
            for col_block in range(d_ffn_blocks):
                row_indices.append(current_row_offset + row_block)
                weight_col_indices.append(expert_id * d_ffn_blocks + col_block)
                output_col_indices.append(current_output_offset + col_block)
        
        current_row_offset += num_blocks
        current_output_offset += d_ffn_blocks
    
    return (torch.tensor(row_indices, dtype=torch.int32),
            torch.tensor(weight_col_indices, dtype=torch.int32),
            torch.tensor(output_col_indices, dtype=torch.int32),
            total_padded_tokens)

def create_sorted_token_mapping(token_expert_indices, num_experts, block_size):
    """
    Create mapping from original token positions to sorted positions.
    Returns sorted token indices and reverse mapping.
    """
    batch_size, seq_len = token_expert_indices.shape
    total_tokens = batch_size * seq_len
    
    token_to_expert = token_expert_indices.flatten()
    
    # Sort tokens by expert assignment
    sorted_token_indices = []
    original_positions = []
    
    for expert_id in range(num_experts):
        expert_tokens = torch.where(token_to_expert == expert_id)[0]
        sorted_token_indices.extend(expert_tokens.tolist())
        original_positions.extend(expert_tokens.tolist())
    
    # Create reverse mapping
    reverse_mapping = torch.zeros(total_tokens, dtype=torch.long)
    for new_pos, orig_pos in enumerate(sorted_token_indices):
        reverse_mapping[orig_pos] = new_pos
    
    return torch.tensor(sorted_token_indices), reverse_mapping

def test_dsd_backward_act_simple():
    """Test with hand-calculated values for verification"""
    print("="*60)
    print("TEST 1: Simple Unit Test (Hand-Calculated)")
    print("="*60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU-dependent tests")
        return False
    
    device = torch.device('cuda')
    
    # Simple setup: 2 tokens, 2 experts, larger dimensions for Triton requirements
    batch_size, seq_len = 1, 2
    hidden_size = 32  # Increased to meet Triton requirements
    d_ffn = 32        # Increased to meet Triton requirements
    num_experts = 2
    block_size = 16   # Increased to meet Triton requirements
    
    # Simple routing: token 0 -> expert 0, token 1 -> expert 1
    token_expert_indices = torch.tensor([[0, 1]], dtype=torch.long)
    
    # Create simple grad_output (2 tokens x 32 hidden) - move to GPU
    grad_output = torch.zeros(2, hidden_size, dtype=torch.float32, device=device)
    grad_output[0, 0] = 1.0  # token 0
    grad_output[1, 1] = 1.0  # token 1
    
    # Create simple w2 weights (2 experts * 32 d_ffn x 32 hidden) - move to GPU
    w2 = torch.zeros(num_experts * d_ffn, hidden_size, dtype=torch.float32, device=device)
    # Expert 0 weights (rows 0-31)
    w2[0, 0] = 1.0  # First row, first column
    w2[1, 1] = 1.0  # Second row, second column
    # Expert 1 weights (rows 32-63)  
    w2[33, 1] = 1.0   # grad_output[1] @ w2[33, :] = [0,1,0,...] @ [0,1,0,...] = 1 (result[1])
    
    w2_t = w2.T  # Transpose for kernel (32 x 64)
    
    print("grad_output shape:", grad_output.shape)
    print("w2 shape:", w2.shape) 
    print("w2.T shape:", w2_t.shape)
    
    # Manual calculation of expected result:
    # For token 0 (expert 0): grad_output[0] @ w2[0:32].T
    # For token 1 (expert 1): grad_output[1] @ w2[32:64].T
    
    expected_grad_act = torch.zeros(2, d_ffn, dtype=torch.float32, device=device)
    # Token 0: expert 0 weights
    expected_grad_act[0] = grad_output[0] @ w2[:d_ffn].T
    # Token 1: expert 1 weights  
    expected_grad_act[1] = grad_output[1] @ w2[d_ffn:2*d_ffn].T
    
    print("Manual check - Token 0:")
    print("grad_output[0]:", grad_output[0][:4])
    print("w2[0:4, 0:4]:", w2[0:4, 0:4])
    print("expected_grad_act[0]:", expected_grad_act[0][:4])
    
    print("Manual check - Token 1:")
    print("grad_output[1]:", grad_output[1][:4])
    print("w2[32:36, 0:4]:", w2[32:36, 0:4])
    print("expected_grad_act[1]:", expected_grad_act[1][:4])
    
    print("\nExpected grad_act:")
    print(expected_grad_act)
    
    # Create indices for kernel
    row_indices, weight_col_indices, output_col_indices, total_padded_tokens = _create_sparse_indices(
        token_expert_indices, num_experts, d_ffn, block_size
    )
    
    print(f"\nKernel indices:")
    print(f"row_indices: {row_indices}")
    print(f"weight_col_indices: {weight_col_indices}")
    print(f"output_col_indices: {output_col_indices}")
    print(f"total_padded_tokens: {total_padded_tokens}")
    
    # Prepare tensors for kernel
    # Need to sort grad_output by expert assignment
    sorted_token_indices, reverse_mapping = create_sorted_token_mapping(token_expert_indices, num_experts, block_size)
    grad_output_sorted = grad_output[sorted_token_indices]
    
    # Pad grad_output to total_padded_tokens
    if total_padded_tokens > grad_output_sorted.shape[0]:
        padding = torch.zeros(total_padded_tokens - grad_output_sorted.shape[0], hidden_size, device=device)
        grad_output_padded = torch.cat([grad_output_sorted, padding], dim=0)
    else:
        grad_output_padded = grad_output_sorted
    
    # Create output tensor
    output_d_ffn = sum((d_ffn + block_size - 1) // block_size for _ in range(num_experts)) * block_size
    grad_act_compact = torch.zeros(total_padded_tokens, output_d_ffn, dtype=torch.float32, device=device)
    
    print(f"\nPadded grad_output shape: {grad_output_padded.shape}")
    print(f"grad_act_compact shape: {grad_act_compact.shape}")
    
    # Move indices to GPU
    row_indices = row_indices.cuda()
    weight_col_indices = weight_col_indices.cuda()
    output_col_indices = output_col_indices.cuda()
    
    # Launch kernel
    grid = (len(row_indices),)
    dsd_backward_act_kernel[grid](
        grad_output_padded, w2_t, grad_act_compact,
        row_indices, weight_col_indices, output_col_indices,
        grad_output_padded.stride(0), grad_output_padded.stride(1),
        w2_t.stride(0), w2_t.stride(1),
        grad_act_compact.stride(0), grad_act_compact.stride(1),
        hidden_size,
        BLOCK_SIZE=block_size, BLOCK_K=16
    )
    
    print("\nKernel output (grad_act_compact):")
    print(grad_act_compact[:2, :d_ffn])  # First 2 tokens, full d_ffn features
    
    # Verify results
    tolerance = 1e-6
    result_subset = grad_act_compact[:2, :d_ffn]  # Extract the relevant part
    max_error = torch.max(torch.abs(result_subset - expected_grad_act)).item()
    
    print(f"\nMax error: {max_error:.6f}")
    
    if max_error < tolerance:
        print("✓ Simple test PASSED!")
        return True
    else:
        print("❌ Simple test FAILED!")
        print("Expected:")
        print(expected_grad_act)
        print("Got:")
        print(result_subset)
        return False

def test_dsd_backward_act_integration():
    """Test against PyTorch autograd with realistic dimensions"""
    print("="*60)
    print("TEST 2: Integration Test (vs PyTorch)")
    print("="*60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU-dependent tests")
        return False
    
    device = torch.device('cuda')
    
    # Realistic dimensions
    batch_size, seq_len = 2, 8
    hidden_size = 64
    d_ffn = 128
    num_experts = 4
    block_size = 16
    
    total_tokens = batch_size * seq_len
    
    # Random routing
    torch.manual_seed(42)
    token_expert_indices = torch.randint(0, num_experts, (batch_size, seq_len))
    
    # Create test tensors - move to GPU
    grad_output = torch.randn(total_tokens, hidden_size, dtype=torch.float32, device=device)
    w2 = torch.randn(num_experts * d_ffn, hidden_size, dtype=torch.float32, device=device) * 0.1
    w2_t = w2.T
    
    print(f"Dimensions: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}")
    print(f"d_ffn={d_ffn}, num_experts={num_experts}, block_size={block_size}")
    
    # Compute reference using PyTorch
    token_to_expert = token_expert_indices.flatten()
    expected_grad_act = torch.zeros(total_tokens, d_ffn, dtype=torch.float32, device=device)
    
    for token_idx in range(total_tokens):
        expert_id = token_to_expert[token_idx].item()
        expert_w2 = w2[expert_id * d_ffn:(expert_id + 1) * d_ffn, :]
        expected_grad_act[token_idx] = grad_output[token_idx] @ expert_w2.T
    
    # Create indices for kernel
    row_indices, weight_col_indices, output_col_indices, total_padded_tokens = _create_sparse_indices(
        token_expert_indices, num_experts, d_ffn, block_size
    )
    
    # Sort tokens for kernel
    sorted_token_indices, reverse_mapping = create_sorted_token_mapping(token_expert_indices, num_experts, block_size)
    grad_output_sorted = grad_output[sorted_token_indices]
    
    # Pad to block boundaries
    if total_padded_tokens > grad_output_sorted.shape[0]:
        padding = torch.zeros(total_padded_tokens - grad_output_sorted.shape[0], hidden_size, device=device)
        grad_output_padded = torch.cat([grad_output_sorted, padding], dim=0)
    else:
        grad_output_padded = grad_output_sorted
    
    # Create output tensor
    output_d_ffn = sum((d_ffn + block_size - 1) // block_size for _ in range(num_experts)) * block_size
    grad_act_compact = torch.zeros(total_padded_tokens, output_d_ffn, dtype=torch.float32, device=device)
    
    # Move indices to GPU
    row_indices = row_indices.cuda()
    weight_col_indices = weight_col_indices.cuda()
    output_col_indices = output_col_indices.cuda()
    
    # Launch kernel
    grid = (len(row_indices),)
    dsd_backward_act_kernel[grid](
        grad_output_padded, w2_t, grad_act_compact,
        row_indices, weight_col_indices, output_col_indices,
        grad_output_padded.stride(0), grad_output_padded.stride(1),
        w2_t.stride(0), w2_t.stride(1),
        grad_act_compact.stride(0), grad_act_compact.stride(1),
        hidden_size,
        BLOCK_SIZE=block_size, BLOCK_K=16
    )
    
    # Extract and unsort results
    kernel_result_sorted = grad_act_compact[:len(sorted_token_indices), :d_ffn]
    kernel_result = torch.zeros_like(expected_grad_act)
    
    # Debug the unsorting process
    print(f"sorted_token_indices: {sorted_token_indices}")
    print(f"kernel_result_sorted shape: {kernel_result_sorted.shape}")
    print(f"expected_grad_act shape: {expected_grad_act.shape}")
    
    # Correct unsorting: put each sorted result back to its original position
    for i, orig_pos in enumerate(sorted_token_indices):
        kernel_result[orig_pos] = kernel_result_sorted[i]
    
    # Compare results
    max_error = torch.max(torch.abs(kernel_result - expected_grad_act)).item()
    mean_error = torch.mean(torch.abs(kernel_result - expected_grad_act)).item()
    
    print(f"Max error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")
    
    # Debug: show some sample comparisons
    print(f"Sample kernel result [0, :5]: {kernel_result[0, :5]}")
    print(f"Sample expected result [0, :5]: {expected_grad_act[0, :5]}")
    print(f"Sample kernel result [1, :5]: {kernel_result[1, :5]}")
    print(f"Sample expected result [1, :5]: {expected_grad_act[1, :5]}")
    
    rtol, atol = 1e-4, 1e-5
    if torch.allclose(kernel_result, expected_grad_act, rtol=rtol, atol=atol):
        print("✓ Integration test PASSED!")
        return True
    else:
        print("❌ Integration test FAILED!")
        return False

def test_dsd_backward_edge_cases():
    """Test edge cases and special scenarios"""
    print("="*60)
    print("TEST 3: Edge Cases")
    print("="*60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU-dependent tests")
        return False
    
    device = torch.device('cuda')
    all_passed = True
    
    # Test 1: Single token
    print("Edge case 1: Single token")
    try:
        batch_size, seq_len = 1, 1
        hidden_size = 32
        d_ffn = 64
        num_experts = 2
        block_size = 16  # Meet Triton requirements
        
        token_expert_indices = torch.tensor([[0]])
        grad_output = torch.randn(1, hidden_size, device=device)
        w2 = torch.randn(num_experts * d_ffn, hidden_size, device=device) * 0.1
        
        row_indices, weight_col_indices, output_col_indices, total_padded_tokens = _create_sparse_indices(
            token_expert_indices, num_experts, d_ffn, block_size
        )
        
        grad_output_padded = torch.zeros(total_padded_tokens, hidden_size, device=device)
        grad_output_padded[0] = grad_output[0]
        
        output_d_ffn = sum((d_ffn + block_size - 1) // block_size for _ in range(num_experts)) * block_size
        grad_act_compact = torch.zeros(total_padded_tokens, output_d_ffn, device=device)
        
        # Move indices to GPU
        row_indices = row_indices.cuda()
        weight_col_indices = weight_col_indices.cuda()
        output_col_indices = output_col_indices.cuda()
        
        if len(row_indices) > 0:
            grid = (len(row_indices),)
            dsd_backward_act_kernel[grid](
                grad_output_padded, w2.T, grad_act_compact,
                row_indices, weight_col_indices, output_col_indices,
                grad_output_padded.stride(0), grad_output_padded.stride(1),
                w2.T.stride(0), w2.T.stride(1),
                grad_act_compact.stride(0), grad_act_compact.stride(1),
                hidden_size,
                BLOCK_SIZE=block_size, BLOCK_K=16
            )
        
        print("✓ Single token test passed")
    except Exception as e:
        print(f"❌ Single token test failed: {e}")
        all_passed = False
    
    # Test 2: All tokens to same expert
    print("Edge case 2: All tokens to same expert")
    try:
        batch_size, seq_len = 2, 4
        hidden_size = 32
        d_ffn = 64
        num_experts = 3
        block_size = 16
        
        token_expert_indices = torch.ones(batch_size, seq_len, dtype=torch.long)  # All to expert 1
        total_tokens = batch_size * seq_len
        
        grad_output = torch.randn(total_tokens, hidden_size, device=device)
        w2 = torch.randn(num_experts * d_ffn, hidden_size, device=device) * 0.1
        
        row_indices, weight_col_indices, output_col_indices, total_padded_tokens = _create_sparse_indices(
            token_expert_indices, num_experts, d_ffn, block_size
        )
        
        grad_output_padded = torch.zeros(total_padded_tokens, hidden_size, device=device)
        grad_output_padded[:total_tokens] = grad_output
        
        output_d_ffn = sum((d_ffn + block_size - 1) // block_size for _ in range(num_experts)) * block_size
        grad_act_compact = torch.zeros(total_padded_tokens, output_d_ffn, device=device)
        
        # Move indices to GPU
        row_indices = row_indices.cuda()
        weight_col_indices = weight_col_indices.cuda()
        output_col_indices = output_col_indices.cuda()
        
        if len(row_indices) > 0:
            grid = (len(row_indices),)
            dsd_backward_act_kernel[grid](
                grad_output_padded, w2.T, grad_act_compact,
                row_indices, weight_col_indices, output_col_indices,
                grad_output_padded.stride(0), grad_output_padded.stride(1),
                w2.T.stride(0), w2.T.stride(1),
                grad_act_compact.stride(0), grad_act_compact.stride(1),
                hidden_size,
                BLOCK_SIZE=block_size, BLOCK_K=16
            )
        
        print("✓ Same expert test passed")
    except Exception as e:
        print(f"❌ Same expert test failed: {e}")
        all_passed = False
    
    # Test 3: Uneven dimensions
    print("Edge case 3: Uneven dimensions")
    try:
        batch_size, seq_len = 1, 3
        hidden_size = 48  # Keep somewhat uneven but work with larger blocks
        d_ffn = 80       # Keep somewhat uneven but work with larger blocks
        num_experts = 2
        block_size = 16
        
        token_expert_indices = torch.tensor([[0, 1, 0]])
        total_tokens = batch_size * seq_len
        
        grad_output = torch.randn(total_tokens, hidden_size, device=device)
        w2 = torch.randn(num_experts * d_ffn, hidden_size, device=device) * 0.1
        
        row_indices, weight_col_indices, output_col_indices, total_padded_tokens = _create_sparse_indices(
            token_expert_indices, num_experts, d_ffn, block_size
        )
        
        sorted_token_indices, _ = create_sorted_token_mapping(token_expert_indices, num_experts, block_size)
        grad_output_sorted = grad_output[sorted_token_indices]
        
        grad_output_padded = torch.zeros(total_padded_tokens, hidden_size, device=device)
        grad_output_padded[:len(sorted_token_indices)] = grad_output_sorted
        
        output_d_ffn = sum((d_ffn + block_size - 1) // block_size for _ in range(num_experts)) * block_size
        grad_act_compact = torch.zeros(total_padded_tokens, output_d_ffn, device=device)
        
        # Move indices to GPU
        row_indices = row_indices.cuda()
        weight_col_indices = weight_col_indices.cuda()
        output_col_indices = output_col_indices.cuda()
        
        if len(row_indices) > 0:
            grid = (len(row_indices),)
            dsd_backward_act_kernel[grid](
                grad_output_padded, w2.T, grad_act_compact,
                row_indices, weight_col_indices, output_col_indices,
                grad_output_padded.stride(0), grad_output_padded.stride(1),
                w2.T.stride(0), w2.T.stride(1),
                grad_act_compact.stride(0), grad_act_compact.stride(1),
                hidden_size,
                BLOCK_SIZE=block_size, BLOCK_K=16
            )
        
        print("✓ Uneven dimensions test passed")
    except Exception as e:
        print(f"❌ Uneven dimensions test failed: {e}")
        all_passed = False
    
    if all_passed:
        print("✓ All edge cases PASSED!")
    else:
        print("❌ Some edge cases FAILED!")
    
    return all_passed

def run_all_tests():
    """Run all tests and report results"""
    tests = [
        test_dsd_backward_act_simple,
        test_dsd_backward_act_integration,
        test_dsd_backward_edge_cases,
    ]
    
    results = []
    for test_fn in tests:
        try:
            success = test_fn()
            results.append((test_fn.__name__, success))
        except Exception as e:
            print(f"❌ {test_fn.__name__} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(s for _, s in results)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print("\n⚠️ SOME TESTS FAILED")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()