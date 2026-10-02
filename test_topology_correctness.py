"""
Correctness test for topology_var.py optimization.
Compares the optimized tensor-based implementation against a reference Python loop.
"""
import torch

# Reference implementation (original Python loop)
def topology_var_reference(padded_bins, expert_block_counts, block_size):
    """Original Python loop implementation for reference."""
    total_nnz = 0
    output_offsets = [0]

    for expert_id in range(expert_block_counts.numel()):
        if expert_id == 0:
            expert_token_blocks = padded_bins[0].item() // block_size
        else:
            expert_token_blocks = (
                padded_bins[expert_id].item() - padded_bins[expert_id - 1].item()
            ) // block_size
        expert_output_size = (
            expert_token_blocks * expert_block_counts[expert_id].item()
        )
        total_nnz += expert_output_size
        output_offsets.append(output_offsets[-1] + expert_output_size)

    output_offsets_tensor = torch.tensor(
        output_offsets[:-1], dtype=torch.int32, device=padded_bins.device
    )
    return total_nnz, output_offsets_tensor


# Optimized implementation (tensor ops)
def topology_var_optimized(padded_bins, expert_block_counts, block_size):
    """Optimized tensor-based implementation."""
    prepend = padded_bins.new_zeros(1)
    bin_sizes = torch.diff(padded_bins, prepend=prepend)
    expert_token_blocks = (bin_sizes // block_size).to(torch.int32)

    expert_output_sizes = expert_token_blocks * expert_block_counts.to(torch.int32)
    total_nnz = expert_output_sizes.sum().item()

    cumsum = expert_output_sizes.cumsum(0)
    output_offsets_tensor = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=padded_bins.device),
        cumsum[:-1]
    ]).to(torch.int32)

    return total_nnz, output_offsets_tensor


def test_correctness():
    """Test that optimized implementation produces identical results."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test cases with different configurations
    test_cases = [
        # (num_experts, block_size, tokens_per_expert_range, expert_block_counts)
        (8, 128, (100, 500), [16, 32, 16, 32, 16, 32, 16, 32]),  # Variable experts
        (8, 128, (0, 1000), [16, 16, 16, 16, 16, 16, 16, 16]),   # Uniform experts
        (4, 128, (50, 300), [8, 16, 8, 16]),                     # Fewer experts
        (16, 128, (100, 400), [8]*8 + [32]*8),                   # More experts
        (8, 128, (0, 0), [16, 32, 16, 32, 16, 32, 16, 32]),      # Edge case: no tokens
    ]

    print("Testing correctness of optimized topology_var implementation...")
    print("=" * 60)

    all_passed = True
    for i, (num_experts, block_size, token_range, block_counts) in enumerate(test_cases):
        # Generate random padded_bins (cumulative token counts)
        if token_range[1] > 0:
            tokens_per_expert = torch.randint(
                token_range[0], token_range[1], (num_experts,),
                device=device, dtype=torch.int64
            )
            # Pad to block size
            tokens_per_expert = ((tokens_per_expert + block_size - 1) // block_size) * block_size
        else:
            tokens_per_expert = torch.zeros(num_experts, device=device, dtype=torch.int64)

        padded_bins = tokens_per_expert.cumsum(0)
        expert_block_counts = torch.tensor(block_counts, device=device, dtype=torch.int32)

        # Run both implementations
        ref_nnz, ref_offsets = topology_var_reference(padded_bins, expert_block_counts, block_size)
        opt_nnz, opt_offsets = topology_var_optimized(padded_bins, expert_block_counts, block_size)

        # Compare
        nnz_match = ref_nnz == opt_nnz
        offsets_match = torch.equal(ref_offsets, opt_offsets)

        status = "PASS" if (nnz_match and offsets_match) else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"Test {i+1}: {status}")
        print(f"  Config: {num_experts} experts, block_size={block_size}")
        print(f"  total_nnz: ref={ref_nnz}, opt={opt_nnz}, match={nnz_match}")
        print(f"  offsets: match={offsets_match}")
        if not offsets_match:
            print(f"    ref_offsets: {ref_offsets.tolist()}")
            print(f"    opt_offsets: {opt_offsets.tolist()}")
        print()

    print("=" * 60)
    if all_passed:
        print("All tests PASSED! Optimized implementation is correct.")
    else:
        print("Some tests FAILED! Check implementation.")

    return all_passed


if __name__ == "__main__":
    test_correctness()
