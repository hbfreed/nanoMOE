#include <cstdint>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

namespace megablocks {
namespace construct_indices {

// We expect the number of outputs per block to be small. For
// example, with ffn_hidden_size=4096, we only need to write
// 32 elements per block per iteration.
const int kThreadsPerBlock = 32;

__global__ void __launch_bounds__(kThreadsPerBlock)
  ConstructIndicesKernel(short * __restrict__ indices,
			 const int * __restrict__ expert_block_counts,  // Number of blocks per expert
			 const int * __restrict__ expert_block_offsets, // Cumulative offsets into weight matrix
			 const int * __restrict__ output_offsets,        // Cumulative offsets into output array
			 int block_size,
			 const int * __restrict__ padded_bins) {
  // Load the offset for this bins indices.
  int start = 0;
  if (blockIdx.x > 0) start = __ldg(padded_bins + blockIdx.x - 1);
  int end = __ldg(padded_bins + blockIdx.x);

  // Divide the start and end into blocks.
  start /= block_size;
  end /= block_size;

  // Load expert-specific block count and offset
  int expert_id = blockIdx.x;
  int blocks_for_this_expert = __ldg(expert_block_counts + expert_id);
  int expert_base_offset = __ldg(expert_block_offsets + expert_id);

  // NEW: Use pre-computed output offset instead of calculating it
  int output_start = __ldg(output_offsets + expert_id);

  // Offset the output buffer to the start of this expert's output
  indices += output_start + blockIdx.y * blocks_for_this_expert + threadIdx.x;

  // Write the indices to the output.
  int bin_offset = blockIdx.y;
  int num_rows = end - start;
  for (; bin_offset < num_rows; num_rows -= gridDim.y) {
    short *out = indices;
    for (int bid = threadIdx.x; bid < blocks_for_this_expert; bid += kThreadsPerBlock) {
      *out = expert_base_offset + bid;
      out += kThreadsPerBlock;
    }
    indices += gridDim.y * blocks_for_this_expert;
  }
}

cudaError_t ConstructIndices(short * __restrict__ indices,
			     int output_block_rows,
			     const int * __restrict__ expert_block_counts,
			     const int * __restrict__ expert_block_offsets,
			     const int * __restrict__ output_offsets,
			     int block_size,
			     const int * __restrict__ padded_bins,
			     int num_bins,
			     cudaStream_t stream) {
  dim3 block_dim(kThreadsPerBlock);
  dim3 grid_dim(num_bins, (int)std::ceil((float)output_block_rows / num_bins));
  ConstructIndicesKernel<<<grid_dim, block_dim, 0, stream>>>(indices,
							     expert_block_counts,
							     expert_block_offsets,
							     output_offsets,
							     block_size,
							     padded_bins);
  return cudaGetLastError();
}

}  // namespace construct_indices

void indices(torch::Tensor padded_bins,
	     torch::Tensor expert_block_counts,
	     torch::Tensor expert_block_offsets,
	     torch::Tensor output_offsets,
	     int block_size,
	     int output_block_rows,
	     torch::Tensor out) {
  TORCH_CHECK(padded_bins.is_cuda());
  TORCH_CHECK(padded_bins.ndimension() == 1);
  TORCH_CHECK(padded_bins.scalar_type() == torch::kInt);

  TORCH_CHECK(expert_block_counts.is_cuda());
  TORCH_CHECK(expert_block_counts.ndimension() == 1);
  TORCH_CHECK(expert_block_counts.scalar_type() == torch::kInt);

  TORCH_CHECK(expert_block_offsets.is_cuda());
  TORCH_CHECK(expert_block_offsets.ndimension() == 1);
  TORCH_CHECK(expert_block_offsets.scalar_type() == torch::kInt);

  TORCH_CHECK(output_offsets.is_cuda());
  TORCH_CHECK(output_offsets.ndimension() == 1);
  TORCH_CHECK(output_offsets.scalar_type() == torch::kInt);

  TORCH_CHECK(out.is_cuda());
  TORCH_CHECK(out.ndimension() == 1);
  TORCH_CHECK(out.scalar_type() == torch::kInt16);

  // Exit early if there is no work to do.
  if (out.numel() == 0) return;

  CUDA_CALL(construct_indices::ConstructIndices(out.data_ptr<short>(),
						output_block_rows,
						expert_block_counts.data_ptr<int>(),
						expert_block_offsets.data_ptr<int>(),
						output_offsets.data_ptr<int>(),
						block_size,
						padded_bins.data_ptr<int>(),
						padded_bins.numel(),
						c10::cuda::getCurrentCUDAStream()));
}

}  // namespace megablocks
