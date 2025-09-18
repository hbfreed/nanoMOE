import torch
import time
from model import GPT, GPTConfig, MoeMLP

# Monkey patch to add detailed timing
original_create_sparse = MoeMLP._create_sparse_indices

def timed_create_sparse_indices(self, tokens_per_expert_padded):
    """New optimized version with dynamic arange."""
    device = tokens_per_expert_padded.device
    timings = {}

    t0 = time.perf_counter()
    num_token_blocks_per_expert = tokens_per_expert_padded // self.block_size
    blocks_per_expert = num_token_blocks_per_expert * self._num_ffn_blocks
    total_blocks = blocks_per_expert.sum()  # 0-dim GPU tensor
    timings['blocks_and_sum'] = time.perf_counter() - t0

    # dynamic arange on device → no host sync, no mask later
    t0 = time.perf_counter()
    indices = torch.arange(total_blocks, device=device, dtype=torch.long)
    timings['arange'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cumsum = blocks_per_expert.cumsum(0)  # [E]
    timings['cumsum'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    expert_ids = torch.searchsorted(cumsum, indices, right=True).clamp(max=self.num_experts - 1)
    timings['searchsorted_clamp'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cumsum_padded = torch.nn.functional.pad(cumsum[:-1], (1, 0))  # [E]
    timings['pad'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    within_expert_idx = indices - cumsum_padded[expert_ids]
    timings['within_expert_idx'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    within_expert_block = within_expert_idx // self._num_ffn_blocks
    within_expert_ffn  = within_expert_idx %  self._num_ffn_blocks
    timings['div_mod'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    token_block_cumsum = num_token_blocks_per_expert.cumsum(0)
    token_block_offset = torch.nn.functional.pad(token_block_cumsum[:-1], (1, 0))
    timings['token_block_cumsum_pad'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    row_indices         = (token_block_offset[expert_ids] + within_expert_block).to(torch.int32)
    weight_col_indices  = (expert_ids * self._num_ffn_blocks + within_expert_ffn).to(torch.int32)
    output_col_indices  = within_expert_ffn.to(torch.int32)
    timings['compute_and_convert'] = time.perf_counter() - t0

    # Store timings in class for accumulation
    if not hasattr(self, '_timing_stats'):
        self._timing_stats = {k: [] for k in timings}
    for k, v in timings.items():
        self._timing_stats[k].append(v * 1000)  # Convert to ms

    return row_indices, weight_col_indices, output_col_indices

# Apply monkey patch
MoeMLP._create_sparse_indices = timed_create_sparse_indices

# Test setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig(
    n_layer=6,
    n_head=6,
    n_embd=384,
    use_moe=True,
    num_experts=64,
    num_experts_per_tok=8,
    n_ctx=1024,
    vocab_size=65,
    bias=False,
    dropout=0.0
)

model = GPT(config).to(device)

# Create test input
batch_size = 4
seq_len = 256
x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

# Warmup
print("Warming up...")
for _ in range(10):
    logits, loss, _ = model(x, targets)
    loss.backward()
    model.zero_grad()

# Clear timing stats after warmup
for module in model.modules():
    if isinstance(module, MoeMLP):
        module._timing_stats = {k: [] for k in module._timing_stats}

# Run profiling
num_iterations = 100
print(f"\nProfiling {num_iterations} iterations on {device}...")

for i in range(num_iterations):
    logits, loss, _ = model(x, targets)
    loss.backward()
    model.zero_grad()
    if (i + 1) % 20 == 0:
        print(f"  Completed {i + 1}/{num_iterations} iterations...")

# Collect and analyze stats
print("\n=== Detailed timing breakdown for _create_sparse_indices ===")
print("(Average time in milliseconds over all layers and iterations)\n")

all_stats = {}
layer_count = 0
for name, module in model.named_modules():
    if isinstance(module, MoeMLP):
        layer_count += 1
        for op, times in module._timing_stats.items():
            if op not in all_stats:
                all_stats[op] = []
            all_stats[op].extend(times)

# Sort by average time
sorted_stats = sorted(
    [(op, sum(times)/len(times), max(times), min(times), len(times))
     for op, times in all_stats.items()],
    key=lambda x: x[1],
    reverse=True
)

print(f"{'Operation':<30} {'Avg (ms)':>10} {'Max (ms)':>10} {'Min (ms)':>10} {'Count':>8}")
print("-" * 78)
total_avg = 0
for op, avg, max_t, min_t, count in sorted_stats:
    print(f"{op:<30} {avg:>10.4f} {max_t:>10.4f} {min_t:>10.4f} {count:>8}")
    total_avg += avg

print("-" * 78)
print(f"{'TOTAL':<30} {total_avg:>10.4f}")
print(f"\nTotal calls analyzed: {layer_count} layers × {num_iterations} iterations = {layer_count * num_iterations}")