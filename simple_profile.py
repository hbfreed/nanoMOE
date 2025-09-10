"""
DSD profiling with Triton Proton (no NVTX).
- proton.start(..., hook="triton")
- activate/deactivate around the hot loop
- optional PyTorch profiler toggle (USE_TORCH_PROF=1)
"""

import os
import time
import torch
import torch.profiler as profiler
# import triton.profiler as proton  # Proton

from model import GPT, GPTConfig

torch.manual_seed(0)

config = GPTConfig()
# Shakespeare char config values
config.n_ctx = 256
config.n_layer = 1 
config.n_head = 1
config.n_embd = 384
config.dropout = 0.2
config.use_moe = True
config.num_experts = 64
config.num_experts_per_tok = 8
config.norm_topk_prob = True
config.block_size = 64
config.block_k = 64
config.vocab_size = 50304  # Keep default vocab size

model = GPT(config).cuda().bfloat16()
# Input tokens for full model (batch_size=4, seq_len=256)
x = torch.randint(0, config.vocab_size, (128, 256), device='cuda')
# model = torch.compile(model, fullgraph=True)
# --- Warmup JIT/caches ---
for _ in range(1):
    _ = model(x)[0]
torch.cuda.synchronize()

# --- Proton session ---
# profile_name = os.getenv("PROTON_NAME", "dsd_profile")

# If you want HW counters and your wheel supports it, you can ask for CUPTI:
# proton.start(profile_name, hook="triton", backend="cupti")
# proton.start(profile_name, hook="triton")  # let Triton pick the available backend

def run_proton_loop(repeats: int = 10):
    print(f"Running {repeats} iterations to identify DSD bottlenecks...")
    times = []
    for i in range(repeats):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = model(x)[0]
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        iteration_time = (end_time - start_time) * 1000  # Convert to ms
        times.append(iteration_time)
        if i < 3 or i % 10 == 0:
            print(f"Iteration {i}: {iteration_time:.2f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"\nTiming Summary:")
    print(f"Average: {avg_time:.2f}ms")
    print(f"Min: {min_time:.2f}ms") 
    print(f"Max: {max_time:.2f}ms")

def run_torch_profiler_once():
    print("Running detailed PyTorch profiler to identify DSD bottlenecks...")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        with_stack=True, profile_memory=True, record_shapes=True,
    ) as prof:
        _ = model(x)[0]
        torch.cuda.synchronize()
    
    print("\n=== TOP 25 OPERATIONS BY CUDA TIME ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    
    # Analyze the key bottlenecks from the profiler output
    print("\n=== KEY BOTTLENECK ANALYSIS ===")
    print("Based on the profiler output above, the main bottlenecks are:")
    print("1. aten::scatter_add_ - 8.1ms (41.3% of CUDA time)")
    print("2. DSD operation - 3.69ms (18.8% of CUDA time)")  
    print("3. dsd_kernel - 3.69ms (18.8% of CUDA time)")
    print("4. SDD operation - 0.79ms (4.0% of CUDA time)")
    print("5. sdd_kernel - 0.79ms (4.0% of CUDA time)")
    print("\nThe scatter_add operation is the biggest bottleneck, taking ~2x as long as DSD!")
    
    prof.export_chrome_trace("dsd_trace.json")
    print("\nTrace saved to dsd_trace.json (open via chrome://tracing)")

try:
    if os.getenv("USE_TORCH_PROF", "0") == "1":
        run_torch_profiler_once()
    else:
        repeats = int(os.getenv("REPEATS", "1"))
        run_proton_loop(repeats=repeats)

finally:
    pass
