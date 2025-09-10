"""
DSD profiling with Triton Proton (no NVTX).
- proton.start(..., hook="triton")
- activate/deactivate around the hot loop
- optional PyTorch profiler toggle (USE_TORCH_PROF=1)
"""

import os
import torch
# import torch.profiler as profiler
# import triton.profiler as proton  # Proton

from model import GPT, GPTConfig

torch.manual_seed(0)

config = GPTConfig()
# Shakespeare char config values
config.n_ctx = 256
config.n_layer = 6
config.n_head = 6
config.n_embd = 384
config.dropout = 0.2
config.use_moe = True
config.num_experts = 8
config.num_experts_per_tok = 2
config.norm_topk_prob = True
config.block_size = 64
config.block_k = 64
config.vocab_size = 50304  # Keep default vocab size

model = GPT(config).cuda().bfloat16()
# Input tokens for full model (batch_size=4, seq_len=256)
x = torch.randint(0, config.vocab_size, (4, 256), device='cuda')
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
    # Record only inside this region
    # proton.activate(0)              # session index 0
    # try:
    for i in range(repeats):
            # with proton.scope(f"GPT/forward_pass/iter={i}"):
            _ = model(x)[0]
            torch.cuda.synchronize()
    # finally:
        # proton.deactivate(0)

# def run_torch_profiler_once():
#     # with profiler.profile(
#         activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
#         with_stack=True, profile_memory=True, record_shapes=True,
#     ) as prof:
#         _ = model(x)[0]
#         torch.cuda.synchronize()
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
#     prof.export_chrome_trace("dsd_trace.json")
#     print("\nTrace saved to dsd_trace.json (open via chrome://tracing)")

# try:
    # if os.getenv("USE_TORCH_PROF", "0") == "1":
        # run_torch_profiler_once()
    # else:
repeats = int(os.getenv("REPEATS", "1"))
run_proton_loop(repeats=repeats)

# finally:
    # proton.finalize()
    # print(f"[Proton] Wrote {profile_name}.hatchet")
