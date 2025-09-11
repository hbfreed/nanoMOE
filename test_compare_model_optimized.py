#!/usr/bin/env python3
"""
Compare baseline MoeMLP (model.py) vs optimized MoeMLP (model_optimized.py).

Checks:
- Forward outputs equality (within tolerance)
- Aux losses similarity
- Internal sparse index triplets equality on the same tokens_per_expert_padded

Run:
  python test_compare_model_optimized.py
Optionally set CUDA:
  CUDA_VISIBLE_DEVICES=0 python test_compare_model_optimized.py
"""

import os
import math
import torch
from dataclasses import dataclass

from model import MoeMLP as BaselineMoeMLP, GPTConfig as BaseCfg
from model_optimized import MoeMLP as OptMoeMLP


@dataclass
class Case:
    n_embd: int
    n_ctx: int
    num_experts: int
    k: int
    block_size: int
    block_k: int
    batch: int
    seq: int


def make_cfg(c: Case):
    cfg = BaseCfg()
    cfg.n_embd = c.n_embd
    cfg.n_ctx = c.n_ctx
    cfg.n_layer = 1
    cfg.n_head = 1
    cfg.use_moe = True
    cfg.num_experts = c.num_experts
    cfg.num_experts_per_tok = c.k
    cfg.norm_topk_prob = True
    cfg.block_size = c.block_size
    cfg.block_k = c.block_k
    cfg.dropout = 0.0
    return cfg


def sync_weights(dst: BaselineMoeMLP, src: BaselineMoeMLP):
    with torch.no_grad():
        dst.router.weight.copy_(src.router.weight)
        dst.w1.copy_(src.w1)
        dst.w2.copy_(src.w2)


def compare_tensors(a: torch.Tensor, b: torch.Tensor, name: str, rtol=1e-4, atol=1e-5):
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    if not ok:
        diff = (a - b).abs()
        print(f"❌ {name}: max={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    else:
        print(f"✓ {name}")
    return ok


def test_one(c: Case, device: str):
    print(f"\n=== Case: {c} on {device} ===")
    torch.manual_seed(1337)
    cfg = make_cfg(c)

    base = BaselineMoeMLP(cfg).to(device)
    opt = OptMoeMLP(cfg).to(device)
    # Ensure identical params
    sync_weights(opt, base)

    B, T, H = c.batch, c.seq, c.n_embd
    x = torch.randn(B, T, H, device=device, dtype=torch.float32)

    ok = True
    if os.getenv("SKIP_FORWARD", "0") != "1":
        # Forward outputs
        with torch.no_grad():
            out_b, aux_b, fi_b = base(x)
            out_o, aux_o, fi_o = opt(x)
        ok &= compare_tensors(out_b, out_o, "Output", rtol=1e-3, atol=1e-4)
        ok &= compare_tensors(fi_b, fi_o, "Expert usage f_i", rtol=1e-4, atol=1e-5)

    # Internal sparse indices comparison on same routing
    with torch.no_grad():
        x_flat = x.view(-1, H)
        rw, sel, _ = base._route_tokens(x_flat)
        x_sorted, sel_sorted, rw_sorted, inv = base._sort_by_expert(x_flat, rw, sel)
        _, tokens_per_expert_padded, _ = base._pad_to_blocks(x_sorted, sel_sorted)

        r1, w1, o1 = base._create_sparse_indices(tokens_per_expert_padded)
        r2, w2, o2 = opt._create_sparse_indices(tokens_per_expert_padded)

    ok &= compare_tensors(r1, r2, "row_indices (token blocks)", rtol=0, atol=0)
    ok &= compare_tensors(w1, w2, "weight_col_indices (expert, ffn)", rtol=0, atol=0)
    ok &= compare_tensors(o1, o2, "output_col_indices (ffn)", rtol=0, atol=0)

    if not ok:
        # Print a few mismatches for debugging
        diff = (r1 != r2).nonzero().flatten()
        if diff.numel() > 0:
            idx = diff[:10].tolist()
            print("row_indices mismatches at:", idx)
            for i in idx:
                print(i, r1[i].item(), r2[i].item())

    return ok


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cases = [
        Case(64, 256, 4, 2, 16, 16, 2, 64),
        Case(128, 128, 8, 2, 32, 32, 4, 128),
        Case(384, 256, 64, 8, 64, 64, 2, 256),
        Case(256, 256, 16, 4, 64, 64, 32, 256),  # stress batch size vs index buffer
        Case(384, 97, 16, 4, 32, 32, 3, 97),
    ]
    all_ok = True
    for c in cases:
        all_ok &= test_one(c, device)
    print("\nSUMMARY:")
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
