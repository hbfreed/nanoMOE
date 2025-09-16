import argparse
import time
import torch
import cProfile
import pstats
import io
import sys

from model_optimized import MoeMLP as moemlp, GPTConfig as OptimGPTConfig
from model import MoeMLPForLoop as moemlpforloop


def time_forward(module, x, iters=20, warmup=5):
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = module(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return times


def main():
    parser = argparse.ArgumentParser(description="Compare forward speed: MoeMLP vs MoeMLPForLoop")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=384)
    # Experts
    parser.add_argument("--experts", type=int, default=64, help="total experts")
    parser.add_argument("--active-experts", "--topk", dest="active_experts", type=int, default=8,
                        help="active experts per token (top-k)")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--block_k", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-cprofile", action="store_true", help="Disable cProfile profiling")
    parser.add_argument("--profile-sort", type=str, default="tottime", help="Sort order for cProfile stats")
    parser.add_argument("--profile-lines", type=int, default=30, help="Number of lines to print from cProfile stats")
    args = parser.parse_args()

    # cProfile is ON by default unless --no-cprofile is passed
    use_cprofile = not args.no_cprofile

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (optimized MoeMLP allocates CUDA buffers).")

    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    cfg = OptimGPTConfig(
        n_ctx=args.seq,
        n_layer=1,
        n_head=1,
        n_embd=args.hidden,
        dropout=0.0,
        bias=False,
        use_moe=True,
        num_experts=args.experts,
        num_experts_per_tok=args.active_experts,
        norm_topk_prob=True,
        block_size=args.block_size,
        block_k=args.block_k,
        vocab_size=50304,
    )

    x = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)

    loop = moemlpforloop(cfg).to(device=device, dtype=dtype).eval()
    opt = moemlp(cfg).to(device=device, dtype=dtype).eval()
    # opt = torch.compile(opt)

    def run_timing():
        try:
            t_loop = time_forward(loop, x, iters=args.iters, warmup=args.warmup)
        except Exception as e:
            raise SystemExit(f"MoeMLPForLoop failed: {e}")

        t_opt = time_forward(opt, x, iters=args.iters, warmup=args.warmup)

        def show(name, arr):
            if not arr:
                print(f"{name:>16}: failed")
                return
            avg, mn, mx = sum(arr)/len(arr), min(arr), max(arr)
            print(f"{name:>16}: avg {avg:.2f} ms | min {mn:.2f} | max {mx:.2f}")

        print("Forward timing (no backward):")
        show("MoeMLPForLoop", t_loop)
        show("MoeMLP (opt)", t_opt)
        if t_opt:
            print(f"Speedup (loop/opt): {(sum(t_loop)/len(t_loop))/(sum(t_opt)/len(t_opt)):.2f}x")

    if use_cprofile:
        pr = cProfile.Profile()
        pr.enable()
        run_timing()
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(args.profile_sort)
        ps.print_stats(args.profile_lines)
        print("\n==== cProfile stats ====")
        print(s.getvalue())
    else:
        run_timing()


if __name__ == "__main__":
    main()
