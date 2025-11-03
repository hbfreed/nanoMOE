"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# cProfile imports - optional profiling support
import cProfile
import pstats
from io import StringIO

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# profiling
profile_enabled = False  # Set to True to enable cProfile profiling
profile_iterations = -1  # Number of iterations to profile (set to -1 for all)
profile_output = "profile_stats.prof"  # Output file for profiling results
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
n_ctx = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# MoE parameters
use_moe = False  # whether to use Mixture of Experts
num_experts = 8  # number of experts in MoE layer
num_experts_per_tok = 2  # number of experts to route to per token
norm_topk_prob = True  # normalize the top-k probabilities to sum to 1
block_size = 128  # Triton kernel tile size for MoE
block_k = 64  # Triton kernel K dimension for MoE
expert_sizes = None  # list of (count, size) tuples for variable expert sizes, e.g., [(32, 128), (32, 512)]
router_aux_loss_coef = 0.01  # auxiliary loss coefficient for load balancing
load_balance_loss_weight = 0.02  # weight for load balance auxiliary loss
router_z_loss_weight = 0.001  # weight for router z-loss auxiliary loss
compute_loss_weight = 0.0  # weight for compute-based auxiliary loss (minimizes compute by biasing toward smaller experts)
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# random seed
seed = 1337  # random seed for reproducibility
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * n_ctx
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)

# TODO: When we upgrade to higher pytorch versions, switch to this.
# torch.backends.cuda.matmul.fp32_precision = 'tf32' # Use TF32 for better performance
# torch.backends.cudnn.conv.fp32_precision = 'tf32' # Use TF32 for cudnn operations
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", dataset)
print(f"using data_dir:{data_dir}")


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - n_ctx, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + n_ctx]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + n_ctx]).astype(np.int64)) for i in ix]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    n_ctx=n_ctx,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)
# Add MoE parameters if they exist in globals (from config file)
if "use_moe" in globals():
    model_args["use_moe"] = use_moe
    if use_moe:  # Only add other MoE params if MoE is enabled
        model_args["num_experts"] = num_experts
        model_args["num_experts_per_tok"] = num_experts_per_tok
        model_args["norm_topk_prob"] = norm_topk_prob
        model_args["block_size"] = block_size
        model_args["block_k"] = block_k
        model_args["expert_sizes"] = expert_sizes
        model_args["load_balance_loss_weight"] = load_balance_loss_weight
        model_args["router_z_loss_weight"] = router_z_loss_weight
        model_args["compute_loss_weight"] = compute_loss_weight
        print(
            f"MoE enabled with {num_experts} experts, {num_experts_per_tok} experts per token"
        )
        if expert_sizes is not None:
            print(f"Using variable expert sizes: {expert_sizes}")
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "n_ctx", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "n_ctx", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if n_ctx < model.config.n_ctx:
    model.crop_block_size(n_ctx)
    model_args["n_ctx"] = n_ctx  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    # Use fullgraph=True for dense models, default for MoE models
    if "use_moe" in globals() and use_moe:
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model)  # , mode='reduce-overhead')
    else:
        model = torch.compile(model, fullgraph=True)  # Dense models can use fullgraph

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        ce_losses = torch.zeros(eval_iters)
        load_balance_losses = torch.zeros(eval_iters)
        router_z_losses = torch.zeros(eval_iters)
        compute_losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss, aux_losses = model(X, Y)
            if aux_losses is not None:
                # Batch all .item() calls together
                items_to_extract = [
                    loss,
                    aux_losses["load_balance_loss"],
                    aux_losses["router_z_loss"],
                ]
                has_flops = "compute_loss" in aux_losses
                has_ce = "ce_loss" in aux_losses

                if has_flops:
                    items_to_extract.append(aux_losses["compute_loss"])
                if has_ce:
                    items_to_extract.append(aux_losses["ce_loss"])

                extracted = [t.item() for t in items_to_extract]

                # Unpack based on what we have
                idx = 0
                losses[k] = extracted[idx]
                idx += 1
                load_balance_losses[k] = extracted[idx]
                idx += 1
                router_z_losses[k] = extracted[idx]
                idx += 1

                if has_flops:
                    compute_losses[k] = extracted[idx]
                    idx += 1

                if has_ce:
                    ce_losses[k] = extracted[idx]
                else:
                    ce_losses[k] = losses[k]  # For dense models, total loss is CE loss
            else:
                loss_val = loss.item()
                losses[k] = loss_val
                ce_losses[k] = loss_val  # For dense models, total loss is CE loss

        out[split] = losses.mean()
        out[f"{split}_ce"] = ce_losses.mean()
        out[f"{split}_load_balance"] = load_balance_losses.mean()
        out[f"{split}_router_z"] = router_z_losses.mean()
        out[f"{split}_flops"] = compute_losses.mean()

    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
combined_aux_loss = None  # Initialize for first eval at iter 0

# Setup profiling if enabled
profile_iter_count = 0
if profile_enabled and master_process:
    print(f"=== PROFILING ENABLED ===")
    print(
        f"Iterations to profile: {profile_iterations if profile_iterations > 0 else 'all'}"
    )
    print(f"Profile output: {profile_output}")
    profiler = cProfile.Profile()

while True:
    # Check if we should start/stop profiling
    if profile_enabled and master_process:
        # Start profiling at the beginning of the iteration
        if profile_iter_count == 0:
            profiler.enable()

        # Check if we should stop profiling
        if profile_iterations > 0 and profile_iter_count >= profile_iterations:
            if profile_iter_count == profile_iterations:
                profiler.disable()
                # Save profiling results
                profiler.dump_stats(profile_output)
                print(f"=== PROFILING COMPLETE ===")
                print(f"Profiled {profile_iterations} iterations")
                print(f"Results saved to {profile_output}")

                # Print top functions by cumulative time
                print("\n=== TOP 20 FUNCTIONS BY CUMULATIVE TIME ===")
                s = StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                ps.print_stats(20)
                print(s.getvalue())

                # Print top functions by total time
                print("\n=== TOP 20 FUNCTIONS BY TOTAL TIME ===")
                s = StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats("tottime")
                ps.print_stats(20)
                print(s.getvalue())

                # Disable further profiling
                profile_enabled = False

        profile_iter_count += 1

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train CE {losses['train_ce']:.4f}, val CE {losses['val_ce']:.4f}"
        )
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "train/ce_loss": losses["train_ce"],
                "val/ce_loss": losses["val_ce"],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            }
            if "train_load_balance" in losses:
                log_dict["train/load_balance_loss"] = losses["train_load_balance"]
                log_dict["val/load_balance_loss"] = losses["val_load_balance"]
                log_dict["train/router_z_loss"] = losses["train_router_z"]
                log_dict["val/router_z_loss"] = losses["val_router_z"]

            if "train_flops" in losses:
                log_dict["train/compute_loss"] = losses["train_flops"]
                log_dict["val/compute_loss"] = losses["val_flops"]

            # Log step-level auxiliary losses if available
            if combined_aux_loss is not None:
                # Batch extract auxiliary losses
                aux_items = [
                    combined_aux_loss["load_balance_loss"],
                    combined_aux_loss["router_z_loss"],
                ]
                if "compute_loss" in combined_aux_loss:
                    aux_items.append(combined_aux_loss["compute_loss"])

                aux_extracted = [t.item() for t in aux_items]
                log_dict["train/load_balance_loss_step"] = aux_extracted[0]
                log_dict["train/router_z_loss_step"] = aux_extracted[1]

                if "compute_loss" in combined_aux_loss:
                    log_dict["train/compute_loss_step"] = aux_extracted[2]

                if "expert_usage" in combined_aux_loss:
                    expert_usage = combined_aux_loss["expert_usage"]
                    # Batch extract all expert usage values at once
                    expert_usage_list = (
                        expert_usage.tolist()
                    )  # Single GPU sync for all values
                    for i, usage in enumerate(expert_usage_list):
                        log_dict[f"expert_usage/expert_{i}"] = usage

                    # Monitor for expert collapse
                    max_usage = max(expert_usage_list)
                    max_idx = expert_usage_list.index(max_usage)
                    log_dict["expert_usage/max_usage"] = max_usage
                    if max_usage > 0.5:
                        print(
                            f"WARNING: Expert collapse detected! Expert {max_idx} has {max_usage:.1%} of tokens"
                        )

                # Log per-layer expert usage
                if "expert_usage_per_layer" in combined_aux_loss:
                    expert_usage_per_layer = combined_aux_loss["expert_usage_per_layer"]
                    for layer_idx, layer_usage in enumerate(expert_usage_per_layer):
                        layer_usage_list = layer_usage.tolist()
                        for expert_idx, usage in enumerate(layer_usage_list):
                            log_dict[f"layer_{layer_idx}/expert_{expert_idx}"] = usage

            wandb.log(log_dict)
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss, aux_loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
            # Keep the last aux_loss for logging (they should be similar across micro steps)
            if aux_loss is not None:
                combined_aux_loss = aux_loss
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # Batch extract losses to minimize CPU-GPU sync points
        if combined_aux_loss is not None and "ce_loss" in combined_aux_loss:
            # Extract both losses in one go
            loss_val, ce_loss_val = loss.item(), combined_aux_loss["ce_loss"].item()
            lossf = loss_val * gradient_accumulation_steps
            ce_lossf = ce_loss_val  # CE loss is already unscaled

            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )

            print(
                f"iter {iter_num}: loss {lossf:.4f}, ce_loss {ce_lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            )
        else:
            lossf = loss.item() * gradient_accumulation_steps

            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )

            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# Save final checkpoint if we're the master process
if master_process:
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    # Save locally first
    checkpoint_path = os.path.join(out_dir, "ckpt.pt")
    print(f"saving final checkpoint to {out_dir}")
    torch.save(checkpoint, checkpoint_path)

# Save profiling results if we were profiling and didn't finish
if profile_enabled and master_process and profile_iter_count > 0:
    profiler.disable()
    profiler.dump_stats(profile_output)
    print(f"\n=== PROFILING COMPLETE (Training Ended) ===")
    print(f"Profiled {profile_iter_count} iterations")
    print(f"Results saved to {profile_output}")

    # Print summary
    print("\n=== TOP 20 FUNCTIONS BY CUMULATIVE TIME ===")
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

if ddp:
    destroy_process_group()
