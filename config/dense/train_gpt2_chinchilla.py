# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/dense/train_gpt2_chinchilla.py

out_dir = "out-chinchilla/dense-baseline"

wandb_log = True
wandb_project = "gpt2-chinchilla"
wandb_run_name = "dense-baseline"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size *13 gradaccum * 3 GPUs = 479,232
batch_size = 12
n_ctx = 1024
gradient_accumulation_steps = 13 * 3

# 5217 gets us to roughly 2.5 billion tokens, which is chinchilla optimal for a model of this size (125m * 20 = 2.5 billion)
max_iters = 5217
lr_decay_iters = 5217

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# MoE configuration (disabled for standard GPT-2)
use_moe = False
