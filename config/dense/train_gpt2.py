# GPT-2 dense model configuration for OpenWebText training
# Train GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# Launch as: torchrun --standalone --nproc_per_node=8 train.py config/dense/train_gpt2.py

dataset = "openwebtext"

# Model configuration - same base size as GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# MoE configuration (disabled for dense)
use_moe = False

out_dir = "out-openwebtext/dense-300B"

wandb_log = True
wandb_project = "owt"
wandb_run_name = "dense-gpt2-124M"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 40 gradaccum * 8 GPUs = 3,932,160
batch_size = 12
n_ctx = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# learning rate
learning_rate = 6e-4
min_lr = 6e-5
