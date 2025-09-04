# config for training MoE GPT-2 (124M parameters but with 8 experts)
# launch as: torchrun --standalone --nproc_per_node=8 train.py config/train_moe_gpt2.py

wandb_log = True
wandb_project = 'moe-owt'
wandb_run_name = 'moe-gpt2-124M'

# Model configuration - same base size as GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# MoE configuration
use_moe = True
num_experts = 8
num_experts_per_tok = 2
norm_topk_prob = True
block_k = 64

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
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

# learning rate - slightly higher for MoE
learning_rate = 6e-4
min_lr = 6e-5