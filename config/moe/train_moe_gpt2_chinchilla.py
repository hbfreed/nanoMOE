# config for training MoE GPT-2 (124M parameters but with 8 experts)
# launch as: torchrun --standalone --nproc_per_node=8 train.py config/moe/train_moe_gpt2_chinchilla.py
# MoE configuration
use_moe = True
num_experts = 64
num_experts_per_tok = 8

out_dir = f"out-chinchilla/moe-{num_experts}total-{num_experts_per_tok}active"

wandb_log = True
wandb_project = "gpt2-chinchilla"
wandb_run_name = f"moe-{num_experts}total-{num_experts_per_tok}active"

# Model configuration - same base size as GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False


norm_topk_prob = True
block_size = 128  # Triton kernel block size
block_k = 64  # Triton kernel K dimension

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size *13 gradaccum * 3 GPUs = 479,232
batch_size = 6
n_ctx = 1024
gradient_accumulation_steps = 13 * 6

# 5217 gets us to roughly 2.5 billion tokens, which is chinchilla optimal for a model of this size (125m * 20 = 2.5 billion)
max_iters = 5217
lr_decay_iters = 5217

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# learning rate - slightly higher for MoE
learning_rate = 6e-4
min_lr = 6e-5
