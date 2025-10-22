# GPT-2 MoE model configuration for OpenWebText training
# Train to 300B tokens on 1 node of 8X A100 40GB
# Launch as: torchrun --standalone --nproc_per_node=8 train.py config/moe/train_moe_gpt2.py

dataset = "openwebtext"

# Model configuration - same base size as GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# MoE configuration
use_moe = True
num_experts = 8
num_experts_per_tok = 2
norm_topk_prob = True
block_size = 128
block_k = 64
expert_sizes = [(num_experts, n_embd * 4 // num_experts_per_tok)]
load_balance_loss_weight = 0.01

# Create string representation of expert sizes for naming
expert_sizes_str = "-".join([f"{h}x{d}" for h, d in expert_sizes])

out_dir = (
    f"out-openwebtext/moe-{num_experts}x{num_experts_per_tok}-{expert_sizes_str}-300B"
)

wandb_log = True
wandb_project = "owt"
wandb_run_name = f"moe-{num_experts}x{num_experts_per_tok}-{expert_sizes_str}"

# these make the total batch size be ~0.5M
# 10 batch size * 1024 block size * 40 gradaccum * 8 GPUs = 3,276,800
batch_size = 10
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

# learning rate - slightly higher for MoE
learning_rate = 6e-4
min_lr = 6e-5
